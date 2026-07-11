"""Local CPU provider: synchronous subprocess execution for unit-scale work."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

from mlip_research_agent.compute.base import manifest_from_directory, new_run_id, write_job_record
from mlip_research_agent.compute.policy import ComputePolicy
from mlip_research_agent.compute.router import redact_secrets
from mlip_research_agent.compute.schemas import (
    AcceleratorId,
    ArtifactManifest,
    AuthStatus,
    CancellationResult,
    CapacityReport,
    JobHandle,
    JobSpec,
    JobState,
    JobStatus,
    LogEvent,
    ProviderName,
)


class LocalProvider:
    """CPU-only provider for deterministic unit tests and development."""

    name = ProviderName.LOCAL

    def __init__(self, policy: ComputePolicy, artifacts_root: Path) -> None:
        self.policy = policy
        self.artifacts_root = artifacts_root
        self._jobs: dict[str, JobStatus] = {}
        self._logs: dict[str, list[str]] = {}
        self._run_dirs: dict[str, Path] = {}

    def validate_auth(self) -> AuthStatus:
        return AuthStatus(
            provider=self.name, authenticated=True, method="local-process", detail="no auth needed"
        )

    def inspect_capacity(self) -> CapacityReport:
        provider = self.policy.provider(self.name)
        return CapacityReport(
            provider=self.name,
            available_accelerators=list(provider.allowed_accelerators),
            max_gpus_per_run=provider.max_gpus_per_run,
            notes="host CPU, synchronous execution",
        )

    def submit(self, spec: JobSpec) -> JobHandle:
        run_id = new_run_id(spec)
        run_dir = self.artifacts_root / "compute" / run_id
        work_dir = run_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        self._run_dirs[run_id] = run_dir

        decision = self.policy.evaluate(self.name, spec, observed_gpu_name=AcceleratorId.CPU.value)
        if not decision.allowed:
            status = JobStatus(
                job_id=run_id, state=JobState.POLICY_DENIED, message="; ".join(decision.reasons)
            )
            self._finish(run_id, status, [])
            return JobHandle(provider=self.name, job_id=run_id, run_id=run_id)

        try:
            proc = subprocess.run(
                spec.command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=spec.max_runtime_minutes * 60,
            )
            lines = (proc.stdout + proc.stderr).splitlines()
            state = JobState.SUCCEEDED if proc.returncode == 0 else JobState.FAILED
            status = JobStatus(job_id=run_id, state=state, exit_code=proc.returncode)
        except subprocess.TimeoutExpired:
            lines = ["local job exceeded max_runtime_minutes"]
            status = JobStatus(job_id=run_id, state=JobState.FAILED, message="timeout")
        except OSError as exc:
            lines = [f"failed to start process: {exc}"]
            status = JobStatus(job_id=run_id, state=JobState.FAILED, message=str(exc))
        self._finish(run_id, status, lines)
        return JobHandle(provider=self.name, job_id=run_id, run_id=run_id)

    def _finish(self, run_id: str, status: JobStatus, lines: list[str]) -> None:
        self._jobs[run_id] = status
        self._logs[run_id] = [redact_secrets(line) for line in lines]
        write_job_record(
            self._run_dirs[run_id], "status.json", status.model_dump(mode="json")
        )

    def status(self, handle: JobHandle) -> JobStatus:
        return self._jobs[handle.job_id]

    def stream_logs(self, handle: JobHandle) -> Iterable[LogEvent]:
        for i, line in enumerate(self._logs.get(handle.job_id, [])):
            yield LogEvent(sequence=i, message=line)

    def collect_artifacts(self, handle: JobHandle) -> ArtifactManifest:
        return manifest_from_directory(handle.job_id, self._run_dirs[handle.job_id] / "work")

    def cancel(self, handle: JobHandle) -> CancellationResult:
        return CancellationResult(
            job_id=handle.job_id, cancelled=False, message="synchronous local job already finished"
        )
