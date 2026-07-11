"""Shared machinery for remote GPU providers (Colab, VESSL).

The lifecycle every remote job follows:

    policy pre-check -> start session -> runtime attestation (immutable)
      -> ALLOW: execute bounded workload -> collect artifacts -> stop session
      -> DENY:  stop session, POLICY_DENIED, the workload never runs

A job may enter RUNNING only after its attestation receives ALLOW.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Protocol

from mlip_research_agent.compute.attestation import perform_attestation, write_attestation
from mlip_research_agent.compute.base import (
    ComputeError,
    manifest_from_directory,
    new_run_id,
    write_job_record,
)
from mlip_research_agent.compute.policy import ComputePolicy, Decision
from mlip_research_agent.compute.router import redact_secrets
from mlip_research_agent.compute.schemas import (
    ArtifactManifest,
    AuthStatus,
    CancellationResult,
    CapacityReport,
    GpuObservation,
    JobHandle,
    JobSpec,
    JobState,
    JobStatus,
    LogEvent,
    ProviderName,
)


class RemoteTransport(Protocol):
    """Raw session operations against a remote runtime. Auth stays external."""

    def auth_status(self) -> tuple[bool, str, str]:
        """(authenticated, method, detail) — detail never contains secrets."""
        ...

    def start_session(self, spec: JobSpec) -> GpuObservation | None:
        """Allocate a runtime and report the device actually assigned."""
        ...

    def execute(self, spec: JobSpec, run_dir: Path) -> tuple[int, list[str], bool]:
        """Run the bounded workload. Returns (exit_code, log_lines, interrupted)."""
        ...

    def stop_session(self) -> None: ...


class MockRemoteTransport:
    """Deterministic transport simulation for tests and offline development."""

    def __init__(
        self,
        assigned_gpu: GpuObservation | None,
        exit_code: int = 0,
        interrupt: bool = False,
        produced_files: dict[str, str] | None = None,
        log_lines: list[str] | None = None,
        method: str = "mock",
    ) -> None:
        self.assigned_gpu = assigned_gpu
        self.exit_code = exit_code
        self.interrupt = interrupt
        self.produced_files = produced_files or {"result.json": "{\"ok\": true}\n"}
        self.log_lines = log_lines or ["mock workload started", "mock workload finished"]
        self.method = method
        self.sessions_started = 0
        self.sessions_stopped = 0

    def auth_status(self) -> tuple[bool, str, str]:
        return True, self.method, "mock transport, no real credentials involved"

    def start_session(self, spec: JobSpec) -> GpuObservation | None:
        self.sessions_started += 1
        return self.assigned_gpu

    def execute(self, spec: JobSpec, run_dir: Path) -> tuple[int, list[str], bool]:
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Files written before an interruption survive, mirroring a real
        # session that dies mid-run: recovery collects partial artifacts.
        for name, content in self.produced_files.items():
            (artifacts_dir / name).write_text(content)
        if self.interrupt:
            return 1, [*self.log_lines, "session interrupted"], True
        return self.exit_code, list(self.log_lines), False

    def stop_session(self) -> None:
        self.sessions_stopped += 1


class RemoteProviderBase:
    """ComputeProvider implementation shared by Colab and VESSL adapters."""

    name: ClassVar[ProviderName]
    auth_guidance: ClassVar[str]

    def __init__(
        self,
        policy: ComputePolicy,
        artifacts_root: Path,
        transport: RemoteTransport | None = None,
    ) -> None:
        self.policy = policy
        self.artifacts_root = artifacts_root
        self.transport = transport
        self._jobs: dict[str, JobStatus] = {}
        self._logs: dict[str, list[str]] = {}
        self._run_dirs: dict[str, Path] = {}

    def validate_auth(self) -> AuthStatus:
        if self.transport is None:
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                detail=self.auth_guidance,
            )
        authenticated, method, detail = self.transport.auth_status()
        return AuthStatus(
            provider=self.name, authenticated=authenticated, method=method, detail=detail
        )

    def inspect_capacity(self) -> CapacityReport:
        provider = self.policy.provider(self.name)
        return CapacityReport(
            provider=self.name,
            available_accelerators=list(provider.allowed_accelerators),
            max_gpus_per_run=provider.max_gpus_per_run,
            notes="; ".join(provider.purpose),
        )

    def submit(self, spec: JobSpec) -> JobHandle:
        if self.transport is None:
            raise ComputeError(
                f"{self.name.value} transport not configured. {self.auth_guidance}"
            )
        run_id = new_run_id(spec)
        run_dir = self.artifacts_root / "compute" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dirs[run_id] = run_dir
        handle = JobHandle(provider=self.name, job_id=run_id, run_id=run_id)

        # Pre-check everything that is knowable before a device exists; the
        # attestation requirement is deferred to the attestation itself.
        pre = self.policy.evaluate(self.name, spec, observed_gpu_name=None)
        binding = [r for r in pre.reasons if "runtime attestation" not in r]
        if binding:
            self._finish(run_id, JobState.POLICY_DENIED, message="; ".join(binding), lines=[])
            return handle

        observation = self.transport.start_session(spec)
        try:
            record = perform_attestation(
                self.policy, self.name, spec, observation, run_id=run_id
            )
            write_attestation(record, self.artifacts_root)
            if record.policy_decision is not Decision.ALLOW:
                self._finish(
                    run_id,
                    JobState.POLICY_DENIED,
                    message="; ".join(record.policy_reasons),
                    lines=[f"attestation denied: {r}" for r in record.policy_reasons],
                )
                return handle
            # Attestation ALLOW: only now may the workload run.
            self._jobs[run_id] = JobStatus(job_id=run_id, state=JobState.RUNNING)
            exit_code, lines, interrupted = self.transport.execute(spec, run_dir)
            if interrupted:
                state, message = JobState.INTERRUPTED, "session interrupted; partial artifacts kept"
            elif exit_code == 0:
                state, message = JobState.SUCCEEDED, ""
            else:
                state, message = JobState.FAILED, f"exit code {exit_code}"
            self._finish(run_id, state, message=message, lines=lines, exit_code=exit_code)
        finally:
            self.transport.stop_session()
        return handle

    def _finish(
        self,
        run_id: str,
        state: JobState,
        *,
        message: str,
        lines: list[str],
        exit_code: int | None = None,
    ) -> None:
        status = JobStatus(job_id=run_id, state=state, message=message, exit_code=exit_code)
        self._jobs[run_id] = status
        self._logs[run_id] = [redact_secrets(line) for line in lines]
        run_dir = self._run_dirs[run_id]
        write_job_record(run_dir, "status.json", status.model_dump(mode="json"))
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "job.log").write_text("\n".join(self._logs[run_id]) + "\n")

    def status(self, handle: JobHandle) -> JobStatus:
        return self._jobs[handle.job_id]

    def stream_logs(self, handle: JobHandle) -> Iterable[LogEvent]:
        for i, line in enumerate(self._logs.get(handle.job_id, [])):
            yield LogEvent(sequence=i, message=line)

    def collect_artifacts(self, handle: JobHandle) -> ArtifactManifest:
        return manifest_from_directory(
            handle.job_id, self._run_dirs[handle.job_id] / "artifacts"
        )

    def cancel(self, handle: JobHandle) -> CancellationResult:
        status = self._jobs.get(handle.job_id)
        if status is not None and status.state is JobState.RUNNING:
            self._finish(handle.job_id, JobState.CANCELLED, message="cancelled", lines=[])
            return CancellationResult(job_id=handle.job_id, cancelled=True)
        return CancellationResult(
            job_id=handle.job_id, cancelled=False, message="job is not running"
        )
