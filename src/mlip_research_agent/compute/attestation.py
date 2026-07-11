"""Runtime GPU attestation: verify the device actually assigned, fail closed.

A job may enter RUNNING only after its attestation receives an ALLOW decision.
Attestation records are immutable once written.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mlip_research_agent.compute.policy import ComputePolicy, Decision, PolicyDecision
from mlip_research_agent.compute.schemas import (
    AcceleratorId,
    GpuObservation,
    JobSpec,
    ProviderName,
    WorkloadClass,
)

ATTESTATION_FILENAME = "attestation.json"


class AttestationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1.0"
    run_id: str
    provider: ProviderName
    workload_class: WorkloadClass
    requested_accelerator: AcceleratorId
    observed_gpu_name: str | None
    canonical_accelerator: AcceleratorId | None
    gpu_uuid: str | None
    vram_mb: int | None
    cuda_runtime: str | None
    driver_version: str | None
    torch_version: str | None
    policy_decision: Decision
    policy_reasons: list[str] = Field(default_factory=list)
    started_at: str
    repo_commit: str
    environment_hash: str
    config_hash: str


def probe_nvidia_smi() -> GpuObservation | None:
    """Query the first visible GPU via nvidia-smi; None when no GPU is visible."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,uuid,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    fields = [f.strip() for f in proc.stdout.strip().splitlines()[0].split(",")]
    if len(fields) < 5:
        return None
    observation = GpuObservation(
        name=fields[0],
        uuid=fields[1],
        memory_total_mb=int(float(fields[2])),
        driver_version=fields[3],
        compute_capability=fields[4],
    )
    return _augment_with_torch(observation)


def _augment_with_torch(observation: GpuObservation) -> GpuObservation:
    try:
        import torch
    except ImportError:
        return observation
    update: dict[str, str | None] = {"torch_version": str(torch.__version__)}
    if torch.cuda.is_available():
        update["cuda_version"] = str(torch.version.cuda)
    return observation.model_copy(update=update)


def perform_attestation(
    policy: ComputePolicy,
    provider: ProviderName,
    spec: JobSpec,
    observation: GpuObservation | None,
    *,
    run_id: str,
    environment_hash: str = "",
    config_hash: str = "",
) -> AttestationRecord:
    """Evaluate the observed device against policy and build the record."""
    observed_name = observation.name if observation is not None else None
    decision: PolicyDecision = policy.evaluate(provider, spec, observed_name)
    return AttestationRecord(
        run_id=run_id,
        provider=provider,
        workload_class=spec.workload_class,
        requested_accelerator=spec.requested_accelerator,
        observed_gpu_name=observed_name,
        canonical_accelerator=decision.canonical_accelerator,
        gpu_uuid=observation.uuid if observation else None,
        vram_mb=observation.memory_total_mb if observation else None,
        cuda_runtime=observation.cuda_version if observation else None,
        driver_version=observation.driver_version if observation else None,
        torch_version=observation.torch_version if observation else None,
        policy_decision=decision.decision,
        policy_reasons=decision.reasons,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        repo_commit=spec.repo_commit,
        environment_hash=environment_hash,
        config_hash=config_hash,
    )


def write_attestation(record: AttestationRecord, artifacts_root: Path) -> Path:
    """Write the immutable attestation record; refuses to overwrite."""
    run_dir = artifacts_root / "compute" / record.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / ATTESTATION_FILENAME
    if path.exists():
        raise FileExistsError(f"attestation is immutable, refusing to overwrite: {path}")
    path.write_text(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return path
