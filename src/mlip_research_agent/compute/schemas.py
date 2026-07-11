"""Typed compute primitives shared by all providers."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ProviderName(StrEnum):
    LOCAL = "local"
    COLAB = "colab"
    VESSL = "vessl"


class AcceleratorId(StrEnum):
    """Canonical accelerator ids. Anything not representable here is denied."""

    CPU = "cpu"
    NVIDIA_L4 = "nvidia-l4"
    NVIDIA_A100_40GB = "nvidia-a100-40gb"
    NVIDIA_A100_80GB = "nvidia-a100-80gb"


class WorkloadClass(StrEnum):
    UNIT_TEST = "unit_test"
    GPU_SMOKE_TEST = "gpu_smoke_test"
    PREFLIGHT = "preflight"
    FAILURE_REPRODUCTION = "failure_reproduction"
    DEVELOPMENT_TRAINING = "development_training"
    PRIMARY_TRAINING = "primary_training"
    ACTIVE_LEARNING = "active_learning"
    BENCHMARK = "benchmark"
    PRODUCTION_RESEARCH = "production_research"


PRODUCTION_WORKLOADS = frozenset(
    {
        WorkloadClass.PRIMARY_TRAINING,
        WorkloadClass.ACTIVE_LEARNING,
        WorkloadClass.BENCHMARK,
        WorkloadClass.PRODUCTION_RESEARCH,
    }
)


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    POLICY_DENIED = "policy_denied"
    INTERRUPTED = "interrupted"


class AuthStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    authenticated: bool
    method: str | None = None
    detail: str = Field(default="", description="Human guidance; never contains secrets")


class CapacityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    available_accelerators: list[AcceleratorId]
    max_gpus_per_run: int = Field(ge=0)
    notes: str = ""


class JobSpec(BaseModel):
    """Provider-neutral job description submitted by scientific SKILLs."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    workload_class: WorkloadClass
    requested_accelerator: AcceleratorId
    n_gpus: int = Field(default=1, ge=0, le=8)
    command: list[str] = Field(min_length=1)
    max_runtime_minutes: int = Field(gt=0, le=7 * 24 * 60)
    repo_commit: str = Field(min_length=7, description="Exact commit SHA, never a branch")
    dependency_hash: str = ""
    mlip_backend: str = "none"
    cuda_compat_class: str = "none"
    workflow_schema_version: str = "0.1.0"
    provider_override: ProviderName | None = None
    description: str = ""


class JobHandle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    job_id: str
    run_id: str


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    state: JobState
    message: str = ""
    exit_code: int | None = None


class LogEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    stream: str = "stdout"
    message: str


class RemoteArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    size_bytes: int = Field(ge=0)


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    artifacts: list[RemoteArtifact] = Field(default_factory=list)


class CancellationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    cancelled: bool
    message: str = ""


class GpuObservation(BaseModel):
    """What the runtime actually reports, before any policy interpretation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    uuid: str | None = None
    memory_total_mb: int | None = Field(default=None, ge=0)
    driver_version: str | None = None
    cuda_version: str | None = None
    torch_version: str | None = None
    compute_capability: str | None = None
