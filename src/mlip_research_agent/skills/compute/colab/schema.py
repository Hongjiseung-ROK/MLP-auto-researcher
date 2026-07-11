"""Typed inputs/outputs for the colab_preflight skill."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mlip_research_agent.compute.schemas import AcceleratorId, JobState, WorkloadClass

VALIDATION_WORKLOADS = (
    WorkloadClass.GPU_SMOKE_TEST,
    WorkloadClass.PREFLIGHT,
    WorkloadClass.FAILURE_REPRODUCTION,
)


class ColabPreflightInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workload_class: WorkloadClass = WorkloadClass.PREFLIGHT
    requested_accelerator: AcceleratorId = AcceleratorId.NVIDIA_L4
    command: list[str] = Field(
        default_factory=lambda: [
            "python",
            "scripts/colab/preflight.py",
            "--output-root",
            "artifacts/colab",
        ]
    )
    max_runtime_minutes: int = Field(default=30, gt=0, le=120)
    repo_commit: str = Field(default="0000000mock", min_length=7)
    policy_path: str = "configs/compute_policy.yaml"
    use_mock: bool = Field(
        default=True,
        description="Mock transport (offline). Real transport requires the reviewed "
        "colab CLI with external ADC auth; see security.md.",
    )
    mock_observed_gpu: str = Field(
        default="NVIDIA L4",
        description="Device name the mock runtime reports; lets tests exercise denials",
    )


class ColabPreflightOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_state: JobState
    canonical_accelerator: AcceleratorId | None
    attestation_artifact: str
    result_artifact: str
    preflight_record_saved: bool
    n_remote_artifacts: int = Field(ge=0)
