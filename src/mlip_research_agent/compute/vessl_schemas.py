"""Sealed records for the current VESSL Cloud ``vesslctl`` execution plane."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from mlip_research_agent.research.auto_research.validators import ContentAddressedModel

SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"


class VesslRecord(ContentAddressedModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VesslCliVersion(VesslRecord):
    executable: Literal["vesslctl"] = "vesslctl"
    version: str = Field(min_length=1)


class VesslAccountContext(VesslRecord):
    authenticated: bool
    organization: str = Field(min_length=1)
    team: str = Field(min_length=1)
    api_host: str = Field(min_length=1)
    principal_present: bool


class VesslBillingSnapshot(VesslRecord):
    organization: str = Field(min_length=1)
    currency: str = Field(min_length=3, max_length=3)
    current_credit: float = Field(ge=0)
    current_burn_rate_per_hour: float = Field(ge=0)
    observed_at: datetime
    raw_payload_sha256: str = Field(pattern=SHA256_PATTERN)


class VesslResourceSpec(VesslRecord):
    slug: str = Field(min_length=1)
    cluster: str = Field(min_length=1)
    gpu_type: str = Field(min_length=1)
    gpu_count: int = Field(ge=0)
    hourly_price: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    usable: bool
    raw_payload_sha256: str = Field(pattern=SHA256_PATTERN)


class VesslStorageExposure(VesslRecord):
    storage_type: Literal["none", "object", "cluster"]
    storage_slug: str | None = None
    volume_slug: str | None = None
    capacity_gb: float = Field(ge=0)
    hourly_rate: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    active: bool

    @model_validator(mode="after")
    def _consistent(self) -> VesslStorageExposure:
        if self.storage_type == "none" and any(
            value is not None for value in (self.storage_slug, self.volume_slug)
        ):
            raise ValueError("storage_type none cannot bind storage or volume slugs")
        if self.storage_type != "none" and not self.storage_slug:
            raise ValueError("persistent storage exposure requires a storage slug")
        return self


class VesslVolumeMount(VesslRecord):
    volume_type: Literal["object", "cluster"]
    volume_slug: str = Field(min_length=1)
    mount_path: str = Field(pattern=r"^/[A-Za-z0-9._/-]+$")
    read_only: bool = False


class VesslCostCard(VesslRecord):
    organization: str = Field(min_length=1)
    team: str = Field(min_length=1)
    cluster: str = Field(min_length=1)
    resource_spec_slug: str = Field(min_length=1)
    gpu_type: str = Field(min_length=1)
    gpu_count: int = Field(gt=0)
    current_hourly_price: float = Field(gt=0)
    current_credit: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    image: str = Field(min_length=1)
    expected_max_duration_minutes: int = Field(gt=0)
    estimated_compute_cost: float = Field(ge=0)
    storage_type: Literal["none", "object", "cluster"]
    storage_capacity_gb: float = Field(ge=0)
    storage_hourly_rate: float = Field(ge=0)
    storage_duration_hours: float = Field(ge=0)
    estimated_storage_cost: float = Field(ge=0)
    volume_mounts: list[VesslVolumeMount]
    timeout_behavior: str = Field(min_length=5)
    cleanup_action: str = Field(min_length=5)
    live_snapshot_sha256: str = Field(pattern=SHA256_PATTERN)
    observed_at: datetime

    @model_validator(mode="after")
    def _costs_and_gpu(self) -> VesslCostCard:
        if "a100" not in self.gpu_type.casefold() or self.gpu_count != 1:
            raise ValueError("MLIP replay requires exactly one live A100")
        expected_compute = self.current_hourly_price * (
            self.expected_max_duration_minutes / 60
        )
        if abs(self.estimated_compute_cost - expected_compute) > 1.0e-9:
            raise ValueError("estimated compute cost does not match price and duration")
        expected_storage = self.storage_hourly_rate * self.storage_duration_hours
        if abs(self.estimated_storage_cost - expected_storage) > 1.0e-9:
            raise ValueError("estimated storage cost does not match rate and duration")
        if self.storage_type == "none" and (
            self.storage_capacity_gb != 0
            or self.storage_hourly_rate != 0
            or self.estimated_storage_cost != 0
            or self.volume_mounts
        ):
            raise ValueError("storage-free cost card cannot declare storage exposure")
        return self


class VesslCostApproval(VesslRecord):
    cost_card_sha256: str = Field(pattern=SHA256_PATTERN)
    approval_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    approved_by: str = Field(min_length=1)
    approved_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def _window(self) -> VesslCostApproval:
        if self.expires_at <= self.approved_at:
            raise ValueError("cost approval expiry must follow approval time")
        return self


class VesslJobRequest(VesslRecord):
    phase: Literal["iteration_1", "iteration_2"]
    organization: str = Field(min_length=1)
    team: str = Field(min_length=1)
    cluster: str = Field(min_length=1)
    resource_spec_slug: str = Field(min_length=1)
    gpu_type: str = Field(min_length=1)
    gpu_count: Literal[1] = 1
    image: str = Field(min_length=1)
    expected_max_duration_minutes: int = Field(gt=0)
    volume_mounts: list[VesslVolumeMount]
    timeout_behavior: str = Field(min_length=5)
    cleanup_action: str = Field(min_length=5)
    git_commit: str = Field(pattern=COMMIT_PATTERN)
    job_config_sha256: str = Field(pattern=SHA256_PATTERN)
    job_config_schema_verified: bool
    bounded_label_view_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    split_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_lock_sha256: str = Field(pattern=SHA256_PATTERN)
    iteration_1_receipt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    review_bundle_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    scientific_status: Literal["infrastructure_only"] = "infrastructure_only"
    claim_eligible: Literal[False] = False
    checkpoint_status: Literal["candidate_only"] = "candidate_only"

    @model_validator(mode="after")
    def _phase_requirements(self) -> VesslJobRequest:
        if "a100" not in self.gpu_type.casefold():
            raise ValueError("VESSL MLIP Jobs require an A100 resource")
        if self.phase == "iteration_1" and any(
            value is not None
            for value in (self.iteration_1_receipt_sha256, self.review_bundle_sha256)
        ):
            raise ValueError("Job 1 cannot consume prior iteration or review receipts")
        if self.phase == "iteration_2" and (
            self.iteration_1_receipt_sha256 is None
            or self.review_bundle_sha256 is None
        ):
            raise ValueError("Job 2 requires sealed Job 1 and review bundle receipts")
        return self


class VesslOptimizerReceipt(VesslRecord):
    phase: Literal["iteration_1", "iteration_2"]
    job_request_sha256: str = Field(pattern=SHA256_PATTERN)
    operation_id: str = Field(min_length=1)
    optimizer_steps: Literal[1] = 1
    artifact_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    bounded_label_view_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    split_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    scientific_status: Literal["infrastructure_only"] = "infrastructure_only"
    claim_eligible: Literal[False] = False


class VesslReviewBundleReceipt(VesslRecord):
    iteration_1_receipt_sha256: str = Field(pattern=SHA256_PATTERN)
    mlip_scientist_sha256: str = Field(pattern=SHA256_PATTERN)
    active_learning_scientist_sha256: str = Field(pattern=SHA256_PATTERN)
    scientific_auditor_sha256: str = Field(pattern=SHA256_PATTERN)
    synthesis_sha256: str = Field(pattern=SHA256_PATTERN)
    proposal_2_sha256: str = Field(pattern=SHA256_PATTERN)
    claim_eligible: Literal[False] = False


class VesslJobIdentity(VesslRecord):
    job_slug: str = Field(min_length=1)
    phase: Literal["iteration_1", "iteration_2"]
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=COMMIT_PATTERN)
    resource_spec_slug: str = Field(min_length=1)
    image: str = Field(min_length=1)


class VesslJobStatus(VesslRecord):
    job_slug: str = Field(min_length=1)
    state: Literal[
        "pending", "running", "succeeded", "failed", "cancelled", "terminated"
    ]
    raw_payload_sha256: str = Field(pattern=SHA256_PATTERN)

    @property
    def terminal(self) -> bool:
        return self.state in {"succeeded", "failed", "cancelled", "terminated"}


class VesslJobAttestation(VesslRecord):
    job_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    observed_gpu_type: str = Field(min_length=1)
    observed_gpu_count: Literal[1] = 1
    observed_git_commit: str = Field(pattern=COMMIT_PATTERN)
    detached_checkout: bool
    clean_checkout: bool
    environment_lock_sha256: str = Field(pattern=SHA256_PATTERN)
    bounded_label_view_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _attested(self) -> VesslJobAttestation:
        if "a100" not in self.observed_gpu_type.casefold():
            raise ValueError("observed VESSL GPU is not A100")
        if not self.detached_checkout or not self.clean_checkout:
            raise ValueError("VESSL Job must use a clean detached exact checkout")
        return self


class VesslCleanupRecord(VesslRecord):
    job_statuses: list[VesslJobStatus]
    surviving_storage: list[VesslStorageExposure]
    checked_at: datetime

    @model_validator(mode="after")
    def _terminal_jobs(self) -> VesslCleanupRecord:
        if not self.job_statuses or not all(status.terminal for status in self.job_statuses):
            raise ValueError("cleanup record requires every VESSL Job to be terminal")
        return self


class VesslArtifactPullback(VesslRecord):
    job_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    file_count: int = Field(ge=0)
    staging_directory: str = Field(min_length=1)
    published_directory: str = Field(min_length=1)
    all_hashes_verified: Literal[True] = True
    unexpected_files: list[str] = Field(default_factory=list, max_length=0)
