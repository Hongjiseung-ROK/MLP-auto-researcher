"""Scientific claim registry schema.

Every numerical claim must reference concrete artifacts; agent text is never
evidence (plan.md data-provenance policy).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClaimStatus(StrEnum):
    REGISTERED = "registered"
    VERIFIED = "verified"
    REJECTED = "rejected"


class ClaimClass(StrEnum):
    """Scientific purpose of the claim (plan_phase_2.md §14.1)."""

    INFRASTRUCTURE = "infrastructure_claim"
    PILOT_SCIENTIFIC = "pilot_scientific_claim"
    REPLICATED_SCIENTIFIC = "replicated_scientific_claim"
    PUBLICATION = "publication_claim"


MINTABLE_CLAIM_CLASSES = frozenset(
    {ClaimClass.INFRASTRUCTURE, ClaimClass.PILOT_SCIENTIFIC}
)


class ScientificEvidenceTier(StrEnum):
    """Maturity is separate from artifact integrity/ClaimStatus."""

    NON_SCIENTIFIC = "non_scientific"
    PILOT_ONLY = "pilot_only"
    REPLICATED = "replicated"
    PUBLICATION_ELIGIBLE = "publication_eligible"


class EvaluationPartition(StrEnum):
    INITIAL_LABELED = "initial_labeled"
    ACQUISITION_POOL = "acquisition_pool"
    VALIDATION = "validation"
    FROZEN_TEST = "frozen_test"
    STRESS_TEST = "stress_test"


class MetricValueBinding(BaseModel):
    """Bind a claim value to one exact field in one metric artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_artifact_reference: str = Field(min_length=1)
    json_pointer: str = Field(pattern=r"^/aggregate/[a-z0-9_]+$")


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$")
    statement: str = Field(min_length=1)
    value: float | int | str
    units: str = ""
    created_by_step: str
    artifact_references: list[str] = Field(
        min_length=1, description="Artifact ids in the run manifest that back this claim"
    )
    # Phase 1 claims predate claim classes; they are infrastructure-tier.
    claim_class: ClaimClass = ClaimClass.INFRASTRUCTURE
    scientific_evidence_tier: ScientificEvidenceTier = ScientificEvidenceTier.NON_SCIENTIFIC
    source_run_references: list[str] = Field(default_factory=list)
    metric_artifact_references: list[str] = Field(default_factory=list)
    metric_value_binding: MetricValueBinding | None = None
    dataset_manifest_reference: str | None = None
    split_manifest_reference: str | None = None
    model_manifest_reference: str | None = None
    human_approval_references: list[str] = Field(default_factory=list)
    replicated_claim_references: list[str] = Field(default_factory=list)
    evaluation_partition: EvaluationPartition | None = None
    derived_from_model_metrics: bool = False
    limitations: list[str] = Field(default_factory=list)
    seed_metadata: list[int] = Field(default_factory=list)
    statistical_metadata: dict[str, float | int | str] = Field(default_factory=dict)
    status: ClaimStatus = ClaimStatus.REGISTERED
    rejection_reason: str | None = None
    rejection_reason_category: str | None = None
    rejection_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _scientific_evidence_contract(self) -> Claim:
        expected_tier = {
            ClaimClass.INFRASTRUCTURE: ScientificEvidenceTier.NON_SCIENTIFIC,
            ClaimClass.PILOT_SCIENTIFIC: ScientificEvidenceTier.PILOT_ONLY,
            ClaimClass.REPLICATED_SCIENTIFIC: ScientificEvidenceTier.REPLICATED,
            ClaimClass.PUBLICATION: ScientificEvidenceTier.PUBLICATION_ELIGIBLE,
        }[self.claim_class]
        if self.scientific_evidence_tier is not expected_tier:
            raise ValueError(
                f"{self.claim_class.value} requires evidence tier {expected_tier.value}"
            )

        refs = set(self.artifact_references)
        metric_refs = set(self.metric_artifact_references)
        if not metric_refs.issubset(refs):
            raise ValueError("metric artifact references must also be claim artifact references")
        if (
            self.metric_value_binding is not None
            and self.metric_value_binding.metric_artifact_reference not in metric_refs
        ):
            raise ValueError("metric value binding must reference a declared metric artifact")
        manifest_refs = {
            ref
            for ref in (
                self.dataset_manifest_reference,
                self.split_manifest_reference,
                self.model_manifest_reference,
            )
            if ref is not None
        }
        if not manifest_refs.issubset(refs):
            raise ValueError("dataset/split/model manifest references must back the claim")
        if not set(self.human_approval_references).issubset(refs):
            raise ValueError("human approval references must back the claim")

        is_scientific = self.claim_class is not ClaimClass.INFRASTRUCTURE
        if is_scientific and self.dataset_manifest_reference is None:
            raise ValueError("scientific claims require a dataset manifest reference")
        if is_scientific and isinstance(self.value, (int, float)) and not metric_refs:
            raise ValueError("numerical scientific claims require metric artifact references")
        if (
            is_scientific
            and isinstance(self.value, (int, float))
            and self.metric_value_binding is None
        ):
            raise ValueError("numerical scientific claims require an exact metric value binding")
        if (
            self.evaluation_partition is EvaluationPartition.FROZEN_TEST
            and self.split_manifest_reference is None
        ):
            raise ValueError("frozen-test claims require the split manifest")
        if self.derived_from_model_metrics and self.model_manifest_reference is None:
            raise ValueError("model-derived claims require the model manifest")

        if (
            self.claim_class is ClaimClass.REPLICATED_SCIENTIFIC
            and len(set(self.source_run_references)) < 2
        ):
            raise ValueError("replicated claims require at least two source runs")
        if self.claim_class is ClaimClass.PUBLICATION:
            if len(set(self.source_run_references)) < 2:
                raise ValueError("publication claims require replicated source runs")
            if not self.replicated_claim_references:
                raise ValueError("publication claims require replicated claim references")
            if not self.human_approval_references:
                raise ValueError("publication claims require human approval references")
        return self
