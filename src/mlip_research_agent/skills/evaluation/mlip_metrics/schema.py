"""Typed artifacts for leakage-safe MLIP metric evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

METRICS_SCHEMA_VERSION = "2.0.0"

EvaluationPartition = Literal["validation", "frozen_test", "stress_test"]


class MLIPMetricsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    predictions_artifact: str = Field(min_length=1)
    dataset_artifact: str = Field(min_length=1)
    dataset_manifest_artifact: str = Field(min_length=1)
    split_manifest_artifact: str = Field(min_length=1)
    model_manifest_artifact: str = Field(min_length=1)
    evaluation_authorization_artifact: str | None = None
    partition: EvaluationPartition
    high_error_threshold_ev_per_a: float = Field(gt=0)

    @model_validator(mode="after")
    def _protected_partition_authorization(self) -> MLIPMetricsInput:
        protected = self.partition in {"frozen_test", "stress_test"}
        if protected and self.evaluation_authorization_artifact is None:
            raise ValueError("protected evaluation requires an authorization artifact")
        if not protected and self.evaluation_authorization_artifact is not None:
            raise ValueError("validation evaluation cannot consume a protected authorization")
        return self


class EvaluationAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    authorization_id: str = Field(min_length=1)
    partition: Literal["frozen_test", "stress_test"]
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    predictions_artifact: str
    model_manifest_artifact: str
    purpose: Literal["final_aggregate_evaluation"] = "final_aggregate_evaluation"
    approved_by: str = Field(min_length=1)


class ModelCheckpointReference(BaseModel):
    model_config = ConfigDict(extra="allow")

    sha256: str = Field(min_length=64, max_length=64)


class EvaluationModelManifest(BaseModel):
    """Minimum evaluator-facing subset of a registered model manifest."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    checkpoint: ModelCheckpointReference
    predictions_artifact: str = Field(min_length=1)
    structure_set_sha256: str = Field(min_length=64, max_length=64)


class AggregateMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    n_structures: int = Field(gt=0)
    n_atoms: int = Field(gt=0)
    energy_mae_ev_per_atom: float = Field(ge=0)
    force_component_mae_ev_per_a: float = Field(ge=0)
    force_component_rmse_ev_per_a: float = Field(ge=0)
    force_vector_error_p95_ev_per_a: float = Field(ge=0)
    high_error_structure_fraction: float = Field(ge=0, le=1)


class GroupMetrics(AggregateMetrics):
    group_name: str = Field(min_length=1)


class InputArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    sha256: str = Field(min_length=64, max_length=64)
    kind: str


class ClaimReferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_artifact_references: list[str]
    dataset_manifest_reference: str
    split_manifest_reference: str
    model_manifest_reference: str
    required_artifact_references: list[str]
    metric_value_paths: dict[str, str]
    evaluation_partition: EvaluationPartition
    derived_from_model_metrics: bool = True


class MetricsArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["2.0.0"] = "2.0.0"
    metric_algorithm: Literal["mlip_metrics_v1"] = "mlip_metrics_v1"
    evaluation_code_sha256: str = Field(min_length=64, max_length=64)
    partition: EvaluationPartition
    dataset_id: str
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    model_id: str
    checkpoint_sha256: str = Field(min_length=64, max_length=64)
    units: dict[str, str]
    high_error_threshold_ev_per_a: float = Field(gt=0)
    metric_definitions: dict[str, str]
    input_artifacts: dict[str, InputArtifactReference]
    aggregate: AggregateMetrics
    groups: list[GroupMetrics]
    grouping_field: Literal["top_group"] = "top_group"
    minimum_group_size: int = Field(default=3, ge=2)
    suppressed_group_structure_count: int = Field(ge=0)
    evaluation_authorization_reference: str | None = None
    claim_reference_metadata: ClaimReferenceMetadata


class MLIPMetricsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics_artifact: str
    metrics_path: str
    partition: EvaluationPartition
    aggregate: AggregateMetrics
    n_groups: int = Field(ge=0)


class BoundedValidationRequest(BaseModel):
    """Least-privilege WP5 request for the infrastructure-only replay."""

    model_config = ConfigDict(extra="forbid")

    predictions_artifact: str
    predictions_rerun_artifact: str
    model_manifest_artifact: str
    bounded_view_sha256: str = Field(min_length=64, max_length=64)
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    high_error_threshold_ev_per_a: float = Field(gt=0)
    partition: Literal["validation"] = "validation"


class BoundedValidationMetricsArtifact(BaseModel):
    """Aggregate-only WP5 evidence with exact input identities."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["1.0.0"] = "1.0.0"
    boundary: Literal["mlip_metrics/bounded_validation/1.0.0"] = (
        "mlip_metrics/bounded_validation/1.0.0"
    )
    evaluation_code_sha256: str = Field(min_length=64, max_length=64)
    partition: Literal["validation"] = "validation"
    bounded_view_sha256: str = Field(min_length=64, max_length=64)
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    model_id: str
    checkpoint_sha256: str = Field(min_length=64, max_length=64)
    prediction_sha256: str = Field(min_length=64, max_length=64)
    prediction_rerun_sha256: str = Field(min_length=64, max_length=64)
    model_manifest_sha256: str = Field(min_length=64, max_length=64)
    input_artifact_ids: list[str] = Field(min_length=3)
    aggregate: AggregateMetrics
    rerun_metric_delta: float = Field(ge=0)
    units: dict[str, str]
    scientific_status: Literal["infrastructure_only"] = "infrastructure_only"
    claim_eligible: Literal[False] = False


class LearningCurvePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels_available: int = Field(ge=0)
    metrics_artifact: str


class LearningCurveSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm: str
    points: list[LearningCurvePoint] = Field(min_length=2)

    @model_validator(mode="after")
    def _increasing_budget(self) -> LearningCurveSpec:
        budgets = [point.labels_available for point in self.points]
        if budgets != sorted(set(budgets)):
            raise ValueError("learning-curve label budgets must be strictly increasing")
        return self


class PostRevealCalibrationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection_artifact: str
    revealed_label_batch_artifact: str
    uncertainty_scores_artifact: str
    metric: Literal["spearman_disagreement_vs_error"] = "spearman_disagreement_vs_error"


class ArmComparisonSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_arm: str
    right_arm: str
    left_metrics_artifact: str
    right_metrics_artifact: str
    equal_label_budget: int = Field(ge=0)
