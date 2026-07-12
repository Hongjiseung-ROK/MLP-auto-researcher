"""Independent aggregate-only evaluator for the bounded MACE replay."""

from __future__ import annotations

import json
from pathlib import Path

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.data.bounded_view import BoundedLabelView
from mlip_research_agent.research.auto_research.evaluation import (
    BaselineReference,
    ConstraintResult,
    EvaluationOutcome,
)
from mlip_research_agent.research.auto_research.mutation import LegalityResult
from mlip_research_agent.skills.evaluation.mlip_metrics.implementation import (
    evaluate_bounded_validation,
)
from mlip_research_agent.skills.evaluation.mlip_metrics.schema import (
    BoundedValidationRequest,
)

EVALUATOR_NAME = "mlip_validation_evaluator/1.0.0"


class MACEValidationEvaluator:
    def __init__(self, label_view_path: Path) -> None:
        self.label_view_path = label_view_path

    @property
    def name(self) -> str:
        return EVALUATOR_NAME

    def evaluate(
        self,
        *,
        evaluation_id: str,
        execution_id: str,
        predictions_path: Path,
        predictions_rerun_path: Path,
        prediction_artifact_ids: list[str],
        metric_artifact_ids: list[str],
        legality: LegalityResult,
        baseline: BaselineReference,
        resource_metrics: dict[str, float],
        constraints: dict[str, float],
        metrics_out_path: Path,
    ) -> EvaluationOutcome:
        view = BoundedLabelView.load(self.label_view_path)
        model_manifest_path = predictions_path.parent / "model_manifest.json"
        predictions_payload = json.loads(predictions_path.read_text())
        wp5 = evaluate_bounded_validation(
            request=BoundedValidationRequest(
                predictions_artifact=prediction_artifact_ids[0],
                predictions_rerun_artifact=prediction_artifact_ids[1],
                model_manifest_artifact=str(predictions_payload["model_manifest_artifact"]),
                bounded_view_sha256=sha256_file(self.label_view_path),
                dataset_content_sha256=view.source_dataset_content_sha256,
                split_semantic_sha256=view.split_semantic_sha256,
                high_error_threshold_ev_per_a=constraints.get("tail_max", 1.0),
            ),
            bounded_view_path=self.label_view_path,
            predictions_path=predictions_path,
            predictions_rerun_path=predictions_rerun_path,
            model_manifest_path=model_manifest_path,
            output_path=metrics_out_path,
        )
        aggregate = wp5.aggregate
        metrics = {key: float(value) for key, value in aggregate.model_dump().items()}
        metrics["rerun_metric_delta"] = wp5.rerun_metric_delta
        results = [
            ConstraintResult(
                constraint_id="tail-max",
                passed=aggregate.force_vector_error_p95_ev_per_a <= constraints["tail_max"],
                observed=aggregate.force_vector_error_p95_ev_per_a,
                limit=constraints["tail_max"],
                description="Validation force-vector p95 stays within the infrastructure limit.",
            ),
            ConstraintResult(
                constraint_id="runtime-max",
                passed=resource_metrics["wall_seconds"] <= constraints["runtime_max_seconds"],
                observed=resource_metrics["wall_seconds"],
                limit=constraints["runtime_max_seconds"],
                description="Execution stays within the bounded runtime.",
            ),
            ConstraintResult(
                constraint_id="memory-max",
                passed=resource_metrics["peak_memory_mb"] <= constraints["memory_max_mb"],
                observed=resource_metrics["peak_memory_mb"],
                limit=constraints["memory_max_mb"],
                description="Execution stays within the bounded memory limit.",
            ),
        ]
        return EvaluationOutcome(
            evaluation_id=evaluation_id,
            execution_id=execution_id,
            evaluator_name=self.name,
            evaluator_source_sha256=sha256_file(Path(__file__)),
            input_prediction_artifacts=prediction_artifact_ids,
            metric_artifacts=metric_artifact_ids,
            legality_result_sha256=legality.content_sha256,
            aggregate_metrics=metrics,
            constraint_results=results,
            baseline_reference=baseline,
            resource_metrics=resource_metrics,
            uncertainty_notes=(
                "Validation aggregates support infrastructure decisions only; no calibrated "
                "uncertainty or protected-partition result is produced."
            ),
            scientific_status="infrastructure_only",
        ).sealed()
