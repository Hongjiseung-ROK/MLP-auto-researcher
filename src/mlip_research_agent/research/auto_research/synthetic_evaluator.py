"""Independent aggregate evaluator for the synthetic fixture.

This module is deliberately separate from both the controller and the
fixture: its identity is the SHA-256 of this source file, recorded in every
:class:`EvaluationOutcome`. It reads prediction artifacts, computes aggregate
metrics, and returns aggregates only — per-record residuals never reach the
controller. The metric and thresholds are explicitly synthetic; nothing here
is an H2 preregistration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.research.auto_research.evaluation import (
    BaselineReference,
    ConstraintResult,
    EvaluationOutcome,
)
from mlip_research_agent.research.auto_research.mutation import LegalityResult

EVALUATOR_NAME = "synthetic_aggregate_evaluator/1.0.0"

#: Synthetic demo constraints — deliberately not scientific thresholds.
SYNTHETIC_PRIMARY_METRIC = "synthetic_validation_loss"
SYNTHETIC_TAIL_METRIC = "synthetic_p95_residual"


def evaluator_source_sha256() -> str:
    """Content hash of this evaluator's own source: pins the exact judge."""
    return sha256_file(Path(__file__).resolve())


def _load_residuals(path: Path) -> list[float]:
    raw = json.loads(path.read_text())
    residuals = raw.get("record_residuals")
    if not isinstance(residuals, dict) or not residuals:
        raise ValueError(f"prediction artifact {path} carries no residuals")
    values = [float(v) for v in residuals.values()]
    if not all(np.isfinite(values)):
        raise ValueError(f"prediction artifact {path} contains non-finite residuals")
    return values


class SyntheticAggregateEvaluator:
    """Evaluator boundary object handed to the controller (identity-pinned)."""

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
        return evaluate(
            evaluation_id=evaluation_id,
            execution_id=execution_id,
            predictions_path=predictions_path,
            predictions_rerun_path=predictions_rerun_path,
            prediction_artifact_ids=prediction_artifact_ids,
            metric_artifact_ids=metric_artifact_ids,
            legality=legality,
            baseline=baseline,
            resource_metrics=resource_metrics,
            constraints=constraints,
            metrics_out_path=metrics_out_path,
        )


def evaluate(
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
    """Aggregate the per-record residuals and render the typed outcome.

    ``constraints`` carries the synthetic limits (``tail_max``,
    ``runtime_max_seconds``, ``memory_max_mb``) used for the constraint
    verdicts recorded here; the acceptance policy re-checks them
    independently when deciding.
    """
    residuals = _load_residuals(predictions_path)
    rerun_residuals = _load_residuals(predictions_rerun_path)

    mean_loss = float(np.mean(residuals))
    p95 = float(np.quantile(residuals, 0.95))
    rerun_delta = abs(mean_loss - float(np.mean(rerun_residuals)))

    aggregate_metrics = {
        SYNTHETIC_PRIMARY_METRIC: mean_loss,
        SYNTHETIC_TAIL_METRIC: p95,
        "rerun_metric_delta": rerun_delta,
        "n_records": float(len(residuals)),
    }
    constraint_results = [
        ConstraintResult(
            constraint_id="tail-max",
            passed=p95 <= constraints["tail_max"],
            observed=p95,
            limit=constraints["tail_max"],
            description="Synthetic p95 residual must stay under the demo tail limit.",
        ),
        ConstraintResult(
            constraint_id="runtime-max",
            passed=resource_metrics.get("wall_seconds", 0.0)
            <= constraints["runtime_max_seconds"],
            observed=resource_metrics.get("wall_seconds", 0.0),
            limit=constraints["runtime_max_seconds"],
            description="Simulated runtime must stay under the demo runtime limit.",
        ),
        ConstraintResult(
            constraint_id="memory-max",
            passed=resource_metrics.get("peak_memory_mb", 0.0)
            <= constraints["memory_max_mb"],
            observed=resource_metrics.get("peak_memory_mb", 0.0),
            limit=constraints["memory_max_mb"],
            description="Simulated peak memory must stay under the demo memory limit.",
        ),
    ]

    metrics_payload = {
        "schema_version": "1.0.0",
        "evaluator_name": EVALUATOR_NAME,
        "aggregate_metrics": aggregate_metrics,
        "scientific_status": "non_scientific",
    }
    metrics_out_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n"
    )

    return EvaluationOutcome(
        evaluation_id=evaluation_id,
        execution_id=execution_id,
        evaluator_name=EVALUATOR_NAME,
        evaluator_source_sha256=evaluator_source_sha256(),
        input_prediction_artifacts=prediction_artifact_ids,
        metric_artifacts=metric_artifact_ids,
        legality_result_sha256=legality.content_sha256,
        aggregate_metrics=aggregate_metrics,
        constraint_results=constraint_results,
        baseline_reference=baseline,
        resource_metrics=resource_metrics,
        uncertainty_notes=(
            "Synthetic fixture metrics: the residual spread comes from a seeded "
            "deterministic noise model, so run-to-run uncertainty is zero by "
            "construction and these numbers support no scientific claim."
        ),
        scientific_status="non_scientific",
    ).sealed()
