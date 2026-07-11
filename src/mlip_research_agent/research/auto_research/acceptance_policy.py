"""Typed, independent acceptance policy (the controller never improvises).

Decision order (first match wins):

1. illegal mutation                      → reject
2. approved bounded failure repair       → refine
3. unrecoverable execution failure       → escalate
4. metric improves, all constraints pass → accept
5. metric improves, a constraint fails   → reject
6. metric regresses                      → reject
(pivot_request is issued only by an explicit scientific-method-change flag,
never inferred from metrics — and stays owner-gated downstream.)

For the local demonstration the validation metric and thresholds are
explicitly synthetic; nothing here is an H2 preregistration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mlip_research_agent.research.auto_research.decision import (
    DecisionValue,
    DimensionAssessment,
    ExperimentDecision,
)
from mlip_research_agent.research.auto_research.evaluation import EvaluationOutcome
from mlip_research_agent.research.auto_research.execution import ExperimentExecution
from mlip_research_agent.research.auto_research.mutation import LegalityResult

ACCEPTANCE_POLICY_VERSION = "acceptance-policy/1.0.0"


class AcceptanceConstraints(BaseModel):
    """Hard constraints; the primary metric alone can never force an accept."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_metric: str = Field(min_length=1)
    minimize: bool = True
    min_relative_improvement: float = Field(gt=0.0, lt=1.0)
    tail_metric: str = Field(min_length=1)
    tail_max: float
    runtime_metric: str = "wall_seconds"
    runtime_max_seconds: float = Field(gt=0)
    memory_metric: str = "peak_memory_mb"
    memory_max_mb: float = Field(gt=0)
    reproducibility_metric: str = "rerun_metric_delta"
    reproducibility_max_delta: float = Field(ge=0)


def _dim(passed: bool, detail: str) -> DimensionAssessment:
    return DimensionAssessment(passed=passed, detail=detail)


def decide(
    *,
    decision_id: str,
    proposal_id: str,
    legality: LegalityResult,
    execution: ExperimentExecution | None,
    evaluation: EvaluationOutcome | None,
    constraints: AcceptanceConstraints,
    baseline_metrics: dict[str, float],
    repair_available: bool,
    compute_budget_ok: bool,
) -> ExperimentDecision:
    """Produce the typed decision. Pure function of its inputs."""

    # 1. Illegal mutation → reject, before anything else is consulted.
    if not legality.legal:
        detail = "; ".join(legality.violations)[:900]
        return ExperimentDecision(
            decision_id=decision_id,
            proposal_id=proposal_id,
            evaluation_id=None,
            decision=DecisionValue.REJECT,
            acceptance_policy_version=ACCEPTANCE_POLICY_VERSION,
            legality=_dim(False, f"illegal mutation: {detail}"),
            scientific_objective=_dim(True, "not reached"),
            engineering_validity=_dim(True, "not reached"),
            metric_result=_dim(True, "not reached"),
            tail_risk=_dim(True, "not reached"),
            compute_budget=_dim(compute_budget_ok, "budget state at decision time"),
            reproducibility=_dim(True, "not reached"),
            uncertainty=_dim(True, "not reached"),
            rationale="Mutation policy verdict is illegal; the proposal never executed.",
        ).sealed()

    # 2/3. Execution failed.
    if execution is None or not execution.succeeded:
        failure_ref = execution.failure_artifact if execution is not None else None
        if repair_available:
            return ExperimentDecision(
                decision_id=decision_id,
                proposal_id=proposal_id,
                evaluation_id=None,
                decision=DecisionValue.REFINE,
                acceptance_policy_version=ACCEPTANCE_POLICY_VERSION,
                legality=_dim(True, "mutation was legal"),
                scientific_objective=_dim(True, "unchanged"),
                engineering_validity=_dim(
                    False, f"execution failed with approved bounded repair ({failure_ref})"
                ),
                metric_result=_dim(True, "not reached"),
                tail_risk=_dim(True, "not reached"),
                compute_budget=_dim(compute_budget_ok, "budget state at decision time"),
                reproducibility=_dim(True, "not reached"),
                uncertainty=_dim(True, "not reached"),
                rationale="Execution failed and an approved bounded repair exists: refine.",
            ).sealed()
        return ExperimentDecision(
            decision_id=decision_id,
            proposal_id=proposal_id,
            evaluation_id=None,
            decision=DecisionValue.ESCALATE,
            acceptance_policy_version=ACCEPTANCE_POLICY_VERSION,
            legality=_dim(True, "mutation was legal"),
            scientific_objective=_dim(True, "unchanged"),
            engineering_validity=_dim(
                False, f"unrecoverable execution failure ({failure_ref})"
            ),
            metric_result=_dim(True, "not reached"),
            tail_risk=_dim(True, "not reached"),
            compute_budget=_dim(compute_budget_ok, "budget state at decision time"),
            reproducibility=_dim(True, "not reached"),
            uncertainty=_dim(True, "not reached"),
            rationale="Execution failed with no approved bounded repair: escalate.",
        ).sealed()

    if evaluation is None:
        raise ValueError("a successful execution requires an evaluation before deciding")

    metrics = evaluation.aggregate_metrics
    primary = metrics.get(constraints.primary_metric)
    baseline = baseline_metrics.get(constraints.primary_metric)
    if primary is None or baseline is None:
        raise ValueError(
            f"primary metric {constraints.primary_metric!r} missing from evaluation or baseline"
        )

    if constraints.minimize:
        relative_improvement = (baseline - primary) / abs(baseline) if baseline != 0 else 0.0
    else:
        relative_improvement = (primary - baseline) / abs(baseline) if baseline != 0 else 0.0
    improved = relative_improvement >= constraints.min_relative_improvement

    tail_observed = metrics.get(constraints.tail_metric)
    tail_ok = tail_observed is not None and tail_observed <= constraints.tail_max
    runtime_observed = evaluation.resource_metrics.get(constraints.runtime_metric, 0.0)
    runtime_ok = runtime_observed <= constraints.runtime_max_seconds
    memory_observed = evaluation.resource_metrics.get(constraints.memory_metric, 0.0)
    memory_ok = memory_observed <= constraints.memory_max_mb
    rerun_delta = metrics.get(constraints.reproducibility_metric)
    repro_ok = rerun_delta is not None and rerun_delta <= constraints.reproducibility_max_delta

    constraints_ok = tail_ok and runtime_ok and memory_ok and repro_ok and compute_budget_ok

    if improved and constraints_ok:
        value = DecisionValue.ACCEPT
        rationale = (
            f"Primary metric improved by {relative_improvement:.4f} relative "
            f"(threshold {constraints.min_relative_improvement}) and every constraint passed."
        )
    elif improved:
        value = DecisionValue.REJECT
        rationale = "Primary metric improved but at least one hard constraint failed."
    else:
        value = DecisionValue.REJECT
        rationale = (
            f"Primary metric did not improve enough: relative change {relative_improvement:.4f} "
            f"< threshold {constraints.min_relative_improvement}."
        )

    return ExperimentDecision(
        decision_id=decision_id,
        proposal_id=proposal_id,
        evaluation_id=evaluation.evaluation_id,
        decision=value,
        acceptance_policy_version=ACCEPTANCE_POLICY_VERSION,
        legality=_dim(True, "mutation was legal under the recorded policy"),
        scientific_objective=_dim(True, "objective unchanged; no method pivot involved"),
        engineering_validity=_dim(True, "execution completed and produced valid artifacts"),
        metric_result=_dim(
            improved,
            f"{constraints.primary_metric}: {primary:.6f} vs baseline {baseline:.6f} "
            f"(relative improvement {relative_improvement:.4f})",
        ),
        tail_risk=_dim(
            tail_ok,
            f"{constraints.tail_metric}: {tail_observed} (max {constraints.tail_max})",
        ),
        compute_budget=_dim(
            compute_budget_ok and runtime_ok and memory_ok,
            f"runtime {runtime_observed:.2f}s (max {constraints.runtime_max_seconds}s), "
            f"memory {memory_observed:.1f}MB (max {constraints.memory_max_mb}MB)",
        ),
        reproducibility=_dim(
            repro_ok,
            f"{constraints.reproducibility_metric}: {rerun_delta} "
            f"(max {constraints.reproducibility_max_delta})",
        ),
        uncertainty=_dim(
            True,
            f"evaluator notes recorded: {evaluation.uncertainty_notes[:200]}",
        ),
        rationale=rationale,
    ).sealed()
