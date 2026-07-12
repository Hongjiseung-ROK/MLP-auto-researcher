"""Fail-closed validators for label-free force-tail scoring."""

from __future__ import annotations

from mlip_research_agent.skills.active_learning.ensemble_uq.validators import (
    validate_inputs as validate_ensemble_inputs,
)
from mlip_research_agent.skills.active_learning.tail_risk_score.schema import (
    TailRiskScoreInput,
)


def validate_inputs(params: TailRiskScoreInput) -> None:
    """Reuse the committee coverage and hidden-field boundary from ensemble UQ."""
    validate_ensemble_inputs(params)  # type: ignore[arg-type]
