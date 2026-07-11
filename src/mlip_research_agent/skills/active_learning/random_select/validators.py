"""Validation for the random-selection arm: label-free, budget-exact."""

from __future__ import annotations

from mlip_research_agent.skills.active_learning.random_select.schema import RandomSelectInput
from mlip_research_agent.skills.active_learning.selection_types import (
    CandidateMeta,
    reject,
    validate_candidate_pool,
)


def validate_inputs(params: RandomSelectInput) -> None:
    validate_candidate_pool(params.pool_candidate_ids, params.metadata)
    missing = sorted(set(params.pool_candidate_ids) - set(params.metadata))
    if missing:
        raise reject(f"pool ids without metadata: {missing[:5]}")
    if params.budget > len(params.pool_candidate_ids):
        raise reject(
            f"budget {params.budget} exceeds pool size {len(params.pool_candidate_ids)}; "
            "exact budget enforcement forbids over-selection"
        )


def validate_selection(
    selected_ids: list[str],
    params: RandomSelectInput,
    metadata: dict[str, CandidateMeta],
) -> None:
    if len(selected_ids) != params.budget:
        raise reject(
            f"selected {len(selected_ids)} candidates but the budget is exactly {params.budget}"
        )
    if len(set(selected_ids)) != len(selected_ids):
        raise reject("duplicate candidates in the selection")
    outside = sorted(set(selected_ids) - set(params.pool_candidate_ids))
    if outside:
        raise reject(f"selection escaped the pool: {outside[:5]}")
    for candidate_id in selected_ids:
        if candidate_id not in metadata:
            raise reject(f"selected candidate {candidate_id} has no metadata")
