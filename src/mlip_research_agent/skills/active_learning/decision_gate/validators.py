"""Validation for the decision gate: signal/pool/descriptor coherence."""

from __future__ import annotations

import math

from mlip_research_agent.skills.active_learning.decision_gate.schema import DecisionGateInput
from mlip_research_agent.skills.active_learning.selection_types import (
    reject,
    validate_candidate_pool,
)


def validate_inputs(params: DecisionGateInput) -> None:
    validate_candidate_pool(params.pool_candidate_ids, params.metadata)
    missing_meta = sorted(set(params.pool_candidate_ids) - set(params.metadata))
    if missing_meta:
        raise reject(f"pool ids without metadata: {missing_meta[:5]}")
    pool = set(params.pool_candidate_ids)
    signal_ids = {s.candidate_id for s in params.signals}
    if signal_ids != pool:
        missing = sorted(pool - signal_ids)
        extra = sorted(signal_ids - pool)
        raise reject(
            f"signals do not match the pool (missing {missing[:3]}, extra {extra[:3]})"
        )
    for signal in params.signals:
        if not signal.invalid and not math.isfinite(signal.force_disagreement):
            raise reject(
                f"candidate {signal.candidate_id}: non-finite disagreement must be "
                "flagged invalid, not ranked"
            )
    descriptor_ids = set(params.descriptors.vectors)
    if not pool <= descriptor_ids:
        missing = sorted(pool - descriptor_ids)
        raise reject(f"descriptors missing for pool candidates: {missing[:3]}")
