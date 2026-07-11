"""Validation for diversity selection: descriptor integrity, fail-closed sources."""

from __future__ import annotations

import numpy as np

from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.active_learning.diversity_select.schema import (
    DiversitySelectInput,
)
from mlip_research_agent.skills.active_learning.selection_types import (
    reject,
    validate_candidate_pool,
)
from mlip_research_agent.skills.base import SkillError


def validate_inputs(params: DiversitySelectInput) -> None:
    if params.descriptors.source == "mace_descriptor_adapter":
        raise SkillError(
            "the MACE descriptor adapter is a declared future boundary and is not "
            "implemented; fixture descriptors are the only supported source",
            failure_class=FailureClass.TOOL_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    validate_candidate_pool(params.pool_candidate_ids, params.metadata)
    missing_meta = sorted(set(params.pool_candidate_ids) - set(params.metadata))
    if missing_meta:
        raise reject(f"pool ids without metadata: {missing_meta[:5]}")
    if params.budget > len(params.pool_candidate_ids):
        raise reject(
            f"budget {params.budget} exceeds pool size {len(params.pool_candidate_ids)}"
        )
    pool = set(params.pool_candidate_ids)
    covered = set(params.descriptors.vectors)
    if covered != pool:
        missing = sorted(pool - covered)
        extra = sorted(covered - pool)
        raise reject(
            f"descriptors do not match the pool (missing {missing[:3]}, extra {extra[:3]})"
        )
    dim = params.descriptors.dimension
    seen: dict[tuple[float, ...], str] = {}
    for candidate_id in sorted(pool):
        vector = params.descriptors.vectors[candidate_id]
        if len(vector) != dim:
            raise reject(
                f"candidate {candidate_id}: descriptor length {len(vector)} != {dim}"
            )
        arr = np.asarray(vector, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise reject(f"candidate {candidate_id}: non-finite descriptor components")
        key = tuple(float(x) for x in vector)
        if key in seen:
            raise reject(
                f"candidates {seen[key]} and {candidate_id} share an identical "
                "descriptor; duplicate descriptors are rejected"
            )
        seen[key] = candidate_id
