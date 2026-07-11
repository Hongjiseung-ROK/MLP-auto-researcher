"""Validation for ensemble-disagreement inputs: identity, shape, label freedom."""

from __future__ import annotations

import numpy as np

from mlip_research_agent.skills.active_learning.ensemble_uq.schema import EnsembleUQInput
from mlip_research_agent.skills.active_learning.selection_types import (
    reject,
    validate_candidate_pool,
)


def validate_inputs(params: EnsembleUQInput) -> None:
    validate_candidate_pool(params.pool_candidate_ids, params.metadata)
    missing_meta = sorted(set(params.pool_candidate_ids) - set(params.metadata))
    if missing_meta:
        raise reject(f"pool ids without metadata: {missing_meta[:5]}")
    pool = set(params.pool_candidate_ids)
    for member in params.members:
        covered = set(member.predicted_forces)
        if covered != pool:
            missing = sorted(pool - covered)
            extra = sorted(covered - pool)
            raise reject(
                f"member {member.member_id}: predictions do not match the pool "
                f"(missing {missing[:3]}, extra {extra[:3]})"
            )
        if params.include_energy_disagreement and set(member.predicted_energy_per_atom) != pool:
            raise reject(
                f"member {member.member_id}: energy-per-atom predictions required for "
                "every pool candidate when energy disagreement is enabled"
            )


def force_array(
    member_id: str, candidate_id: str, forces: list[list[float]], n_atoms: int
) -> np.ndarray:
    arr = np.asarray(forces, dtype=float)
    if arr.ndim != 2 or arr.shape != (n_atoms, 3):
        raise reject(
            f"member {member_id}, candidate {candidate_id}: force shape {arr.shape} "
            f"!= ({n_atoms}, 3)"
        )
    return arr
