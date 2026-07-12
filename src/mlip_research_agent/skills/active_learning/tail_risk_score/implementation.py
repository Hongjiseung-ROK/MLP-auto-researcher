"""CVaR-like force-tail acquisition score from committee predictions only."""

from __future__ import annotations

import json
import math

import numpy as np
from pydantic import BaseModel

from mlip_research_agent.skills.active_learning.ensemble_uq.validators import force_array
from mlip_research_agent.skills.active_learning.tail_risk_score.schema import (
    POLICY_VERSION,
    TailRiskScoreInput,
    TailRiskScoreOutput,
)
from mlip_research_agent.skills.active_learning.tail_risk_score.validators import (
    validate_inputs,
)
from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill

SCORE_FILENAME = "force_tail_proxy.json"


def force_tail_proxy(
    member_forces: np.ndarray,
    *,
    tail_fraction: float,
    magnitude_weight: float,
    disagreement_weight: float,
) -> tuple[float, int, np.ndarray, np.ndarray, np.ndarray]:
    """Return score and atomic components for an ``(M, N, 3)`` force array.

    Member disagreement is the population RMS Euclidean deviation from the
    committee mean. The candidate score is the mean of the largest
    ``ceil(tail_fraction * N)`` weighted atomic proxy values.
    """
    if member_forces.ndim != 3 or member_forces.shape[0] < 3 or member_forces.shape[2] != 3:
        raise ValueError("member forces must have shape (at least 3, n_atoms, 3)")
    if not np.all(np.isfinite(member_forces)):
        raise ValueError("member forces must be finite")
    mean_forces = member_forces.mean(axis=0)
    mean_magnitude = np.linalg.norm(mean_forces, axis=1)
    squared_vector_deviation = np.sum((member_forces - mean_forces) ** 2, axis=2)
    disagreement = np.sqrt(squared_vector_deviation.mean(axis=0))
    atomic_proxy = magnitude_weight * mean_magnitude + disagreement_weight * disagreement
    count = max(1, math.ceil(tail_fraction * atomic_proxy.size))
    largest = np.partition(atomic_proxy, atomic_proxy.size - count)[-count:]
    return (
        float(largest.mean()),
        count,
        atomic_proxy,
        mean_magnitude,
        disagreement,
    )


@register_skill
class TailRiskScoreSkill(Skill):
    name = "tail_risk_score"
    input_model = TailRiskScoreInput
    output_model = TailRiskScoreOutput
    cost_class = "cheap"
    permission_level = "auto"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, TailRiskScoreInput)
        validate_inputs(params)

        records: list[dict[str, float | int | str]] = []
        invalid_member_flags: dict[str, list[str]] = {}
        for candidate_id in sorted(params.pool_candidate_ids):
            n_atoms = params.metadata[candidate_id].n_atoms
            arrays: list[np.ndarray] = []
            invalid: list[str] = []
            for member in params.members:
                array = force_array(
                    member.member_id,
                    candidate_id,
                    member.predicted_forces[candidate_id],
                    n_atoms,
                )
                if not np.all(np.isfinite(array)):
                    invalid.append(member.member_id)
                else:
                    arrays.append(array)
            if invalid:
                invalid_member_flags[candidate_id] = sorted(set(invalid))
                continue
            score, count, proxy, magnitude, disagreement = force_tail_proxy(
                np.stack(arrays),
                tail_fraction=params.tail_fraction,
                magnitude_weight=params.magnitude_weight,
                disagreement_weight=params.disagreement_weight,
            )
            records.append(
                {
                    "candidate_id": candidate_id,
                    "force_tail_proxy_ev_per_a": score,
                    "tail_atom_count": count,
                    "mean_atomic_proxy_ev_per_a": float(proxy.mean()),
                    "max_atomic_proxy_ev_per_a": float(proxy.max()),
                    "mean_predicted_force_magnitude_ev_per_a": float(magnitude.mean()),
                    "mean_population_disagreement_ev_per_a": float(disagreement.mean()),
                    "source_group": params.metadata[candidate_id].source_group,
                }
            )

        records.sort(
            key=lambda record: (
                -float(record["force_tail_proxy_ev_per_a"]),
                str(record["candidate_id"]),
            )
        )
        for rank, record in enumerate(records):
            record["rank"] = rank

        payload = {
            "schema_version": "1.0.0",
            "campaign_id": params.campaign_id,
            "round_id": params.round_id,
            "policy_version": POLICY_VERSION,
            "signal_kind": "force_tail_proxy",
            "signal_caveat": (
                "This label-free score combines predicted force magnitude and committee "
                "disagreement; it is neither observed error nor calibrated uncertainty."
            ),
            "formula": {
                "population_ddof": 0,
                "tail_fraction": params.tail_fraction,
                "magnitude_weight": params.magnitude_weight,
                "disagreement_weight": params.disagreement_weight,
                "aggregation": "mean_of_largest_ceil_tail_fraction_times_n_atoms",
            },
            "member_identities": [
                {
                    "member_id": member.member_id,
                    "provenance_sha256": member.provenance_sha256,
                }
                for member in params.members
            ],
            "invalid_member_flags": invalid_member_flags,
            "records": records,
        }
        path = ctx.step_dir / SCORE_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        artifact = ctx.registry.register(path, kind="selection_signal", step_id=ctx.step_id)
        return TailRiskScoreOutput(
            score_artifact=artifact.artifact_id,
            score_path=artifact.relative_path,
            n_candidates_ranked=len(records),
            n_candidates_invalid=len(invalid_member_flags),
            invalid_member_flags=invalid_member_flags,
            policy_version=POLICY_VERSION,
        )
