"""Farthest-point diversity selection over explicit descriptors."""

from __future__ import annotations

import json

import numpy as np
from pydantic import BaseModel

from mlip_research_agent.skills.active_learning.diversity_select.schema import (
    POLICY_VERSION,
    DiversitySelectInput,
    DiversitySelectOutput,
)
from mlip_research_agent.skills.active_learning.diversity_select.validators import (
    validate_inputs,
)
from mlip_research_agent.skills.active_learning.selection_types import SelectionRecord
from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill

SELECTION_FILENAME = "diversity_selection.json"


def farthest_point_selection(
    ids: list[str], matrix: np.ndarray, budget: int
) -> list[tuple[str, float]]:
    """Deterministic FPS. Returns (candidate_id, diversity_distance) in
    selection order; distance is the min Euclidean distance to the already
    selected set at selection time (inf-analogue for the seed point).

    Deterministic tie-breaking everywhere: the seed point is the candidate
    farthest from the pool centroid (ties by lexicographic id); each step
    picks the largest min-distance (ties by lexicographic id).
    """
    centroid = matrix.mean(axis=0)
    centroid_distance = np.linalg.norm(matrix - centroid, axis=1)
    order = sorted(range(len(ids)), key=lambda i: (-float(centroid_distance[i]), ids[i]))
    seed_index = order[0]
    selected = [seed_index]
    result = [(ids[seed_index], float(centroid_distance[seed_index]))]

    min_distance = np.linalg.norm(matrix - matrix[seed_index], axis=1)
    while len(selected) < budget:
        candidates = sorted(
            (i for i in range(len(ids)) if i not in set(selected)),
            key=lambda i: (-float(min_distance[i]), ids[i]),
        )
        chosen = candidates[0]
        selected.append(chosen)
        result.append((ids[chosen], float(min_distance[chosen])))
        distance_to_chosen = np.linalg.norm(matrix - matrix[chosen], axis=1)
        min_distance = np.minimum(min_distance, distance_to_chosen)
    return result


@register_skill
class DiversitySelectSkill(Skill):
    name = "diversity_select"
    input_model = DiversitySelectInput
    output_model = DiversitySelectOutput
    cost_class = "cheap"
    permission_level = "auto"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, DiversitySelectInput)
        validate_inputs(params)

        ids = sorted(params.pool_candidate_ids)
        matrix = np.asarray(
            [params.descriptors.vectors[c] for c in ids], dtype=float
        )
        picks = farthest_point_selection(ids, matrix, params.budget)

        records = [
            SelectionRecord(
                candidate_id=candidate_id,
                uncertainty_score=None,
                diversity_distance=distance,
                source_group=params.metadata[candidate_id].source_group,
                rank=rank,
                selection_reason=(
                    "farthest-point step: min distance to already-selected set "
                    f"{distance:.6f} (descriptor source: {params.descriptors.source})"
                ),
                policy_version=POLICY_VERSION,
            )
            for rank, (candidate_id, distance) in enumerate(picks)
        ]

        selected_indices = [ids.index(candidate_id) for candidate_id, _ in picks]
        chosen = matrix[selected_indices]
        if len(chosen) > 1:
            diffs = chosen[:, None, :] - chosen[None, :, :]
            pairwise = np.linalg.norm(diffs, axis=2)
            upper = pairwise[np.triu_indices(len(chosen), k=1)]
            mean_pairwise = float(np.mean(upper))
        else:
            mean_pairwise = 0.0

        payload = {
            "schema_version": "1.0.0",
            "campaign_id": params.campaign_id,
            "round_id": params.round_id,
            "policy_version": POLICY_VERSION,
            "descriptor_source": params.descriptors.source,
            "descriptor_dimension": params.descriptors.dimension,
            "mean_pairwise_distance": mean_pairwise,
            "records": [r.model_dump(mode="json") for r in records],
        }
        path = ctx.step_dir / SELECTION_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        artifact = ctx.registry.register(path, kind="selection", step_id=ctx.step_id)
        return DiversitySelectOutput(
            selection_artifact=artifact.artifact_id,
            selection_path=artifact.relative_path,
            n_selected=len(records),
            mean_pairwise_distance=mean_pairwise,
            policy_version=POLICY_VERSION,
        )
