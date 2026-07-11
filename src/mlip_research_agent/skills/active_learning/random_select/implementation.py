"""Deterministic, label-free random selection (the AL control arm)."""

from __future__ import annotations

import json

import numpy as np
from pydantic import BaseModel

from mlip_research_agent.skills.active_learning.random_select.schema import (
    POLICY_VERSION,
    RandomSelectInput,
    RandomSelectOutput,
)
from mlip_research_agent.skills.active_learning.random_select.validators import (
    validate_inputs,
    validate_selection,
)
from mlip_research_agent.skills.active_learning.selection_types import SelectionRecord
from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill

SELECTION_FILENAME = "random_selection.json"


@register_skill
class RandomSelectSkill(Skill):
    name = "random_select"
    input_model = RandomSelectInput
    output_model = RandomSelectOutput
    cost_class = "trivial"
    permission_level = "auto"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, RandomSelectInput)
        validate_inputs(params)

        # Stable ordering first, then a seeded permutation: the same pool and
        # seed always select the same candidates regardless of input order.
        ordered_pool = sorted(params.pool_candidate_ids)
        rng = np.random.default_rng([params.seed, len(ordered_pool)])
        permutation = rng.permutation(len(ordered_pool))
        selected_ids = [ordered_pool[int(i)] for i in permutation[: params.budget]]
        validate_selection(selected_ids, params, params.metadata)

        records = [
            SelectionRecord(
                candidate_id=candidate_id,
                uncertainty_score=None,
                diversity_distance=None,
                source_group=params.metadata[candidate_id].source_group,
                rank=rank,
                selection_reason=(
                    f"seeded permutation position {rank} of {len(ordered_pool)} "
                    f"(seed {params.seed}); control arm uses no model signal"
                ),
                policy_version=POLICY_VERSION,
            )
            for rank, candidate_id in enumerate(selected_ids)
        ]
        payload = {
            "schema_version": "1.0.0",
            "campaign_id": params.campaign_id,
            "round_id": params.round_id,
            "policy_version": POLICY_VERSION,
            "seed": params.seed,
            "budget": params.budget,
            "pool_size": len(ordered_pool),
            "records": [r.model_dump(mode="json") for r in records],
        }
        path = ctx.step_dir / SELECTION_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        artifact = ctx.registry.register(path, kind="selection", step_id=ctx.step_id)
        return RandomSelectOutput(
            selection_artifact=artifact.artifact_id,
            selection_path=artifact.relative_path,
            n_selected=len(records),
            policy_version=POLICY_VERSION,
        )
