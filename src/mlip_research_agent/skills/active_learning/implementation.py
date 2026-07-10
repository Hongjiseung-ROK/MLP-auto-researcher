"""Mock acquisition: seeded uncertainty proxy over candidate structures.

The proxy treats larger perturbation amplitude as higher model uncertainty,
plus seeded noise, so selection is informative-looking yet fully deterministic.
The real skill will mix committee disagreement, diversity, and energetic
viability (plan.md AL policy mixer).
"""

from __future__ import annotations

import json

import numpy as np
from pydantic import BaseModel

from mlip_research_agent.skills.active_learning.schema import (
    AcquisitionInput,
    AcquisitionOutput,
    AcquisitionStrategy,
)
from mlip_research_agent.skills.active_learning.validators import (
    load_structures,
    validate_selection_size,
)
from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill

SELECTION_FILENAME = "selection.json"


@register_skill
class MockAcquisitionSkill(Skill):
    name = "mock_acquisition"
    input_model = AcquisitionInput
    output_model = AcquisitionOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, AcquisitionInput)
        structures = load_structures(ctx.run_dir, params.structures_path)
        n = len(structures.systems)
        validate_selection_size(params.n_select, n)

        rng = np.random.default_rng([ctx.seed, 17])
        if params.strategy is AcquisitionStrategy.MOCK_UNCERTAINTY:
            scales = np.array([s.perturbation_scale for s in structures.systems])
            scores = scales * (1.0 + 0.1 * rng.standard_normal(n))
        else:
            scores = rng.random(n)
        order = np.argsort(-scores, kind="stable")
        selected = sorted(int(i) for i in order[: params.n_select])

        path = ctx.step_dir / SELECTION_FILENAME
        payload = {
            "strategy": params.strategy.value,
            "selected_indices": selected,
            "scores": [round(float(s), 12) for s in scores],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        artifact = ctx.registry.register(path, kind="selection", step_id=ctx.step_id)
        return AcquisitionOutput(
            selection_artifact=artifact.artifact_id,
            selection_path=artifact.relative_path,
            n_selected=len(selected),
            strategy=params.strategy,
        )
