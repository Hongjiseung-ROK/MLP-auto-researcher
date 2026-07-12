"""Acquisition decision gate: filter → window → diversify → exact budget."""

from __future__ import annotations

import json
import math

import numpy as np
from pydantic import BaseModel

from mlip_research_agent.skills.active_learning.decision_gate.schema import (
    POLICY_VERSION,
    DecisionGateInput,
    DecisionGateOutput,
)
from mlip_research_agent.skills.active_learning.decision_gate.validators import (
    validate_inputs,
)
from mlip_research_agent.skills.active_learning.diversity_select.implementation import (
    farthest_point_selection,
    resolve_descriptors,
)
from mlip_research_agent.skills.active_learning.diversity_select.schema import (
    DiversitySelectInput,
)
from mlip_research_agent.skills.active_learning.diversity_select.validators import (
    validate_inputs as validate_descriptor_inputs,
)
from mlip_research_agent.skills.active_learning.selection_types import (
    SelectionRecord,
    reject,
)
from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill

SELECTION_FILENAME = "gate_selection.json"


@register_skill
class DecisionGateSkill(Skill):
    name = "decision_gate"
    input_model = DecisionGateInput
    output_model = DecisionGateOutput
    cost_class = "cheap"
    permission_level = "auto"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, DecisionGateInput)
        descriptors = resolve_descriptors(params, ctx)
        validate_inputs(params, descriptors)

        # Stage 1: invalid filtering.
        valid = [s for s in params.signals if not s.invalid]
        n_invalid = len(params.signals) - len(valid)
        if len(valid) < params.budget:
            raise reject(
                f"only {len(valid)} valid candidates remain after invalid filtering "
                f"but the budget requires exactly {params.budget}"
            )

        # Stage 2: uncertainty window — top fraction by disagreement,
        # deterministic tiebreak by candidate id, never smaller than budget.
        ordered = sorted(valid, key=lambda s: (-s.force_disagreement, s.candidate_id))
        window_size = max(
            params.budget, math.ceil(len(ordered) * params.uncertainty_window_fraction)
        )
        window = ordered[:window_size]
        window_ids = sorted(s.candidate_id for s in window)
        signal_by_id = {s.candidate_id: s for s in window}

        # Stage 3: diversity selection inside the window (validated with the
        # same fail-closed descriptor rules as diversity_select).
        window_descriptors = descriptors.model_copy(
            update={
                "vectors": {
                    c: descriptors.vectors[c] for c in window_ids
                }
            }
        )
        descriptor_input = DiversitySelectInput(
                pool_candidate_ids=window_ids,
                metadata={c: params.metadata[c] for c in window_ids},
                descriptors=window_descriptors,
                budget=params.budget,
                campaign_id=params.campaign_id,
                round_id=params.round_id,
            )
        validate_descriptor_inputs(descriptor_input, window_descriptors)
        matrix = np.asarray(
            [descriptors.vectors[c] for c in window_ids], dtype=float
        )
        picks = farthest_point_selection(window_ids, matrix, params.budget)

        # Stage 4: exact budget enforcement.
        if len(picks) != params.budget:
            raise reject(
                f"gate produced {len(picks)} selections; the budget is exactly "
                f"{params.budget}"
            )
        if len({c for c, _ in picks}) != len(picks):
            raise reject("gate produced duplicate selections")

        # Stage 5: machine-readable reason artifact.
        records = [
            SelectionRecord(
                candidate_id=candidate_id,
                uncertainty_score=signal_by_id[candidate_id].force_disagreement,
                diversity_distance=distance,
                source_group=params.metadata[candidate_id].source_group,
                rank=rank,
                selection_reason=(
                    f"in top-{window_size} disagreement window; farthest-point "
                    f"distance {distance:.6f} at pick {rank}"
                ),
                policy_version=POLICY_VERSION,
            )
            for rank, (candidate_id, distance) in enumerate(picks)
        ]
        payload = {
            "schema_version": "1.0.0",
            "campaign_id": params.campaign_id,
            "round_id": params.round_id,
            "policy_version": POLICY_VERSION,
            "descriptor_artifact": params.descriptor_artifact,
            "pipeline": [
                "invalid_filtering",
                "uncertainty_window",
                "diversity_selection",
                "exact_budget",
                "reason_artifact",
            ],
            "n_pool": len(params.pool_candidate_ids),
            "n_filtered_invalid": n_invalid,
            "window_size": window_size,
            "uncertainty_window_fraction": params.uncertainty_window_fraction,
            "records": [r.model_dump(mode="json") for r in records],
        }
        path = ctx.step_dir / SELECTION_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        artifact = ctx.registry.register(path, kind="selection", step_id=ctx.step_id)
        return DecisionGateOutput(
            selection_artifact=artifact.artifact_id,
            selection_path=artifact.relative_path,
            n_selected=len(records),
            n_filtered_invalid=n_invalid,
            n_in_window=window_size,
            policy_version=POLICY_VERSION,
        )
