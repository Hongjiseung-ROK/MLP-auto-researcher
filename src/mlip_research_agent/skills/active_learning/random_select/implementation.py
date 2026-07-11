"""Atomic SKILL wrapper for random selection."""

from __future__ import annotations

import json
from pathlib import Path
import random

from pydantic import BaseModel

from mlip_research_agent.data.manifests import NormalizedDataset
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.active_learning.random_select.schema import (
    RandomSelectInput,
    RandomSelectOutput,
)
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.active_learning.random_select.validators import run_relative_file

SELECTION_FILE = "random_selection.json"

@register_skill
class RandomSelectSkill(Skill):
    name = "random_select"
    input_model = RandomSelectInput
    output_model = RandomSelectOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, RandomSelectInput)
        dataset_path = run_relative_file(ctx.run_dir, params.dataset_path, "pool dataset")

        try:
            dataset = NormalizedDataset.load(dataset_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SkillError(
                f"dataset artifact is malformed: {exc}",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            ) from exc

        # Pool membership validation
        dataset_ids = {c.config_id for c in dataset.configurations}
        invalid_ids = [pid for pid in params.pool_ids if pid not in dataset_ids]
        if invalid_ids:
            raise SkillError(
                f"pool_ids contains IDs not in dataset: {invalid_ids}",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )
            
        # Duplicate rejection
        if len(set(params.pool_ids)) != len(params.pool_ids):
            raise SkillError(
                "pool_ids contains duplicates",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )

        if len(params.pool_ids) < params.budget:
            raise SkillError(
                f"budget ({params.budget}) exceeds pool size ({len(params.pool_ids)})",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )

        # Deterministic random selection
        rng = random.Random(params.seed)
        selected_ids = rng.sample(sorted(params.pool_ids), params.budget)
        
        reasons = {sid: "selected_via_uniform_random_sampling" for sid in selected_ids}
        
        # Stable serialization
        selection_record = {
            "selected_ids": selected_ids,
            "reasons": reasons,
            "seed": params.seed,
            "budget": params.budget,
        }
        
        out_path = ctx.step_dir / SELECTION_FILE
        out_path.write_text(json.dumps(selection_record, indent=2, sort_keys=True))
        
        artifact = ctx.registry.register(out_path, kind="random_selection_record", step_id=ctx.step_id)
        
        return RandomSelectOutput(
            selected_ids=selected_ids,
            reasons=reasons,
            selection_artifact=artifact.artifact_id,
            selection_path=artifact.relative_path,
        )
