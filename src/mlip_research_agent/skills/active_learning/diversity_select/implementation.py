"""Atomic SKILL wrapper for diversity selection (farthest point sampling)."""

from __future__ import annotations

import json
from pathlib import Path
import random

from pydantic import BaseModel
import numpy as np

from mlip_research_agent.data.manifests import NormalizedDataset
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.active_learning.diversity_select.schema import (
    DiversitySelectInput,
    DiversitySelectOutput,
)
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.active_learning.diversity_select.validators import run_relative_file

SELECTION_FILE = "diversity_selection.json"

@register_skill
class DiversitySelectSkill(Skill):
    name = "diversity_select"
    input_model = DiversitySelectInput
    output_model = DiversitySelectOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, DiversitySelectInput)
        
        # Deduplication
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

        descriptor_p = run_relative_file(ctx.run_dir, params.descriptor_path, "descriptors")
        try:
            descriptors = json.loads(descriptor_p.read_text())
        except (OSError, ValueError) as exc:
            raise SkillError(
                f"descriptor artifact is malformed: {exc}",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            ) from exc
            
        for pid in params.pool_ids:
            if pid not in descriptors:
                raise SkillError(
                    f"pool_id {pid} missing from descriptors",
                    failure_class=FailureClass.VALIDATION_ERROR,
                    severity=Severity.HIGH,
                    retryable=False,
                )

        rng = random.Random(params.seed)
        pool = sorted(params.pool_ids)
        
        # Prepare arrays for FPS
        desc_list = [descriptors[pid] for pid in pool]
        X = np.array(desc_list, dtype=np.float64) # shape (N, D)
        
        selected_indices = []
        
        # First point chosen randomly (with tie-breaking seed)
        idx = rng.randrange(len(pool))
        selected_indices.append(idx)
        
        distances = np.sum((X - X[idx])**2, axis=1) # shape (N,)
        
        for _ in range(params.budget - 1):
            # Find the point with the maximum minimum distance to the selected points
            # To ensure stable tie-breaking, we use the deterministic max arg
            # In case of ties, np.max will return the first occurrence which is stable
            # However, we sort the initial array so the first occurrence is deterministic
            max_dist = np.max(distances)
            candidates = np.where(distances == max_dist)[0]
            # Tie breaking deterministically
            if len(candidates) > 1:
                next_idx = int(rng.choice(candidates))
            else:
                next_idx = int(candidates[0])
                
            selected_indices.append(next_idx)
            
            # Update distances
            new_distances = np.sum((X - X[next_idx])**2, axis=1)
            distances = np.minimum(distances, new_distances)
            
        selected_ids = [pool[i] for i in selected_indices]
        reasons = {sid: f"farthest_point_sampling_rank_{rank+1}" for rank, sid in enumerate(selected_ids)}
        
        record = {
            "selected_ids": selected_ids,
            "reasons": reasons,
            "seed": params.seed,
            "budget": params.budget,
        }
        
        out_path = ctx.step_dir / SELECTION_FILE
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        
        artifact = ctx.registry.register(out_path, kind="diversity_selection_record", step_id=ctx.step_id)
        
        return DiversitySelectOutput(
            selected_ids=selected_ids,
            reasons=reasons,
            selection_artifact=artifact.artifact_id,
            selection_path=artifact.relative_path,
        )
