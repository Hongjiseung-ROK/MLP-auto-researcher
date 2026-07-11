"""Atomic SKILL wrapper for final active learning decision gate."""

from __future__ import annotations

import json
import random
import numpy as np

from pydantic import BaseModel

from mlip_research_agent.data.manifests import NormalizedDataset
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.active_learning.decision_gate.schema import (
    DecisionGateInput,
    DecisionGateOutput,
)
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.active_learning.decision_gate.validators import run_relative_file

DECISION_FILE = "decision_gate.json"

@register_skill
class DecisionGateSkill(Skill):
    name = "decision_gate"
    input_model = DecisionGateInput
    output_model = DecisionGateOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, DecisionGateInput)
        
        if len(params.pool_ids) < params.budget:
            raise SkillError(
                f"budget ({params.budget}) exceeds pool size ({len(params.pool_ids)})",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )

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

        # Load uncertainty scores
        uq_path = run_relative_file(ctx.run_dir, params.uncertainty_scores_path, "uncertainty scores")
        try:
            uq_data = json.loads(uq_path.read_text())
            # Support both direct id->float map or ensemble_uq output format
            if "uncertainty_scores" in uq_data:
                uq_scores = uq_data["uncertainty_scores"]
            else:
                uq_scores = uq_data
        except (OSError, ValueError) as exc:
            raise SkillError(
                f"uncertainty scores artifact malformed: {exc}",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            ) from exc

        # Load descriptors
        desc_path = run_relative_file(ctx.run_dir, params.descriptors_path, "descriptors")
        try:
            descriptors = json.loads(desc_path.read_text())
        except (OSError, ValueError) as exc:
            raise SkillError(
                f"descriptors artifact malformed: {exc}",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            ) from exc

        # Validate inputs
        for pid in params.pool_ids:
            if pid not in uq_scores:
                raise SkillError(
                    f"pool_id {pid} missing from uncertainty scores",
                    failure_class=FailureClass.VALIDATION_ERROR,
                    severity=Severity.HIGH,
                    retryable=False,
                )
            if pid not in descriptors:
                raise SkillError(
                    f"pool_id {pid} missing from descriptors",
                    failure_class=FailureClass.VALIDATION_ERROR,
                    severity=Severity.HIGH,
                    retryable=False,
                )

        # Source hashes mapping
        source_hashes = {}
        for c in dataset.configurations:
            if c.config_id in params.pool_ids:
                source_hashes[c.config_id] = c.source_record_sha256

        # Step 1: Uncertainty Filtering (keep top N, where N = min(budget * 5, len(pool)))
        # Sort pool_ids by uncertainty score descending
        pool_sorted = sorted(params.pool_ids, key=lambda x: uq_scores[x], reverse=True)
        filter_cutoff = min(len(pool_sorted), params.budget * 5)
        filtered_in = pool_sorted[:filter_cutoff]
        filtered_out = pool_sorted[filter_cutoff:]
        
        reasons = {}
        for pid in filtered_out:
            reasons[pid] = "rejected: low uncertainty"

        # Step 2: Diversity Selection on filtered_in
        # Farthest point sampling
        rng = random.Random(params.seed)
        
        desc_list = [descriptors[pid] for pid in filtered_in]
        X = np.array(desc_list, dtype=np.float64)
        
        selected_indices = []
        if len(filtered_in) > 0:
            idx = rng.randrange(len(filtered_in))
            selected_indices.append(idx)
            
            distances = np.sum((X - X[idx])**2, axis=1)
            
            for _ in range(params.budget - 1):
                max_dist = np.max(distances)
                candidates = np.where(distances == max_dist)[0]
                if len(candidates) > 1:
                    next_idx = int(rng.choice(candidates))
                else:
                    next_idx = int(candidates[0])
                    
                selected_indices.append(next_idx)
                
                new_distances = np.sum((X - X[next_idx])**2, axis=1)
                distances = np.minimum(distances, new_distances)
                
        selected_ids = [filtered_in[i] for i in selected_indices]
        for pid in selected_ids:
            reasons[pid] = "selected: top uncertainty + diversity"
            
        rejected_ids = filtered_out[:]
        for pid in filtered_in:
            if pid not in selected_ids:
                rejected_ids.append(pid)
                reasons[pid] = "rejected: redundant by diversity"

        record = {
            "selected_ids": selected_ids,
            "rejected_ids": rejected_ids,
            "reasons": reasons,
            "source_hashes": source_hashes,
            "policy_identity": params.policy_identity,
            "budget": params.budget,
            "seed": params.seed,
        }
        
        out_path = ctx.step_dir / DECISION_FILE
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        
        artifact = ctx.registry.register(out_path, kind="decision_gate_record", step_id=ctx.step_id)
        
        return DecisionGateOutput(
            selected_ids=selected_ids,
            rejected_ids=rejected_ids,
            reasons=reasons,
            source_hashes=source_hashes,
            decision_artifact=artifact.artifact_id,
            decision_path=artifact.relative_path,
        )
