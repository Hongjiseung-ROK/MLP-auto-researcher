"""Atomic SKILL wrapper for ensemble uncertainty quantification."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from mlip_research_agent.data.manifests import NormalizedDataset
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.active_learning.ensemble_uq.schema import (
    EnsembleUqInput,
    EnsembleUqOutput,
)
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.active_learning.ensemble_uq.validators import run_relative_file

import numpy as np

UQ_FILE = "ensemble_uq.json"

@register_skill
class EnsembleUqSkill(Skill):
    name = "ensemble_uq"
    input_model = EnsembleUqInput
    output_model = EnsembleUqOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, EnsembleUqInput)
        
        if len(params.member_identities) != len(params.member_prediction_paths):
            raise SkillError(
                "member_identities and member_prediction_paths must have the same length",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )

        # Validate that the member prediction datasets can be loaded
        datasets = []
        for p in params.member_prediction_paths:
            resolved_p = run_relative_file(ctx.run_dir, p, "member prediction dataset")
            try:
                datasets.append(NormalizedDataset.load(resolved_p))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise SkillError(
                    f"prediction dataset artifact is malformed: {exc}",
                    failure_class=FailureClass.VALIDATION_ERROR,
                    severity=Severity.HIGH,
                    retryable=False,
                ) from exc

        # Extract predictions for each pool id
        uncertainty_scores = {}
        for pool_id in params.pool_ids:
            forces = []
            energies = []
            
            for ds in datasets:
                config = next((c for c in ds.configurations if c.config_id == pool_id), None)
                if config is None:
                    raise SkillError(
                        f"pool_id {pool_id} not found in all member predictions",
                        failure_class=FailureClass.VALIDATION_ERROR,
                        severity=Severity.HIGH,
                        retryable=False,
                    )
                forces.append(config.forces_ev_per_a)
                energies.append(config.energy_ev)
            
            # calculate force disagreement (std dev over members, mean over atoms and xyz)
            f_array = np.array(forces) # shape: (n_members, n_atoms, 3)
            # variance over members:
            f_var = np.var(f_array, axis=0)
            # mean std dev over atoms and components:
            f_score = float(np.mean(np.sqrt(f_var)))
            
            e_score = 0.0
            if params.include_energy_disagreement:
                e_array = np.array(energies)
                e_score = float(np.std(e_array))
                
            # Aggregate based on formula
            if params.aggregation_formula == "mean_std_dev":
                score = f_score + e_score
            else:
                score = f_score + e_score
                
            if not np.isfinite(score):
                raise SkillError(
                    f"non-finite uncertainty score for {pool_id}",
                    failure_class=FailureClass.VALIDATION_ERROR,
                    severity=Severity.HIGH,
                    retryable=False,
                )
            uncertainty_scores[pool_id] = score
            
        ranked_ids = sorted(params.pool_ids, key=lambda x: uncertainty_scores[x], reverse=True)
        
        record = {
            "ranked_ids": ranked_ids,
            "uncertainty_scores": uncertainty_scores,
            "aggregation_formula": params.aggregation_formula,
        }
        
        out_path = ctx.step_dir / UQ_FILE
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        
        artifact = ctx.registry.register(out_path, kind="ensemble_uq_record", step_id=ctx.step_id)
        
        return EnsembleUqOutput(
            ranked_ids=ranked_ids,
            uncertainty_scores=uncertainty_scores,
            uq_artifact=artifact.artifact_id,
            uq_path=artifact.relative_path,
        )
