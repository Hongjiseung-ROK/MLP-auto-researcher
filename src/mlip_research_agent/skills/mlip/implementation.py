"""Mock MLIP training and evaluation.

The 'model' is a mean-energy baseline over a seeded train split. It carries
its split with it so evaluation is honest: metrics come only from held-out
labels, and the resulting claim references the exact backing artifacts.
"""

from __future__ import annotations

import json

import numpy as np
from pydantic import BaseModel

from mlip_research_agent.evaluation.metrics import mean_absolute_error
from mlip_research_agent.schemas.claims import Claim
from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill
from mlip_research_agent.skills.mlip.schema import (
    EvaluationInput,
    EvaluationOutput,
    TrainingInput,
    TrainingOutput,
)
from mlip_research_agent.skills.mlip.validators import (
    load_json_artifact,
    validate_label_count,
    validate_model_name,
)

MODEL_FILENAME = "model.json"
METRICS_FILENAME = "metrics.json"


@register_skill
class MockMLIPTrainingSkill(Skill):
    name = "mock_mlip_training"
    input_model = TrainingInput
    output_model = TrainingOutput
    cost_class = "cheap"  # the real skill is 'expensive' (GPU fine-tuning)

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, TrainingInput)
        validate_model_name(params.model_name)
        labels_payload = load_json_artifact(ctx.run_dir, params.labels_path, "label set")
        labels = labels_payload["labels"]
        validate_label_count(len(labels))

        rng = np.random.default_rng([ctx.seed, 43])
        permutation = rng.permutation(len(labels))
        n_holdout = max(1, round(len(labels) * params.holdout_fraction))
        holdout_positions = sorted(int(i) for i in permutation[:n_holdout])
        train_positions = sorted(int(i) for i in permutation[n_holdout:])
        train_energies = [float(labels[i]["energy"]) for i in train_positions]

        model = {
            "model_name": params.model_name,
            "parameters": {"mean_energy": round(float(np.mean(train_energies)), 10)},
            "train_positions": train_positions,
            "holdout_positions": holdout_positions,
            "trained_on": params.labels_path,
            "seed": ctx.seed,
        }
        path = ctx.step_dir / MODEL_FILENAME
        path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
        artifact = ctx.registry.register(path, kind="model", step_id=ctx.step_id)
        return TrainingOutput(
            model_artifact=artifact.artifact_id,
            model_path=artifact.relative_path,
            model_name=params.model_name,
            n_train=len(train_positions),
            n_holdout=len(holdout_positions),
        )


@register_skill
class MockEvaluationSkill(Skill):
    name = "mock_evaluation"
    input_model = EvaluationInput
    output_model = EvaluationOutput
    cost_class = "cheap"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, EvaluationInput)
        model = load_json_artifact(ctx.run_dir, params.model_path, "model")
        labels_payload = load_json_artifact(ctx.run_dir, params.labels_path, "label set")
        labels = labels_payload["labels"]

        holdout_positions = [int(i) for i in model["holdout_positions"]]
        targets = [float(labels[i]["energy"]) for i in holdout_positions]
        predictions = [float(model["parameters"]["mean_energy"])] * len(targets)
        energy_mae = round(mean_absolute_error(predictions, targets), 10)

        path = ctx.step_dir / METRICS_FILENAME
        metrics = {
            "energy_mae": energy_mae,
            "n_holdout": len(targets),
            "model_path": params.model_path,
            "labels_path": params.labels_path,
        }
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        artifact = ctx.registry.register(path, kind="metrics", step_id=ctx.step_id)

        model_artifact = f"training:{MODEL_FILENAME}"
        backing = [artifact.artifact_id]
        if ctx.registry.get(model_artifact) is not None:
            backing.append(model_artifact)
        ctx.register_claim(
            Claim(
                claim_id="energy_mae_holdout",
                statement=(
                    f"The {model['model_name']} model reaches a holdout energy MAE of "
                    f"{energy_mae} eV on {len(targets)} held-out mock labels."
                ),
                value=energy_mae,
                units="eV",
                created_by_step=ctx.step_id,
                artifact_references=backing,
            )
        )
        return EvaluationOutput(
            metrics_artifact=artifact.artifact_id,
            metrics_path=artifact.relative_path,
            energy_mae=energy_mae,
            n_holdout=len(targets),
        )
