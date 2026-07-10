"""Correctness tests for the mock MLIP training and evaluation skills."""

import json
from pathlib import Path
from typing import Any

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.schemas.failure import FailureClass
from mlip_research_agent.skills.base import SkillContext, SkillError
from mlip_research_agent.skills.mlip.implementation import (
    MockEvaluationSkill,
    MockMLIPTrainingSkill,
)
from mlip_research_agent.skills.mlip.schema import (
    EvaluationInput,
    EvaluationOutput,
    TrainingInput,
    TrainingOutput,
)


def write_labels(tmp_path: Path, energies: list[float]) -> str:
    labels_dir = tmp_path / "steps" / "labeling"
    labels_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "mock_lj",
        "settings": {},
        "labels": [{"index": i, "energy": e, "fmax": 0.0} for i, e in enumerate(energies)],
    }
    (labels_dir / "labels.json").write_text(json.dumps(payload, sort_keys=True))
    return "steps/labeling/labels.json"


def make_ctx(
    tmp_path: Path, step_id: str, registry: ArtifactRegistry, seed: int = 7
) -> SkillContext:
    return SkillContext(run_dir=tmp_path, step_id=step_id, seed=seed, attempt=0, registry=registry)


def train(
    tmp_path: Path, registry: ArtifactRegistry, labels_path: str, seed: int = 7
) -> dict[str, Any]:
    ctx = make_ctx(tmp_path, "training", registry, seed)
    out = MockMLIPTrainingSkill().run(TrainingInput(labels_path=labels_path), ctx)
    assert isinstance(out, TrainingOutput)
    model: dict[str, Any] = json.loads((tmp_path / out.model_path).read_text())
    return model


def test_training_deterministic(tmp_path: Path) -> None:
    energies = [float(i) for i in range(8)]
    labels_a = write_labels(tmp_path / "a", energies)
    labels_b = write_labels(tmp_path / "b", energies)
    model_a = train(tmp_path / "a", ArtifactRegistry(tmp_path / "a"), labels_a)
    model_b = train(tmp_path / "b", ArtifactRegistry(tmp_path / "b"), labels_b)
    assert model_a == model_b


def test_no_train_holdout_leakage(tmp_path: Path) -> None:
    labels_path = write_labels(tmp_path, [float(i) for i in range(8)])
    model = train(tmp_path, ArtifactRegistry(tmp_path), labels_path)
    train_set = set(model["train_positions"])
    holdout_set = set(model["holdout_positions"])
    assert train_set.isdisjoint(holdout_set)
    assert train_set | holdout_set == set(range(8))


def test_evaluation_mae_matches_hand_computation(tmp_path: Path) -> None:
    labels_path = write_labels(tmp_path, [float(i) for i in range(8)])
    registry = ArtifactRegistry(tmp_path)
    model = train(tmp_path, registry, labels_path)
    ctx = make_ctx(tmp_path, "evaluation", registry)
    out = MockEvaluationSkill().run(
        EvaluationInput(model_path="steps/training/model.json", labels_path=labels_path), ctx
    )
    assert isinstance(out, EvaluationOutput)
    mean_energy = model["parameters"]["mean_energy"]
    expected = sum(abs(mean_energy - float(i)) for i in model["holdout_positions"]) / len(
        model["holdout_positions"]
    )
    assert out.energy_mae == pytest.approx(expected, abs=1e-9)


def test_evaluation_registers_backed_claim(tmp_path: Path) -> None:
    labels_path = write_labels(tmp_path, [float(i) for i in range(8)])
    registry = ArtifactRegistry(tmp_path)
    train(tmp_path, registry, labels_path)
    ctx = make_ctx(tmp_path, "evaluation", registry)
    out = MockEvaluationSkill().run(
        EvaluationInput(model_path="steps/training/model.json", labels_path=labels_path), ctx
    )
    assert isinstance(out, EvaluationOutput)
    assert len(ctx.claims) == 1
    claim = ctx.claims[0]
    assert claim.claim_id == "energy_mae_holdout"
    assert out.metrics_artifact in claim.artifact_references
    assert all(registry.verify(ref) for ref in claim.artifact_references)


def test_small_label_set_rejected(tmp_path: Path) -> None:
    labels_path = write_labels(tmp_path, [0.0, 1.0])
    with pytest.raises(SkillError) as excinfo:
        train(tmp_path, ArtifactRegistry(tmp_path), labels_path)
    assert excinfo.value.failure_class is FailureClass.VALIDATION_ERROR
