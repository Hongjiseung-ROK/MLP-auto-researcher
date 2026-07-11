"""Correctness tests for the tea_time_with_reading_poem skill."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.schemas.failure import FailureClass
from mlip_research_agent.skills.base import SkillContext, SkillError
from mlip_research_agent.skills.reflection.implementation import TeaTimeWithReadingPoemSkill
from mlip_research_agent.skills.reflection.schema import TeaTimeInput, TeaTimeOutput

QUESTION = "Why does the model trust the perturbed structures more than the pristine one?"
ENERGIES = [1.0, 2.0, 3.0, 4.0, 100.0]


def write_labels(tmp_path: Path) -> str:
    labels_dir = tmp_path / "steps" / "labeling"
    labels_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "mock_lj",
        "settings": {},
        "labels": [{"index": i, "energy": e, "fmax": 0.0} for i, e in enumerate(ENERGIES)],
    }
    (labels_dir / "labels.json").write_text(json.dumps(payload, sort_keys=True))
    return "steps/labeling/labels.json"


def run_skill(
    tmp_path: Path, seed: int = 7, **overrides: object
) -> tuple[TeaTimeOutput, SkillContext]:
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="tea_time",
        seed=seed,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )
    inputs = TeaTimeInput.model_validate({"focus_question": QUESTION, **overrides})
    out = TeaTimeWithReadingPoemSkill().run(inputs, ctx)
    assert isinstance(out, TeaTimeOutput)
    return out, ctx


def test_byte_deterministic_for_equal_seeds(tmp_path: Path) -> None:
    data_a = write_labels(tmp_path / "a")
    data_b = write_labels(tmp_path / "b")
    out_a, _ = run_skill(tmp_path / "a", seed=7, data_path=data_a)
    out_b, _ = run_skill(tmp_path / "b", seed=7, data_path=data_b)
    for rel_a, rel_b in [
        (out_a.report_path, out_b.report_path),
        (out_a.reframings_path, out_b.reframings_path),
    ]:
        assert ((tmp_path / "a") / rel_a).read_bytes() == ((tmp_path / "b") / rel_b).read_bytes()


def test_seed_changes_the_break(tmp_path: Path) -> None:
    out_a, _ = run_skill(tmp_path / "a", seed=7)
    out_b, _ = run_skill(tmp_path / "b", seed=8)
    content_a = ((tmp_path / "a") / out_a.reframings_path).read_text()
    content_b = ((tmp_path / "b") / out_b.reframings_path).read_text()
    assert content_a != content_b


def test_alternative_views_match_numpy(tmp_path: Path) -> None:
    data_path = write_labels(tmp_path)
    out, _ = run_skill(tmp_path, data_path=data_path)
    views = json.loads((tmp_path / out.reframings_path).read_text())["alternative_data_views"]
    arr = np.asarray(ENERGIES)
    assert views["mean"] == pytest.approx(float(arr.mean()))
    assert views["median"] == pytest.approx(float(np.median(arr)))
    assert views["strangest_index"] == 4  # 100.0 is farthest from the median
    assert views["rank_top3_indices"][0] == 4
    assert views["n_values"] == len(ENERGIES)


def test_zero_claims_invariant(tmp_path: Path) -> None:
    _, ctx = run_skill(tmp_path)
    assert ctx.claims == []
    kinds = {a.kind for a in ctx.registry.all()}
    assert kinds == {"reflection"}


def test_poem_override_and_unknown_poem(tmp_path: Path) -> None:
    out, _ = run_skill(tmp_path, poem_key="whitman-learnd-astronomer")
    assert out.poem_key == "whitman-learnd-astronomer"
    with pytest.raises(SkillError) as excinfo:
        run_skill(tmp_path / "bad", poem_key="vogon-jeltz-ode")
    assert excinfo.value.failure_class is FailureClass.VALIDATION_ERROR


def test_missing_data_rejected(tmp_path: Path) -> None:
    with pytest.raises(SkillError) as excinfo:
        run_skill(tmp_path, data_path="steps/nowhere/data.json")
    assert excinfo.value.failure_class is FailureClass.VALIDATION_ERROR


def test_n_provocations_bounds() -> None:
    with pytest.raises(ValidationError):
        TeaTimeInput(focus_question=QUESTION, n_provocations=0)
    with pytest.raises(ValidationError):
        TeaTimeInput(focus_question=QUESTION, n_provocations=8)
