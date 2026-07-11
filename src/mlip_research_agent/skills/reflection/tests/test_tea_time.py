"""Correctness tests for the deterministic Tea Time research pause."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.schemas.failure import FailureClass
from mlip_research_agent.skills.base import SkillContext, SkillError
from mlip_research_agent.skills.reflection.implementation import TeaTimeWithReadingPoemSkill
from mlip_research_agent.skills.reflection.schema import (
    TeaTimeInput,
    TeaTimeOutput,
    TeaTimeTrigger,
)

QUESTION = "Why does the model trust the perturbed structures more than the pristine one?"
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
    out_a, _ = run_skill(tmp_path / "a", seed=7)
    out_b, _ = run_skill(tmp_path / "b", seed=7)
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


def test_required_research_pause_sections_are_structured(tmp_path: Path) -> None:
    out, _ = run_skill(
        tmp_path,
        trigger=TeaTimeTrigger.CHECKPOINT_E0_COMPLETE,
        reusable_components=["artifact registry", "model-agnostic evaluator"],
        benchmark_specific_components=["Cu fixture"],
        open_owner_questions=[
            {
                "question_id": "H2",
                "question": "Freeze the checkpoint after due diligence?",
                "why_needed": "Checkpoint selection is a scientific gate.",
                "current_evidence": "Two pinned candidates and an E0 diagnostic exist.",
                "recommended_default": "Wait for the comparison memo.",
                "choices": ["A. Approve", "B. Gather more evidence"],
            }
        ],
    )
    payload = json.loads((tmp_path / out.reframings_path).read_text())
    assert payload["current_stage_purpose"] == out.purpose_summary
    assert payload["benchmark_overfitting_audit"]["risk"] == "low"
    assert len(payload["alternative_paths"]) in {2, 3}
    assert payload["owner_question_packet_draft"]["status"] == "questions_pending"
    assert out.trigger is TeaTimeTrigger.CHECKPOINT_E0_COMPLETE
    assert out.n_owner_questions == 1


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


def test_data_artifact_access_is_schema_forbidden() -> None:
    with pytest.raises(ValidationError):
        TeaTimeInput.model_validate(
            {
                "focus_question": QUESTION,
                "data_path": "steps/evaluation/test_metrics.json",
            }
        )


def test_serialized_pause_contains_no_restricted_research_payload(tmp_path: Path) -> None:
    out, _ = run_skill(tmp_path)
    payload = json.loads((tmp_path / out.reframings_path).read_text())
    text = json.dumps(payload, sort_keys=True)
    for forbidden in (
        '"api_key"',
        '"energy_ev"',
        '"forces_ev_per_a"',
        '"hidden_labels"',
        '"per_record_errors"',
        '"test_metric_details"',
    ):
        assert forbidden not in text


def test_n_provocations_bounds() -> None:
    with pytest.raises(ValidationError):
        TeaTimeInput(focus_question=QUESTION, n_provocations=0)
    with pytest.raises(ValidationError):
        TeaTimeInput(focus_question=QUESTION, n_provocations=8)
