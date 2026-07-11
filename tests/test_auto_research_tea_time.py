"""Tea Time boundary tests: required triggers, isolation, non-evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_research_agent.research.auto_research import (
    LocalDemoConfig,
    MutationClass,
    MutationPolicy,
    SyntheticAggregateEvaluator,
    SyntheticQuadraticAdapter,
)
from mlip_research_agent.research.auto_research.controller import AutoResearchController
from mlip_research_agent.research.auto_research.tea_time_boundary import (
    CONSECUTIVE_FAILURE_THRESHOLD,
    REPEATED_MUTATION_CLASS_THRESHOLD,
    REQUIRED_TRIGGERS,
    select_trigger,
)
from mlip_research_agent.skills.reflection.schema import TeaTimeInput, TeaTimeTrigger

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_demo(tmp_path: Path) -> Path:
    demo = LocalDemoConfig.load(REPO_ROOT / "configs/research/ralphthon_local_demo.yaml")
    policy = MutationPolicy.load(REPO_ROOT / demo.mutation_policy_path)
    controller = AutoResearchController(
        run_id="tea-run",
        run_dir=tmp_path / "tea-run",
        objective=demo.objective.sealed(),
        mutation_policy=policy,
        adapter=SyntheticQuadraticAdapter(),
        evaluator=SyntheticAggregateEvaluator(),
        acceptance=demo.acceptance,
        base_config=dict(demo.base_config),  # type: ignore[arg-type]
        fixture_seed=demo.fixture_seed,
        git_commit="0" * 40,
    )
    controller.run()
    return controller.run_dir


def test_tea_time_invoked_before_every_iteration(tmp_path: Path) -> None:
    run_dir = run_demo(tmp_path)
    t1 = json.loads((run_dir / "iteration-001/tea_time.json").read_text())
    t2 = json.loads((run_dir / "iteration-002/tea_time.json").read_text())
    assert t1["trigger"] == "auto_research_first_proposal"
    assert TeaTimeTrigger(t2["trigger"]) in REQUIRED_TRIGGERS
    # The reflection skill actually ran and registered its artifacts.
    for record in (t1, t2):
        assert record["skill_report_artifact"]
        assert record["skill_reframings_artifact"]


def test_tea_time_output_is_never_evidence(tmp_path: Path) -> None:
    run_dir = run_demo(tmp_path)
    record = json.loads((run_dir / "iteration-001/tea_time.json").read_text())
    assert record["evidence_class"] == "agent_text"
    assert record["claim_eligible"] is False
    # The required reflection fields are all present.
    for field in (
        "objective_alignment",
        "benchmark_lock_in_risk",
        "post_hoc_selection_risk",
        "alternative_routes",
        "continue_or_refine_recommendation",
        "pivot_request_rationale",
        "owner_question",
    ):
        assert field in record


def test_protected_data_cannot_enter_tea_time() -> None:
    # The reflection skill's input model structurally rejects data payloads.
    with pytest.raises(ValidationError):
        TeaTimeInput.model_validate(
            {
                "focus_question": "Should we mutate the learning rate?",
                "data_path": "data_registry/datasets/cu_phase2/records.json",
            }
        )
    with pytest.raises(ValidationError):
        TeaTimeInput.model_validate(
            {
                "focus_question": "Should we mutate the learning rate?",
                "hidden_labels": {"rec-1": -3.7},
            }
        )


def test_trace_has_no_label_or_secret_keys(tmp_path: Path) -> None:
    run_dir = run_demo(tmp_path)
    forbidden = {"hidden_labels", "api_key", "per_record_test_errors", "frozen_test_records"}
    for path in run_dir.rglob("*.json"):
        payload = path.read_text()
        for key in forbidden:
            assert f'"{key}"' not in payload, f"{path} leaked {key}"


def test_required_boundary_triggers() -> None:
    base = {
        "is_first_proposal": False,
        "proposal_is_remote": False,
        "prior_mutation_classes": [],
        "consecutive_failures": 0,
        "is_pivot_request": False,
        "changes_checkpoint_family": False,
        "renders_claims": False,
    }
    assert (
        select_trigger(**{**base, "is_first_proposal": True})
        is TeaTimeTrigger.AUTO_RESEARCH_FIRST_PROPOSAL
    )
    assert (
        select_trigger(**{**base, "proposal_is_remote": True})
        is TeaTimeTrigger.AUTO_RESEARCH_REMOTE_PRELAUNCH
    )
    assert (
        select_trigger(
            **{**base, "consecutive_failures": CONSECUTIVE_FAILURE_THRESHOLD}
        )
        is TeaTimeTrigger.AUTO_RESEARCH_CONSECUTIVE_FAILURES
    )
    assert (
        select_trigger(**{**base, "is_pivot_request": True})
        is TeaTimeTrigger.AUTO_RESEARCH_PIVOT_REQUEST
    )
    assert (
        select_trigger(**{**base, "changes_checkpoint_family": True})
        is TeaTimeTrigger.AUTO_RESEARCH_CHECKPOINT_FAMILY_CHANGE
    )
    assert (
        select_trigger(**{**base, "renders_claims": True})
        is TeaTimeTrigger.AUTO_RESEARCH_CLAIM_RENDERING
    )


def test_repeated_mutation_class_trigger() -> None:
    streak = [MutationClass.BOUNDED_MUTABLE] * REPEATED_MUTATION_CLASS_THRESHOLD
    trigger = select_trigger(
        is_first_proposal=False,
        proposal_is_remote=False,
        prior_mutation_classes=streak,
        consecutive_failures=0,
        is_pivot_request=False,
        changes_checkpoint_family=False,
        renders_claims=False,
    )
    assert trigger is TeaTimeTrigger.AUTO_RESEARCH_REPEATED_MUTATION_CLASS


def test_pivot_recommendation_remains_owner_gated(tmp_path: Path) -> None:
    """A pivot pause produces a rationale and an owner question — the trace
    carries no pivot approval, and no gate exists that Tea Time could satisfy."""
    run_dir = run_demo(tmp_path)
    record = json.loads((run_dir / "iteration-001/tea_time.json").read_text())
    assert "?" in record["owner_question"]
    # Nothing in the trace records an owner approval minted by Tea Time.
    approvals = [p for p in run_dir.rglob("*approval*")]
    assert approvals == []
