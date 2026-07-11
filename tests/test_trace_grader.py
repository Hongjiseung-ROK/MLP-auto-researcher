"""Adversarial trace-grader tests: every tampering mode must fail the grade."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mlip_research_agent.research.auto_research import (
    EvaluationOutcome,
    ExperimentDecision,
    LocalDemoConfig,
    MutationPolicy,
    SyntheticAggregateEvaluator,
    SyntheticQuadraticAdapter,
)
from mlip_research_agent.research.auto_research.controller import AutoResearchController
from mlip_research_agent.verification.trace_grader import grade_trace

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo() -> LocalDemoConfig:
    return LocalDemoConfig.load(REPO_ROOT / "configs/research/ralphthon_local_demo.yaml")


@pytest.fixture(scope="module")
def complete_run(tmp_path_factory: pytest.TempPathFactory, demo: LocalDemoConfig) -> Path:
    tmp = tmp_path_factory.mktemp("grader")
    policy = MutationPolicy.load(REPO_ROOT / demo.mutation_policy_path)
    controller = AutoResearchController(
        run_id="grade-run",
        run_dir=tmp / "grade-run",
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


def tampered_copy(complete_run: Path, tmp_path: Path) -> Path:
    target = tmp_path / "tampered"
    shutil.copytree(complete_run, target)
    return target


def checks_of(run_dir: Path, demo: LocalDemoConfig) -> set[str]:
    report = grade_trace(run_dir, demo.acceptance)
    assert not report.passed
    return {v.check for v in report.violations}


def test_complete_trace_passes(complete_run: Path, demo: LocalDemoConfig) -> None:
    report = grade_trace(complete_run, demo.acceptance)
    assert report.passed, [v.model_dump() for v in report.violations]
    assert report.n_iterations_graded == 2


def test_missing_tea_time_fails(
    complete_run: Path, tmp_path: Path, demo: LocalDemoConfig
) -> None:
    run_dir = tampered_copy(complete_run, tmp_path)
    (run_dir / "iteration-001/tea_time.json").unlink()
    assert "missing_artifact" in checks_of(run_dir, demo)


def test_missing_lesson_fails(
    complete_run: Path, tmp_path: Path, demo: LocalDemoConfig
) -> None:
    run_dir = tampered_copy(complete_run, tmp_path)
    (run_dir / "iteration-002/lesson.json").unlink()
    assert "missing_artifact" in checks_of(run_dir, demo)


def test_evaluator_identity_collision_fails(
    complete_run: Path, tmp_path: Path, demo: LocalDemoConfig
) -> None:
    run_dir = tampered_copy(complete_run, tmp_path)
    path = run_dir / "iteration-001/evaluation.json"
    evaluation = EvaluationOutcome.model_validate_json(path.read_text())
    forged = evaluation.model_copy(
        update={"evaluator_name": "controller-self-evaluation/0.0.1"}
    ).sealed()
    path.write_text(forged.canonical_text())
    checks = checks_of(run_dir, demo)
    assert "evaluator_independence" in checks


def test_illegal_mutation_fails(
    complete_run: Path, tmp_path: Path, demo: LocalDemoConfig
) -> None:
    """Rewriting the config history to touch a protected key must fail."""
    run_dir = tampered_copy(complete_run, tmp_path)
    state_path = run_dir / "config_store/state-001.json"
    state = json.loads(state_path.read_text())
    state["dataset_content_sha256"] = "f" * 64
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    assert "immutable_preservation" in checks_of(run_dir, demo)


def test_hidden_label_leak_fails(
    complete_run: Path, tmp_path: Path, demo: LocalDemoConfig
) -> None:
    run_dir = tampered_copy(complete_run, tmp_path)
    leak = {"hidden_labels": {"rec-0001": -3.74}}
    (run_dir / "iteration-001/workdir/leak.json").write_text(json.dumps(leak))
    assert "protected_data" in checks_of(run_dir, demo)


def test_unapproved_remote_run_fails(
    complete_run: Path, tmp_path: Path, demo: LocalDemoConfig
) -> None:
    run_dir = tampered_copy(complete_run, tmp_path)
    path = run_dir / "iteration-001/execution.json"
    from mlip_research_agent.research.auto_research import ExperimentExecution

    execution = ExperimentExecution.model_validate_json(path.read_text())
    forged = execution.model_copy(
        update={"compute_attestation": "colab-l4-session-42"}
    ).sealed()
    path.write_text(forged.canonical_text())
    assert "remote_execution" in checks_of(run_dir, demo)


def test_unsupported_accept_decision_fails(
    complete_run: Path, tmp_path: Path, demo: LocalDemoConfig
) -> None:
    """Forging an accept out of a run whose metrics do not support it."""
    run_dir = tampered_copy(complete_run, tmp_path)
    path = run_dir / "iteration-002/evaluation.json"
    evaluation = EvaluationOutcome.model_validate_json(path.read_text())
    # Claim a *worse* loss while keeping the recorded accept decision.
    worse = dict(evaluation.aggregate_metrics)
    worse["synthetic_validation_loss"] = worse["synthetic_validation_loss"] * 100
    forged = evaluation.model_copy(update={"aggregate_metrics": worse}).sealed()
    path.write_text(forged.canonical_text())
    checks = checks_of(run_dir, demo)
    assert "decision_consistency" in checks


def test_tampered_seal_fails(
    complete_run: Path, tmp_path: Path, demo: LocalDemoConfig
) -> None:
    run_dir = tampered_copy(complete_run, tmp_path)
    path = run_dir / "iteration-001/decision.json"
    decision = json.loads(path.read_text())
    decision["rationale"] = "Post-hoc rewritten rationale."
    path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    assert "seal_mismatch" in checks_of(run_dir, demo)


def test_artifact_integrity_failure(
    complete_run: Path, tmp_path: Path, demo: LocalDemoConfig
) -> None:
    run_dir = tampered_copy(complete_run, tmp_path)
    # Corrupt a registered prediction artifact without touching the manifest.
    target = run_dir / "iteration-001/workdir/predictions.json"
    target.write_text(target.read_text().replace("0.0", "0.1", 1))
    assert "artifact_integrity" in checks_of(run_dir, demo)


def test_inflated_scientific_status_fails(
    complete_run: Path, tmp_path: Path, demo: LocalDemoConfig
) -> None:
    run_dir = tampered_copy(complete_run, tmp_path)
    status_path = run_dir / "completion_status.json"
    status = json.loads(status_path.read_text())
    status["scientific_status"] = "publication_eligible"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    assert "unsupported_claims" in checks_of(run_dir, demo)


def test_missing_run_dir_fails(tmp_path: Path, demo: LocalDemoConfig) -> None:
    report = grade_trace(tmp_path / "does-not-exist", demo.acceptance)
    assert not report.passed


def test_decision_never_rederivable_from_forged_reject(
    complete_run: Path, tmp_path: Path, demo: LocalDemoConfig
) -> None:
    """Flipping an accept to reject (resealed) must break consistency or rollback."""
    run_dir = tampered_copy(complete_run, tmp_path)
    path = run_dir / "iteration-001/decision.json"
    decision = ExperimentDecision.model_validate_json(path.read_text())
    forged = decision.model_copy(
        update={
            "decision": decision.decision.__class__("reject"),
            "metric_result": decision.metric_result.model_copy(
                update={"passed": False, "detail": "forged regression"}
            ),
        }
    ).sealed()
    path.write_text(forged.canonical_text())
    checks = checks_of(run_dir, demo)
    assert {"decision_consistency", "rollback"} & checks
