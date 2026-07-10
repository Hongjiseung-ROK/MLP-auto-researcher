"""Bounded retry and failure-classification tests."""

from mlip_research_agent.runtime.recovery import build_failure_record, decide
from mlip_research_agent.schemas.failure import FailureClass, RecoveryDecision, Severity
from mlip_research_agent.skills.base import SkillError


def make_error(**kw: object) -> SkillError:
    defaults: dict = {
        "failure_class": FailureClass.CONVERGENCE_FAILURE,
        "severity": Severity.MEDIUM,
        "retryable": True,
    }
    return SkillError("boom", **{**defaults, **kw})


def test_retryable_with_budget_retries() -> None:
    assert decide(make_error(), remaining_budget=2) is RecoveryDecision.RETRY


def test_retryable_with_repair_refines() -> None:
    error = make_error(repair_params={"scf_damping": 0.7})
    assert decide(error, remaining_budget=2) is RecoveryDecision.REFINE


def test_exhausted_budget_escalates() -> None:
    assert decide(make_error(), remaining_budget=0) is RecoveryDecision.ESCALATE


def test_critical_severity_aborts() -> None:
    assert decide(make_error(severity=Severity.CRITICAL), remaining_budget=5) is (
        RecoveryDecision.ABORT
    )


def test_unsupported_claim_escalates() -> None:
    error = make_error(failure_class=FailureClass.UNSUPPORTED_CLAIM, retryable=False)
    assert decide(error, remaining_budget=5) is RecoveryDecision.ESCALATE


def test_non_retryable_pivot_recommendation_respected() -> None:
    error = make_error(
        failure_class=FailureClass.OOD_BEHAVIOR,
        retryable=False,
        recommended_action=RecoveryDecision.PIVOT,
    )
    assert decide(error, remaining_budget=5) is RecoveryDecision.PIVOT


def test_failure_record_captures_evidence() -> None:
    error = make_error(likely_causes=["SCF oscillation"], repair_params={"scf_damping": 0.7})
    record = build_failure_record(
        error,
        skill_name="mock_dft_labeling",
        remaining_budget=1,
        attempted_repairs=["{\"scf_damping\": 0.5}"],
        artifact_references=["structures:structures.json"],
    )
    assert record.failure_class is FailureClass.CONVERGENCE_FAILURE
    assert record.retryable
    assert record.observed_evidence == "boom"
    assert record.likely_causes == ["SCF oscillation"]
    assert record.attempted_repairs == ["{\"scf_damping\": 0.5}"]
    assert record.remaining_budget == 1
    assert record.recommended_action is RecoveryDecision.REFINE
    assert record.artifact_references == ["structures:structures.json"]
