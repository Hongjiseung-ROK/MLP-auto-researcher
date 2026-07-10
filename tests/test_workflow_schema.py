"""Workflow DAG schema validation tests."""

import pytest
from pydantic import ValidationError

from mlip_research_agent.schemas.workflow import WorkflowSpec, WorkflowStep
from mlip_research_agent.workflows.compiler import compile_campaign


def step(step_id: str, depends_on: list[str] | None = None) -> WorkflowStep:
    return WorkflowStep(step_id=step_id, skill="mock_literature", depends_on=depends_on or [])


def test_duplicate_step_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate step ids"):
        WorkflowSpec(name="w", steps=[step("a"), step("a")])


def test_unknown_dependency_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown steps"):
        WorkflowSpec(name="w", steps=[step("a", ["ghost"])])


def test_self_dependency_rejected() -> None:
    with pytest.raises(ValidationError, match="depends on itself"):
        WorkflowSpec(name="w", steps=[step("a", ["a"])])


def test_cycle_rejected() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        WorkflowSpec(name="w", steps=[step("a", ["b"]), step("b", ["a"])])


def test_topological_order_respects_dependencies() -> None:
    spec = WorkflowSpec(name="w", steps=[step("c", ["a", "b"]), step("a"), step("b", ["a"])])
    order = spec.topological_order()
    assert order.index("a") < order.index("b") < order.index("c")


def test_compiled_example_campaign_is_valid_dag(example_campaign) -> None:  # type: ignore[no-untyped-def]
    workflow = compile_campaign(example_campaign)
    order = workflow.topological_order()
    assert order[0] in ("literature", "structures")
    assert order[-1] == "verification"
    labeling = workflow.step("labeling")
    assert labeling.params["inject_failure_times"] == 1


def test_failure_injection_unknown_step_rejected(example_campaign) -> None:  # type: ignore[no-untyped-def]
    campaign = example_campaign.model_copy(deep=True)
    campaign.failure_injection.step = "nonexistent"
    with pytest.raises(ValueError, match="not a step"):
        compile_campaign(campaign)
