"""Executor tests: bounded retries, checkpoint resume, param wiring."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mlip_research_agent.runtime.executor import WorkflowExecutor, _resolve_params
from mlip_research_agent.schemas.events import EventType
from mlip_research_agent.schemas.failure import FailureClass, RecoveryDecision
from mlip_research_agent.schemas.workflow import WorkflowSpec, WorkflowStep
from mlip_research_agent.skills.base import Skill, SkillContext, SkillError, register_skill


class FlakyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    marker_name: str = "marker.txt"
    always_fail: bool = False


class FlakyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    succeeded_on_attempt: int


@register_skill
class FlakyTestSkill(Skill):
    """Fails until a marker file exists (simulates an interrupted run)."""

    name = "flaky_test_skill"
    input_model = FlakyInput
    output_model = FlakyOutput

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        assert isinstance(inputs, FlakyInput)
        marker = ctx.run_dir / inputs.marker_name
        if inputs.always_fail or not marker.is_file():
            marker.write_text("attempted")
            raise SkillError(
                "flaky failure",
                failure_class=FailureClass.TOOL_ERROR,
                retryable=True,
            )
        return FlakyOutput(succeeded_on_attempt=ctx.attempt)


def flaky_workflow(**params: object) -> WorkflowSpec:
    return WorkflowSpec(
        name="flaky",
        seed=0,
        steps=[
            WorkflowStep(step_id="lit", skill="mock_literature", params={"query": "mlip papers"}),
            WorkflowStep(
                step_id="flaky", skill="flaky_test_skill", params=params, depends_on=["lit"]
            ),
        ],
    )


def test_retries_are_bounded(tmp_path: Path) -> None:
    workflow = flaky_workflow(always_fail=True)
    executor = WorkflowExecutor(workflow, tmp_path, run_id="r1", max_retries_per_step=2)
    result = executor.execute()
    assert result.status == "failed"
    assert result.final_decision is RecoveryDecision.ESCALATE
    events = executor.events.replay()
    starts = [e for e in events if e.event_type is EventType.STEP_STARTED and e.step_id == "flaky"]
    assert len(starts) == 3  # initial attempt + 2 bounded retries, never more
    failures = [e for e in events if e.event_type is EventType.STEP_FAILED]
    assert len(failures) == 3
    assert any(e.event_type is EventType.RUN_FAILED for e in events)


def test_checkpoint_resume_skips_completed_steps(tmp_path: Path) -> None:
    workflow = flaky_workflow()
    executor = WorkflowExecutor(workflow, tmp_path, run_id="r1", max_retries_per_step=0)
    first = executor.execute()
    assert first.status == "failed"
    assert "lit" in first.step_outputs  # literature completed and checkpointed

    resumed_executor = WorkflowExecutor(workflow, tmp_path, run_id="r1", max_retries_per_step=0)
    second = resumed_executor.execute(resume=True)
    assert second.status == "completed"
    events = resumed_executor.events.replay()
    resumed = [e for e in events if e.event_type is EventType.STEP_RESUMED_FROM_CHECKPOINT]
    assert [e.step_id for e in resumed] == ["lit"]
    # The literature skill ran exactly once across both executions.
    lit_starts = [
        e for e in events if e.event_type is EventType.STEP_STARTED and e.step_id == "lit"
    ]
    assert len(lit_starts) == 1


def test_resume_rejects_changed_workflow(tmp_path: Path) -> None:
    executor = WorkflowExecutor(flaky_workflow(), tmp_path, run_id="r1", max_retries_per_step=0)
    executor.execute()
    changed = flaky_workflow(marker_name="other.txt")
    other = WorkflowExecutor(changed, tmp_path, run_id="r1", max_retries_per_step=0)
    try:
        other.execute(resume=True)
        raise AssertionError("expected ValueError for changed workflow")
    except ValueError as exc:
        assert "different workflow" in str(exc)


def test_step_reference_resolution() -> None:
    outputs = {"structures": {"structures_path": "steps/structures/structures.json"}}
    params = {
        "structures_path": "$steps.structures.structures_path",
        "nested": {"path": "$steps.structures.structures_path"},
        "plain": 5,
    }
    resolved = _resolve_params(params, outputs)
    assert resolved["structures_path"] == "steps/structures/structures.json"
    assert resolved["nested"]["path"] == "steps/structures/structures.json"
    assert resolved["plain"] == 5
