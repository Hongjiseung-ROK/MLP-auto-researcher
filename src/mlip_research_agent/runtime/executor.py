"""Workflow executor: topological execution with validation, event sourcing,
checkpointing, and bounded recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.runtime.checkpoint import Checkpoint
from mlip_research_agent.runtime.events import EventLog
from mlip_research_agent.runtime.recovery import build_failure_record
from mlip_research_agent.schemas.claims import Claim
from mlip_research_agent.schemas.events import EventType
from mlip_research_agent.schemas.failure import (
    FailureClass,
    FailureRecord,
    RecoveryDecision,
    Severity,
)
from mlip_research_agent.schemas.workflow import WorkflowSpec
from mlip_research_agent.skills.base import Skill, SkillContext, SkillError, get_skill

CLAIMS_NAME = "claims.json"
_STEP_REF_PREFIX = "$steps."


class ExecutionAborted(Exception):
    def __init__(self, failure: FailureRecord, decision: RecoveryDecision) -> None:
        super().__init__(f"step {failure.skill_name} stopped with decision {decision.value}")
        self.failure = failure
        self.decision = decision


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["completed", "failed"]
    step_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    failures: list[FailureRecord] = Field(default_factory=list)
    final_decision: RecoveryDecision | None = None


class WorkflowExecutor:
    def __init__(
        self,
        workflow: WorkflowSpec,
        run_dir: Path,
        run_id: str,
        max_retries_per_step: int = 2,
    ) -> None:
        self.workflow = workflow
        self.run_dir = run_dir
        self.run_id = run_id
        self.max_retries_per_step = max_retries_per_step
        run_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventLog(run_dir, run_id)
        self.registry = ArtifactRegistry.load(run_dir)
        self.claims: list[Claim] = _load_claims(run_dir)

    def execute(self, resume: bool = False) -> RunResult:
        checkpoint = Checkpoint.load(self.run_dir) if resume else None
        if checkpoint is not None and checkpoint.workflow_hash != self.workflow.content_hash():
            raise ValueError(
                "checkpoint belongs to a different workflow "
                f"(hash {checkpoint.workflow_hash[:12]} != {self.workflow.content_hash()[:12]})"
            )
        if checkpoint is None:
            checkpoint = Checkpoint(
                run_id=self.run_id,
                workflow_hash=self.workflow.content_hash(),
                seed=self.workflow.seed,
            )

        result = RunResult(run_id=self.run_id, status="completed")
        result.step_outputs.update(checkpoint.completed_steps)
        self.events.emit(
            EventType.RUN_STARTED,
            payload={
                "workflow": self.workflow.name,
                "workflow_hash": self.workflow.content_hash(),
                "seed": self.workflow.seed,
                "resumed": resume,
            },
        )

        for step_id in self.workflow.topological_order():
            if step_id in checkpoint.completed_steps:
                self.events.emit(EventType.STEP_RESUMED_FROM_CHECKPOINT, step_id=step_id)
                continue
            try:
                outputs = self._execute_step(step_id, result)
            except ExecutionAborted as aborted:
                result.status = "failed"
                result.final_decision = aborted.decision
                self._persist(checkpoint)
                self.events.emit(
                    EventType.RUN_FAILED,
                    step_id=step_id,
                    payload={
                        "decision": aborted.decision.value,
                        "failure_id": aborted.failure.failure_id,
                    },
                )
                return result
            result.step_outputs[step_id] = outputs
            checkpoint.completed_steps[step_id] = outputs
            self._persist(checkpoint)
            self.events.emit(EventType.CHECKPOINT_SAVED, step_id=step_id)

        self.events.emit(EventType.RUN_COMPLETED, payload={"steps": len(self.workflow.steps)})
        return result

    def _execute_step(self, step_id: str, result: RunResult) -> dict[str, Any]:
        step = self.workflow.step(step_id)
        skill_cls = get_skill(step.skill)
        skill: Skill = skill_cls()
        attempt = 0
        repair_overrides: dict[str, Any] = {}
        attempted_repairs: list[str] = []

        while True:
            self.events.emit(
                EventType.STEP_STARTED,
                step_id=step_id,
                payload={"skill": step.skill, "attempt": attempt},
            )
            try:
                params = _resolve_params(step.params, result.step_outputs)
                params.update(repair_overrides)
                try:
                    inputs = skill_cls.input_model.model_validate(params)
                except ValidationError as exc:
                    raise SkillError(
                        f"input validation failed for skill {step.skill!r}: {exc}",
                        failure_class=FailureClass.VALIDATION_ERROR,
                        severity=Severity.HIGH,
                        retryable=False,
                    ) from exc
                ctx = SkillContext(
                    run_dir=self.run_dir,
                    step_id=step_id,
                    seed=self.workflow.seed,
                    attempt=attempt,
                    registry=self.registry,
                )
                raw_output = skill.run(inputs, ctx)
                if not isinstance(raw_output, skill_cls.output_model):
                    raise SkillError(
                        f"skill {step.skill!r} returned {type(raw_output).__name__}, "
                        f"expected {skill_cls.output_model.__name__}",
                        failure_class=FailureClass.VALIDATION_ERROR,
                        severity=Severity.HIGH,
                        retryable=False,
                    )
            except SkillError as error:
                attempt, repair_overrides = self._handle_failure(
                    error,
                    step_id=step_id,
                    skill_name=step.skill,
                    attempt=attempt,
                    repair_overrides=repair_overrides,
                    attempted_repairs=attempted_repairs,
                    result=result,
                )
                continue

            for claim in ctx.claims:
                self.claims.append(claim)
                self.events.emit(
                    EventType.CLAIM_REGISTERED,
                    step_id=step_id,
                    payload={"claim_id": claim.claim_id},
                )
            outputs: dict[str, Any] = raw_output.model_dump(mode="json")
            self.events.emit(
                EventType.STEP_COMPLETED,
                step_id=step_id,
                payload={"skill": step.skill, "attempt": attempt, "outputs": outputs},
            )
            return outputs

    def _handle_failure(
        self,
        error: SkillError,
        *,
        step_id: str,
        skill_name: str,
        attempt: int,
        repair_overrides: dict[str, Any],
        attempted_repairs: list[str],
        result: RunResult,
    ) -> tuple[int, dict[str, Any]]:
        """Record the failure and either return the next attempt state or abort."""
        remaining = self.max_retries_per_step - attempt
        record = build_failure_record(
            error,
            skill_name=skill_name,
            remaining_budget=max(remaining, 0),
            attempted_repairs=list(attempted_repairs),
            artifact_references=[a.artifact_id for a in self.registry.all()],
        )
        result.failures.append(record)
        self.events.emit(
            EventType.STEP_FAILED, step_id=step_id, payload=record.model_dump(mode="json")
        )
        decision = record.recommended_action
        self.events.emit(
            EventType.RECOVERY_DECISION,
            step_id=step_id,
            payload={
                "decision": decision.value,
                "failure_id": record.failure_id,
                "repair_params": error.repair_params,
            },
        )
        if decision is RecoveryDecision.RETRY:
            return attempt + 1, repair_overrides
        if decision is RecoveryDecision.REFINE:
            merged = {**repair_overrides, **error.repair_params}
            attempted_repairs.append(json.dumps(error.repair_params, sort_keys=True))
            return attempt + 1, merged
        raise ExecutionAborted(record, decision)

    def _persist(self, checkpoint: Checkpoint) -> None:
        checkpoint.save(self.run_dir)
        self.registry.save()
        _save_claims(self.run_dir, self.claims)


def _resolve_params(
    params: dict[str, Any], step_outputs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Resolve '$steps.<step_id>.<field>' references against completed outputs."""

    def resolve(value: Any) -> Any:
        if isinstance(value, str) and value.startswith(_STEP_REF_PREFIX):
            ref = value[len(_STEP_REF_PREFIX) :]
            step_id, _, field_name = ref.partition(".")
            if not field_name:
                raise ValueError(f"malformed step reference {value!r}")
            if step_id not in step_outputs:
                raise ValueError(f"reference {value!r} points to incomplete step {step_id!r}")
            outputs = step_outputs[step_id]
            if field_name not in outputs:
                raise ValueError(f"step {step_id!r} has no output field {field_name!r}")
            return outputs[field_name]
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return {k: resolve(v) for k, v in params.items()}


def _load_claims(run_dir: Path) -> list[Claim]:
    path = run_dir / CLAIMS_NAME
    if not path.is_file():
        return []
    return [Claim.model_validate(item) for item in json.loads(path.read_text())]


def _save_claims(run_dir: Path, claims: list[Claim]) -> Path:
    path = run_dir / CLAIMS_NAME
    payload = [c.model_dump(mode="json") for c in sorted(claims, key=lambda c: c.claim_id)]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
