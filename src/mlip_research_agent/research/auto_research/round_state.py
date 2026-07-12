"""Controller state machine + resumable round state.

The state graph is explicit and fail-closed: a transition not listed in
``ALLOWED_TRANSITIONS`` raises. The round state is checkpointed after every
transition so an interrupted run resumes deterministically at the exact
boundary it stopped on, without repeating an optimizer step, oracle reveal,
or decision.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROUND_STATE_NAME = "round_state.json"


class LoopState(StrEnum):
    INITIALIZED = "initialized"
    PROPOSAL_READY = "proposal_ready"
    TEA_TIME_REVIEWED = "tea_time_reviewed"
    LEGALITY_APPROVED = "legality_approved"
    MUTATION_APPLIED = "mutation_applied"
    EXECUTING = "executing"
    EXECUTED = "executed"
    EVALUATED = "evaluated"
    DECIDED = "decided"
    ROLLED_BACK = "rolled_back"
    LESSON_RECORDED = "lesson_recorded"
    NEXT_PROPOSAL_READY = "next_proposal_ready"
    AWAITING_EXTERNAL_REVIEW = "awaiting_external_review"
    COMPLETE = "complete"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"


#: Terminal states: nothing transitions out of them.
TERMINAL_STATES: frozenset[LoopState] = frozenset(
    {LoopState.COMPLETE, LoopState.PARTIAL, LoopState.BLOCKED, LoopState.FAILED}
)

ALLOWED_TRANSITIONS: dict[LoopState, frozenset[LoopState]] = {
    LoopState.INITIALIZED: frozenset({LoopState.PROPOSAL_READY, LoopState.BLOCKED}),
    LoopState.PROPOSAL_READY: frozenset(
        {LoopState.TEA_TIME_REVIEWED, LoopState.BLOCKED, LoopState.FAILED}
    ),
    LoopState.TEA_TIME_REVIEWED: frozenset(
        # An illegal proposal short-circuits to DECIDED (reject) without
        # touching the config store.
        {LoopState.LEGALITY_APPROVED, LoopState.DECIDED, LoopState.BLOCKED}
    ),
    LoopState.LEGALITY_APPROVED: frozenset({LoopState.MUTATION_APPLIED, LoopState.FAILED}),
    LoopState.MUTATION_APPLIED: frozenset({LoopState.EXECUTING, LoopState.FAILED}),
    LoopState.EXECUTING: frozenset({LoopState.EXECUTED, LoopState.FAILED}),
    LoopState.EXECUTED: frozenset({LoopState.EVALUATED, LoopState.DECIDED, LoopState.FAILED}),
    LoopState.EVALUATED: frozenset({LoopState.DECIDED, LoopState.FAILED}),
    LoopState.DECIDED: frozenset(
        {LoopState.ROLLED_BACK, LoopState.LESSON_RECORDED, LoopState.BLOCKED}
    ),
    LoopState.ROLLED_BACK: frozenset({LoopState.LESSON_RECORDED}),
    LoopState.LESSON_RECORDED: frozenset(
        # BLOCKED: an escalate decision needs owner input before any new proposal.
        {
            LoopState.NEXT_PROPOSAL_READY,
            LoopState.AWAITING_EXTERNAL_REVIEW,
            LoopState.COMPLETE,
            LoopState.PARTIAL,
            LoopState.BLOCKED,
        }
    ),
    LoopState.NEXT_PROPOSAL_READY: frozenset(
        {LoopState.TEA_TIME_REVIEWED, LoopState.BLOCKED, LoopState.FAILED}
    ),
    LoopState.AWAITING_EXTERNAL_REVIEW: frozenset(
        {LoopState.NEXT_PROPOSAL_READY, LoopState.BLOCKED, LoopState.FAILED}
    ),
    LoopState.COMPLETE: frozenset(),
    LoopState.PARTIAL: frozenset(),
    LoopState.BLOCKED: frozenset(),
    LoopState.FAILED: frozenset(),
}


class InvalidTransitionError(Exception):
    """Raised on any transition not present in ALLOWED_TRANSITIONS."""


class IterationRecord(BaseModel):
    """Progress bookkeeping for one iteration (ids only, no science)."""

    model_config = ConfigDict(extra="forbid")

    iteration_id: str = Field(pattern=r"^iteration-\d{3}$")
    proposal_id: str
    decision: str | None = None
    accepted_config_index: int | None = None
    completed: bool = False


class RoundState(BaseModel):
    """The checkpointed controller state. Saved after every transition."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    run_id: str = Field(min_length=3)
    objective_sha256: str = Field(min_length=64, max_length=64)
    state: LoopState = LoopState.INITIALIZED
    iteration_index: int = Field(default=0, ge=0)
    iterations: list[IterationRecord] = Field(default_factory=list)
    consecutive_failures: int = Field(default=0, ge=0)
    mutation_class_streak: str | None = None
    mutation_class_streak_length: int = Field(default=0, ge=0)
    compute_seconds_used: float = Field(default=0.0, ge=0)
    labels_used: int = Field(default=0, ge=0)
    partial_reason: str = ""
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    last_failure_category: str | None = None

    def transition(self, new_state: LoopState) -> None:
        allowed = ALLOWED_TRANSITIONS[self.state]
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"{self.state.value} -> {new_state.value} is not an allowed transition"
            )
        self.state = new_state

    def save(self, run_dir: Path) -> Path:
        path = run_dir / ROUND_STATE_NAME
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, run_dir: Path) -> RoundState:
        return cls.model_validate(json.loads((run_dir / ROUND_STATE_NAME).read_text()))

    @property
    def current_iteration_id(self) -> str:
        return f"iteration-{self.iteration_index + 1:03d}"


class CompletionStatus(BaseModel):
    """Honest terminal status of an Auto Research run (never inflated)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    run_id: str
    status: LoopState
    iterations_completed: int = Field(ge=0)
    iterations_planned: int = Field(gt=0)
    reason: str = Field(min_length=3, max_length=1000)
    scientific_status: str = "non_scientific"
    claim_eligible: bool = False

    def save(self, run_dir: Path) -> Path:
        if self.status not in TERMINAL_STATES:
            raise ValueError("completion status may only record a terminal state")
        path = run_dir / "completion_status.json"
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
        return path
