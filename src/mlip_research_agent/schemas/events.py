"""Append-only execution events (OpenHands-inspired event sourcing)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_RESUMED_FROM_CHECKPOINT = "step_resumed_from_checkpoint"
    RECOVERY_DECISION = "recovery_decision"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CLAIM_REGISTERED = "claim_registered"
    APPROVAL_EVENT = "approval_event"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    run_id: str
    event_type: EventType
    step_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="microseconds")
    )
