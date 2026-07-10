"""Append-only JSONL event log with replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlip_research_agent.schemas.events import Event, EventType

EVENT_LOG_NAME = "events.jsonl"


class EventLog:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.path = run_dir / EVENT_LOG_NAME
        self.run_id = run_id
        self._next_sequence = 0
        if self.path.is_file():
            events = self.replay()
            if events:
                self._next_sequence = events[-1].sequence + 1

    def emit(
        self,
        event_type: EventType,
        step_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        event = Event(
            sequence=self._next_sequence,
            run_id=self.run_id,
            event_type=event_type,
            step_id=step_id,
            payload=payload or {},
        )
        with self.path.open("a") as fh:
            fh.write(event.model_dump_json() + "\n")
        self._next_sequence += 1
        return event

    def replay(self) -> list[Event]:
        """Reconstruct the full ordered event stream from disk."""
        if not self.path.is_file():
            return []
        events = [
            Event.model_validate_json(line)
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]
        for i, event in enumerate(events):
            if event.sequence != i:
                raise ValueError(
                    f"event log corrupted: expected sequence {i}, found {event.sequence}"
                )
        return events
