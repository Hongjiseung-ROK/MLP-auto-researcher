"""Event log append/replay tests."""

from pathlib import Path

import pytest

from mlip_research_agent.runtime.events import EventLog
from mlip_research_agent.schemas.events import EventType


def test_replay_roundtrip(tmp_path: Path) -> None:
    log = EventLog(tmp_path, run_id="run-1")
    log.emit(EventType.RUN_STARTED, payload={"seed": 7})
    log.emit(EventType.STEP_STARTED, step_id="s1", payload={"skill": "x", "attempt": 0})
    log.emit(EventType.STEP_COMPLETED, step_id="s1")

    events = EventLog(tmp_path, run_id="run-1").replay()
    assert [e.event_type for e in events] == [
        EventType.RUN_STARTED,
        EventType.STEP_STARTED,
        EventType.STEP_COMPLETED,
    ]
    assert [e.sequence for e in events] == [0, 1, 2]
    assert events[0].payload == {"seed": 7}


def test_append_continues_sequence_after_reload(tmp_path: Path) -> None:
    log = EventLog(tmp_path, run_id="run-1")
    log.emit(EventType.RUN_STARTED)
    reloaded = EventLog(tmp_path, run_id="run-1")
    event = reloaded.emit(EventType.RUN_COMPLETED)
    assert event.sequence == 1
    assert len(reloaded.replay()) == 2


def test_corrupted_sequence_detected(tmp_path: Path) -> None:
    log = EventLog(tmp_path, run_id="run-1")
    log.emit(EventType.RUN_STARTED)
    lines = log.path.read_text()
    log.path.write_text(lines.replace('"sequence":0', '"sequence":5'))
    with pytest.raises(ValueError, match="corrupted"):
        EventLog(tmp_path, run_id="run-1").replay()
