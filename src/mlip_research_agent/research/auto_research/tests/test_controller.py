"""Tests for auto_research controller."""
import json
from pathlib import Path

from mlip_research_agent.research.auto_research.controller import (
    AutoResearchController,
)
from mlip_research_agent.research.auto_research.round_state import RoundState


def test_controller_initialization_and_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    controller = AutoResearchController(run_dir)
    assert controller.state.current_round == 1
    assert controller.state.round_state == RoundState.INITIALIZED

    # Test advance
    controller.advance()
    assert controller.state.round_state == RoundState.PROPOSAL_READY  # type: ignore[comparison-overlap]

    # Advance to NEXT_PROPOSAL_READY
    controller.advance() # TEA_TIME_REVIEWED
    controller.advance() # LEGALITY_APPROVED
    controller.advance() # EXECUTING
    controller.advance() # EXECUTED
    controller.advance() # EVALUATED
    controller.advance() # DECIDED
    controller.advance() # LESSON_RECORDED
    controller.advance() # NEXT_PROPOSAL_READY

    # It should transition to NEXT_PROPOSAL_READY since round = 1 < 2
    assert controller.state.round_state == RoundState.NEXT_PROPOSAL_READY

    # Advance to NEXT round PROPOSAL_READY
    controller.advance()
    assert controller.state.current_round == 2
    assert controller.state.round_state == RoundState.PROPOSAL_READY

    # Advance until LESSON_RECORDED in round 2
    for _ in range(7):
        controller.advance()

    assert controller.state.round_state == RoundState.LESSON_RECORDED
    # It should now transition to COMPLETE since round >= 2
    controller.advance()
    assert controller.state.round_state == RoundState.COMPLETE

    # Advance should no-op
    controller.advance()
    assert controller.state.round_state == RoundState.COMPLETE

def test_controller_load_save(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    controller = AutoResearchController(run_dir)
    controller.advance()

    state_file = run_dir / "controller_state.json"
    assert state_file.exists()

    data = json.loads(state_file.read_text())
    assert data["current_round"] == 1
    assert data["round_state"] == RoundState.PROPOSAL_READY

    # Re-initialize to test loading
    controller2 = AutoResearchController(run_dir)
    assert controller2.state.current_round == 1
    assert controller2.state.round_state == RoundState.PROPOSAL_READY
