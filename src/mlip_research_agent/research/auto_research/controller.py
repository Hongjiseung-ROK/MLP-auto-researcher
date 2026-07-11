"""Auto Research Controller managing the outer research loop."""

from __future__ import annotations

import json
from pathlib import Path

from mlip_research_agent.research.auto_research.round_state import RoundState


class ControllerState:
    """The saved persistent state of the research controller."""
    def __init__(self, current_round: int, round_state: RoundState) -> None:
        self.current_round = current_round
        self.round_state = round_state

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "current_round": self.current_round,
            "round_state": self.round_state.value
        }, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> ControllerState | None:
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        return cls(
            current_round=data["current_round"],
            round_state=RoundState(data["round_state"])
        )


class AutoResearchController:
    """Generic multi-iteration auto-research loop."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.state_file = run_dir / "controller_state.json"
        self.state = ControllerState.load(self.state_file) or ControllerState(
            1, RoundState.INITIALIZED
        )

    def advance(self) -> None:
        """Advance the controller state machine one step."""
        if self.state.round_state == RoundState.COMPLETE:
            return

        if self.state.round_state == RoundState.INITIALIZED:
            # Create a bounded proposal
            self._transition(RoundState.PROPOSAL_READY)
        elif self.state.round_state == RoundState.PROPOSAL_READY:
            # Run Tea Time
            self._transition(RoundState.TEA_TIME_REVIEWED)
        elif self.state.round_state == RoundState.TEA_TIME_REVIEWED:
            # Validate mutation legality
            self._transition(RoundState.LEGALITY_APPROVED)
        elif self.state.round_state == RoundState.LEGALITY_APPROVED:
            # Apply mutation and trigger execution
            self._transition(RoundState.EXECUTING)
        elif self.state.round_state == RoundState.EXECUTING:
            # Check execution status (mocking completion for now)
            self._transition(RoundState.EXECUTED)
        elif self.state.round_state == RoundState.EXECUTED:
            # Evaluate independently
            self._transition(RoundState.EVALUATED)
        elif self.state.round_state == RoundState.EVALUATED:
            # Decide (accept/reject/refine)
            self._transition(RoundState.DECIDED)
        elif self.state.round_state == RoundState.DECIDED:
            # Record lesson
            self._transition(RoundState.LESSON_RECORDED)
        elif self.state.round_state == RoundState.LESSON_RECORDED:
            if self.state.current_round >= 2:
                self._transition(RoundState.COMPLETE)
            else:
                self._transition(RoundState.NEXT_PROPOSAL_READY)
        elif self.state.round_state == RoundState.NEXT_PROPOSAL_READY:
            # Advance to next round
            self.state.current_round += 1
            self._transition(RoundState.PROPOSAL_READY)

    def _transition(self, new_state: RoundState) -> None:
        self.state.round_state = new_state
        self.state.save(self.state_file)

    def run(self) -> None:
        """Run the controller until blocked or complete."""
        while self.state.round_state not in (
            RoundState.COMPLETE,
            RoundState.BLOCKED,
            RoundState.FAILED,
        ):
            self.advance()
