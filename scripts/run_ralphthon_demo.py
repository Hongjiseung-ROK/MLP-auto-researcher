#!/usr/bin/env python3
"""Run the local RALPHTON two-iteration demonstration."""

from pathlib import Path

from mlip_research_agent.research.auto_research.controller import AutoResearchController
from mlip_research_agent.research.auto_research.round_state import RoundState


def main() -> None:
    print("Starting Local RALPHTON Auto Research Demonstration...")
    run_dir = Path("artifacts/research/demo")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    controller = AutoResearchController(run_dir)
    print(f"Initial State: Round {controller.state.current_round}, State: {controller.state.round_state.value}")
    
    while controller.state.round_state not in (RoundState.COMPLETE, RoundState.BLOCKED, RoundState.FAILED):
        controller.advance()
        print(f"Advanced to: Round {controller.state.current_round}, State: {controller.state.round_state.value}")
    
    print("Demonstration finished.")


if __name__ == "__main__":
    main()
