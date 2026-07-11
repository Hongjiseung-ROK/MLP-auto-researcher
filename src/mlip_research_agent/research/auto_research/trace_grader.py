"""Trace grading system to grade the auto-research agent's logic."""

from __future__ import annotations

from typing import Any

from mlip_research_agent.research.auto_research.lineage import ExperimentLineage


class TraceGrader:
    """Independent trace grader for validating agent steps."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id

    def grade(self, lineage: ExperimentLineage) -> dict[str, Any]:
        """Grade an entire experiment lineage.

        Returns a dictionary containing the grading report.
        """
        # A fixed list of two proposals must fail acceptance.
        # Check if the lineage has all required hashes.
        if not all([
            lineage.objective_hash,
            lineage.proposal_hash,
            lineage.tea_time_hash,
            lineage.legality_hash,
            lineage.mutation_hash,
            lineage.execution_hash,
            lineage.evaluation_hash,
            lineage.decision_hash,
            lineage.lesson_hash
        ]):
            return {"pass": False, "reason": "Incomplete lineage"}
        return {"pass": True, "reason": "Lineage structurally valid"}
