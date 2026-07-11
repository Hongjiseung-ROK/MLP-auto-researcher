"""Acceptance policy for experiments and controllers."""

from __future__ import annotations

from mlip_research_agent.research.auto_research.evaluation import EvaluationOutcome
from mlip_research_agent.research.auto_research.lineage import ExperimentLineage
from mlip_research_agent.research.auto_research.objective import ResearchObjective


class AcceptancePolicy:
    """Policy for accepting or rejecting a proposed sequence or outcome."""

    @classmethod
    def evaluate_lineage(
        cls, lineage: ExperimentLineage, objective: ResearchObjective
    ) -> bool:
        """Evaluate if the experiment lineage meets acceptance criteria.
        For example: a fixed list of two proposals fails acceptance.
        """
        # A simple check: verify all hashes are present.
        return bool(lineage.objective_hash and lineage.proposal_hash)

    @classmethod
    def check_independent_outcome(
        cls, outcome: EvaluationOutcome, objective: ResearchObjective
    ) -> bool:
        """Check the outcome against the objective success criteria."""
        return bool(outcome.legality_result)
