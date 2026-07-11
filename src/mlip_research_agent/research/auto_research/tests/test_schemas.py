"""Tests for auto_research schemas."""
import hashlib
import json

from mlip_research_agent.research.auto_research.decision import DecisionType, ExperimentDecision
from mlip_research_agent.research.auto_research.objective import ResearchObjective
from mlip_research_agent.research.auto_research.proposal import ExperimentProposal


def test_research_objective_content_hash() -> None:
    objective = ResearchObjective(
        objective_id="obj_1",
        research_goal="Improve efficiency",
        scientific_domain="physics",
        benchmark_adapter="adapter_x",
        success_criteria="x > 5",
        budgets={"time": 100},
        maximum_iterations=10,
        protected_constants=["c"],
        allowed_mutation_classes=["A"],
        evaluator_identity="eval_1",
        stopping_rules={"max_time": 100},
        approval_requirements=["manual"],
        scientific_status_ceiling="pending"
    )
    h = objective.content_hash()
    canonical = json.dumps(objective.model_dump(mode="json"), sort_keys=True)
    expected_hash = hashlib.sha256(canonical.encode()).hexdigest()
    assert h == expected_hash

def test_experiment_proposal_content_hash() -> None:
    proposal = ExperimentProposal(
        proposal_id="prop_1",
        parent_objective_id="obj_1",
        explicit_hypothesis="H1",
        mutation={"param": "value"},
        expected_mechanism="mech_1",
        predicted_observable="obs_1",
        falsification_condition="cond_1",
        estimated_compute="1h",
        affected_config_paths=["path1"],
        risk_class="low",
        required_approval="none",
        seed=42,
        proposal_source="agent"
    )
    h = proposal.content_hash()
    canonical = json.dumps(proposal.model_dump(mode="json"), sort_keys=True)
    expected_hash = hashlib.sha256(canonical.encode()).hexdigest()
    assert h == expected_hash

def test_experiment_decision_content_hash() -> None:
    decision = ExperimentDecision(
        decision=DecisionType.ACCEPT,
        legality=True,
        engineering_success=True,
        validation_improvement=True,
        cost=10.0,
        reproducibility=True,
        scientific_uncertainty=0.1
    )
    h = decision.content_hash()
    canonical = json.dumps(decision.model_dump(mode="json"), sort_keys=True)
    expected_hash = hashlib.sha256(canonical.encode()).hexdigest()
    assert h == expected_hash
