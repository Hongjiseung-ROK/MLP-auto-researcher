"""Typed, auditable Auto Research (Ralphthon) loop.

ResearchObjective → ExperimentProposal → TeaTimeReview → MutationValidation →
ExperimentExecution → IndependentEvaluation → ExperimentDecision →
ResearchLesson → NextExperimentProposal.

Every node is content-addressed and registered; the controller is benchmark-
agnostic; evaluation is an independent boundary; mutations are fail-closed
and transactional; Tea Time output remains agent text.
"""

from mlip_research_agent.research.auto_research.acceptance_policy import (
    ACCEPTANCE_POLICY_VERSION,
    AcceptanceConstraints,
    decide,
)
from mlip_research_agent.research.auto_research.adapters import (
    AdapterRunResult,
    BenchmarkAdapter,
    SyntheticQuadraticAdapter,
)
from mlip_research_agent.research.auto_research.controller import (
    AutoResearchController,
    ControllerError,
    EvaluatorBoundary,
)
from mlip_research_agent.research.auto_research.decision import (
    DecisionValue,
    DimensionAssessment,
    ExperimentDecision,
)
from mlip_research_agent.research.auto_research.evaluation import (
    BaselineReference,
    ConstraintResult,
    EvaluationOutcome,
)
from mlip_research_agent.research.auto_research.execution import (
    EventLogRange,
    ExperimentExecution,
    ResourceUsage,
)
from mlip_research_agent.research.auto_research.lesson import ResearchLesson
from mlip_research_agent.research.auto_research.lineage import (
    ExperimentLineage,
    IterationLineage,
    LineageNode,
    LineageNodeKind,
)
from mlip_research_agent.research.auto_research.mutation import (
    AllowedRange,
    LegalityResult,
    Mutation,
    MutationClass,
    MutationDiff,
    MutationPolicy,
    TransactionalConfigStore,
)
from mlip_research_agent.research.auto_research.objective import (
    LocalDemoConfig,
    ObjectiveBudget,
    ProtectedConstant,
    ResearchObjective,
    StoppingRule,
    SuccessCriterion,
)
from mlip_research_agent.research.auto_research.proposal import (
    EstimatedCompute,
    ExpectedObservable,
    ExperimentProposal,
)
from mlip_research_agent.research.auto_research.proposal_policy import (
    PROPOSAL_POLICY_VERSION,
    PriorOutcome,
    generate_proposal,
)
from mlip_research_agent.research.auto_research.review import (
    AgentReview,
    ReviewPacket,
    ReviewSynthesis,
)
from mlip_research_agent.research.auto_research.round_state import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    CompletionStatus,
    InvalidTransitionError,
    IterationRecord,
    LoopState,
    RoundState,
)
from mlip_research_agent.research.auto_research.synthetic_evaluator import (
    SyntheticAggregateEvaluator,
)
from mlip_research_agent.research.auto_research.tea_time_boundary import (
    TeaTimeReviewRecord,
    run_tea_time_review,
    select_trigger,
)
from mlip_research_agent.research.auto_research.validators import (
    ContentAddressedModel,
    SealError,
    require_sealed,
)

__all__ = [
    "ACCEPTANCE_POLICY_VERSION",
    "ALLOWED_TRANSITIONS",
    "PROPOSAL_POLICY_VERSION",
    "TERMINAL_STATES",
    "AcceptanceConstraints",
    "AdapterRunResult",
    "AgentReview",
    "AllowedRange",
    "AutoResearchController",
    "BaselineReference",
    "BenchmarkAdapter",
    "CompletionStatus",
    "ConstraintResult",
    "ContentAddressedModel",
    "ControllerError",
    "DecisionValue",
    "DimensionAssessment",
    "EstimatedCompute",
    "EvaluationOutcome",
    "EvaluatorBoundary",
    "EventLogRange",
    "ExpectedObservable",
    "ExperimentDecision",
    "ExperimentExecution",
    "ExperimentLineage",
    "ExperimentProposal",
    "InvalidTransitionError",
    "IterationLineage",
    "IterationRecord",
    "LegalityResult",
    "LineageNode",
    "LineageNodeKind",
    "LocalDemoConfig",
    "LoopState",
    "Mutation",
    "MutationClass",
    "MutationDiff",
    "MutationPolicy",
    "ObjectiveBudget",
    "PriorOutcome",
    "ProtectedConstant",
    "ResearchLesson",
    "ResearchObjective",
    "ResourceUsage",
    "ReviewPacket",
    "ReviewSynthesis",
    "RoundState",
    "SealError",
    "StoppingRule",
    "SuccessCriterion",
    "SyntheticAggregateEvaluator",
    "SyntheticQuadraticAdapter",
    "TeaTimeReviewRecord",
    "TransactionalConfigStore",
    "decide",
    "generate_proposal",
    "require_sealed",
    "run_tea_time_review",
    "select_trigger",
]
