"""Typed schemas: campaign specs, workflows, events, failures, claims."""

from mlip_research_agent.schemas.campaign import (
    CampaignMode,
    CampaignSpec,
    LabelBudget,
    StoppingRule,
    TargetSystem,
)
from mlip_research_agent.schemas.claims import Claim, ClaimStatus
from mlip_research_agent.schemas.events import Event, EventType
from mlip_research_agent.schemas.failure import (
    FailureClass,
    FailureRecord,
    RecoveryDecision,
    Severity,
)
from mlip_research_agent.schemas.workflow import WorkflowSpec, WorkflowStep

__all__ = [
    "CampaignMode",
    "CampaignSpec",
    "Claim",
    "ClaimStatus",
    "Event",
    "EventType",
    "FailureClass",
    "FailureRecord",
    "LabelBudget",
    "RecoveryDecision",
    "Severity",
    "StoppingRule",
    "TargetSystem",
    "WorkflowSpec",
    "WorkflowStep",
]
