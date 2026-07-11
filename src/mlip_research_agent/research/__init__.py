"""Phase 2 research protocol layer: preregistration, matrix, pilot status.

The scientific method is fixed here *before* results exist and cannot change
afterwards without a recorded PIVOT plus human approval (plan_phase_2.md §12).
"""

from mlip_research_agent.research.experiment_matrix import (
    ArmName,
    ArmSpec,
    ExperimentMatrix,
    default_phase2a_matrix,
)
from mlip_research_agent.research.pilot_status import (
    PilotStage,
    PilotStatus,
    RoundState,
)
from mlip_research_agent.research.preregistration import (
    ApprovalRecord,
    HumanGate,
    Preregistration,
    load_pilot_config,
    require_approval,
)

__all__ = [
    "ApprovalRecord",
    "ArmName",
    "ArmSpec",
    "ExperimentMatrix",
    "HumanGate",
    "PilotStage",
    "PilotStatus",
    "Preregistration",
    "RoundState",
    "default_phase2a_matrix",
    "load_pilot_config",
    "require_approval",
]
