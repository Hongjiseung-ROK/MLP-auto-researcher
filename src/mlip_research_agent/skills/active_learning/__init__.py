"""Mock acquisition skill: uncertainty-proxy candidate selection."""

from mlip_research_agent.skills.active_learning.implementation import MockAcquisitionSkill
from mlip_research_agent.skills.active_learning.oracle_reveal import OracleRevealSkill

__all__ = ["MockAcquisitionSkill", "OracleRevealSkill"]
