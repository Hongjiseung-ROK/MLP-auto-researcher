"""Deterministic planner stub.

This is the seam where the LLM planner will live (plan.md: hypothesis-to-
campaign compiler). In v0 it produces a conservative template campaign so the
whole pipeline is exercisable without any API key. The LLM's future job is to
draft and parameterize a CampaignSpec from a research goal; validation and
compilation stay deterministic.
"""

from __future__ import annotations

from mlip_research_agent.schemas.campaign import (
    CampaignMode,
    CampaignSpec,
    LabelBudget,
    StoppingRule,
    TargetSystem,
)


def draft_campaign(goal: str, name: str = "drafted-campaign") -> CampaignSpec:
    """Draft a template campaign for a research goal (no LLM in v0)."""
    return CampaignSpec(
        name=name,
        description=goal,
        mode=CampaignMode.ACTIVE_LEARNING,
        target_system=TargetSystem(formula="Cu", crystal_structure="fcc", lattice_constant=3.6),
        budget=LabelBudget(max_labels=8, max_retries_per_step=2),
        stopping=StoppingRule(max_iterations=1),
        seed=0,
    )
