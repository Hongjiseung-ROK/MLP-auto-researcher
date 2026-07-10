"""Compile a validated CampaignSpec into a declarative workflow DAG.

Deterministic, rule-based compilation. The LLM planner (post-v0) will sit in
front of this: it drafts/parameterizes the CampaignSpec, never the DAG wiring.
"""

from __future__ import annotations

from typing import Any

from mlip_research_agent.schemas.campaign import CampaignMode, CampaignSpec
from mlip_research_agent.schemas.workflow import WorkflowSpec, WorkflowStep
from mlip_research_agent.skills.base import get_skill


def compile_campaign(spec: CampaignSpec) -> WorkflowSpec:
    n_select = min(spec.budget.max_labels, spec.n_candidates)
    strategy = "random" if spec.mode is CampaignMode.FINE_TUNE_ONLY else "mock_uncertainty"

    steps = [
        WorkflowStep(
            step_id="literature",
            skill="mock_literature",
            params={"query": spec.description or spec.name, "max_results": 5},
        ),
        WorkflowStep(
            step_id="structures",
            skill="structure_generation",
            params={
                "formula": spec.target_system.formula,
                "crystal_structure": spec.target_system.crystal_structure,
                "lattice_constant": spec.target_system.lattice_constant,
                "supercell": list(spec.target_system.supercell),
                "n_candidates": spec.n_candidates,
            },
        ),
        WorkflowStep(
            step_id="selection",
            skill="mock_acquisition",
            params={
                "structures_path": "$steps.structures.structures_path",
                "n_select": n_select,
                "strategy": strategy,
            },
            depends_on=["structures"],
        ),
        WorkflowStep(
            step_id="labeling",
            skill="mock_dft_labeling",
            params={
                "structures_path": "$steps.structures.structures_path",
                "selection_path": "$steps.selection.selection_path",
                "method": "mock_lj",
            },
            depends_on=["structures", "selection"],
        ),
        WorkflowStep(
            step_id="training",
            skill="mock_mlip_training",
            params={"labels_path": "$steps.labeling.labels_path"},
            depends_on=["labeling"],
        ),
        WorkflowStep(
            step_id="evaluation",
            skill="mock_evaluation",
            params={
                "model_path": "$steps.training.model_path",
                "labels_path": "$steps.labeling.labels_path",
            },
            depends_on=["training", "labeling"],
        ),
        WorkflowStep(
            step_id="verification",
            skill="claim_verification",
            params={"claims_path": "claims.json", "strict": True},
            depends_on=["evaluation"],
        ),
    ]

    if spec.failure_injection is not None:
        _inject_failure(steps, spec.failure_injection.step, spec.failure_injection.times)

    return WorkflowSpec(name=f"{spec.name}-workflow", seed=spec.seed, steps=steps)


def _inject_failure(steps: list[WorkflowStep], step_id: str, times: int) -> None:
    for step in steps:
        if step.step_id != step_id:
            continue
        skill_cls = get_skill(step.skill)
        if "inject_failure_times" not in skill_cls.input_model.model_fields:
            raise ValueError(
                f"step {step_id!r} (skill {step.skill!r}) does not support failure injection"
            )
        params: dict[str, Any] = dict(step.params)
        params["inject_failure_times"] = times
        step.params = params
        return
    raise ValueError(f"failure_injection.step {step_id!r} is not a step in the compiled workflow")
