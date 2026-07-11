"""Typed inputs/outputs for the deterministic research-pause skill."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TeaTimeTrigger(StrEnum):
    """Approved pause points before consequential research actions."""

    LOCAL_STATE_RECOVERY = "local_state_recovery"
    CHECKPOINT_E0_COMPLETE = "checkpoint_e0_complete"
    PREREGISTRATION_FREEZE = "preregistration_freeze"
    COLAB_STAGING_PRELAUNCH = "colab_staging_prelaunch"
    STAGED_FAILURE = "staged_failure"


class AlternativePath(BaseModel):
    """A bounded alternative considered without executing it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    summary: str = Field(min_length=5, max_length=500)
    cost: str = Field(min_length=2, max_length=200)
    risk: str = Field(min_length=2, max_length=300)


class OwnerQuestionDraft(BaseModel):
    """A consequential question draft; Tea Time never grants approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{0,31}$")
    question: str = Field(min_length=5, max_length=500)
    why_needed: str = Field(min_length=5, max_length=500)
    current_evidence: str = Field(min_length=5, max_length=1000)
    recommended_default: str = Field(min_length=1, max_length=500)
    choices: list[str] = Field(min_length=2, max_length=3)


def _default_alternatives() -> list[AlternativePath]:
    return [
        AlternativePath(
            path_id="boundary_first",
            summary="Keep the benchmark fixed and harden the next reusable boundary.",
            cost="low",
            risk="May postpone broader scientific exploration.",
        ),
        AlternativePath(
            path_id="bounded_comparison",
            summary="Compare one preregistered alternative before an irreversible choice.",
            cost="medium",
            risk="Adds evidence work but reduces single-path lock-in.",
        ),
    ]


class TeaTimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus_question: str = Field(
        min_length=5,
        max_length=500,
        description="The research question the agent has been staring at",
    )
    trigger: TeaTimeTrigger = TeaTimeTrigger.LOCAL_STATE_RECOVERY
    stage_objective: str = Field(
        default="Harden a reusable AI-research pipeline without expanding scientific claims.",
        min_length=5,
        max_length=1000,
    )
    benchmark_role: str = Field(
        default="Bounded pipeline-hardening scaffold, not the final scientific topic.",
        min_length=5,
        max_length=500,
    )
    reusable_components: list[str] = Field(default_factory=list, max_length=20)
    benchmark_specific_components: list[str] = Field(default_factory=list, max_length=20)
    alternatives: list[AlternativePath] = Field(
        default_factory=_default_alternatives, min_length=2, max_length=3
    )
    open_owner_questions: list[OwnerQuestionDraft] = Field(default_factory=list, max_length=10)
    data_path: None = Field(
        default=None,
        description=(
            "Reserved compatibility field. Tea Time cannot read labels, test metrics, "
            "secrets, or other data artifacts."
        ),
    )
    n_provocations: int = Field(default=4, ge=1, le=7)
    poem_key: str | None = Field(
        default=None, description="Override the seeded poem choice (must exist in the corpus)"
    )


class TeaTimeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_artifact: str
    report_path: str
    reframings_artifact: str
    reframings_path: str
    poem_key: str
    techniques: list[str]
    n_provocations: int = Field(ge=1)
    trigger: TeaTimeTrigger
    purpose_summary: str
    benchmark_overfitting_risk: str
    alternative_path_ids: list[str] = Field(min_length=2, max_length=3)
    n_owner_questions: int = Field(ge=0)
