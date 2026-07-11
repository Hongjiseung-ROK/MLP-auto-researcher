"""Machine-readable Phase 2a experiment matrix.

Validators enforce the fairness invariants from plan_phase_2.md §6.2/§16:
random and hybrid receive identical initial data, added-label budget, round
count, and fine-tuning protocol; zero-shot and static receive no acquisition.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArmName(StrEnum):
    ZERO_SHOT = "zero_shot"
    STATIC_FINETUNE = "static_finetune"
    RANDOM = "random"
    HYBRID = "hybrid"


class AcquisitionPolicy(StrEnum):
    NONE = "none"
    RANDOM = "random"
    HYBRID_UNCERTAINTY_DIVERSITY = "hybrid_uncertainty_diversity"


class ArmSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ArmName
    initial_labels: int = Field(ge=0)
    added_label_budget: int = Field(ge=0)
    acquisition_rounds: int = Field(ge=0)
    acquisition_policy: AcquisitionPolicy
    finetune: bool
    selection_seed: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> ArmSpec:
        acquires = self.acquisition_policy is not AcquisitionPolicy.NONE
        if acquires and (self.added_label_budget == 0 or self.acquisition_rounds == 0):
            raise ValueError(f"arm {self.name}: acquiring arm needs budget and rounds")
        if not acquires and (self.added_label_budget or self.acquisition_rounds):
            raise ValueError(f"arm {self.name}: non-acquiring arm must have zero budget/rounds")
        return self


class ExperimentMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arms: list[ArmSpec] = Field(min_length=1)
    ensemble_seeds: list[int] = Field(min_length=3, max_length=3)
    acquisition_batch_min: int = Field(gt=0)
    acquisition_batch_max: int = Field(gt=0)

    @model_validator(mode="after")
    def _fairness(self) -> ExperimentMatrix:
        names = [arm.name for arm in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("duplicate arm names")

        by_name = {arm.name: arm for arm in self.arms}
        required = {ArmName.ZERO_SHOT, ArmName.STATIC_FINETUNE, ArmName.RANDOM, ArmName.HYBRID}
        missing = required - set(by_name)
        if missing:
            raise ValueError(f"Phase 2a requires all four arms; missing: {sorted(missing)}")

        zero = by_name[ArmName.ZERO_SHOT]
        if zero.finetune or zero.initial_labels or zero.added_label_budget:
            raise ValueError("zero_shot arm must have no labels and no fine-tuning")

        static = by_name[ArmName.STATIC_FINETUNE]
        if not static.finetune or static.added_label_budget:
            raise ValueError("static_finetune arm fine-tunes on the initial set only")

        random_arm = by_name[ArmName.RANDOM]
        hybrid = by_name[ArmName.HYBRID]
        if random_arm.acquisition_policy is not AcquisitionPolicy.RANDOM:
            raise ValueError("random arm must use the random policy")
        if hybrid.acquisition_policy is not AcquisitionPolicy.HYBRID_UNCERTAINTY_DIVERSITY:
            raise ValueError("hybrid arm must use the hybrid policy")
        # The core fairness contract: equal budgets everywhere it matters.
        for field in ("initial_labels", "added_label_budget", "acquisition_rounds"):
            if getattr(random_arm, field) != getattr(hybrid, field):
                raise ValueError(f"random and hybrid arms must have equal {field}")
        if static.initial_labels != random_arm.initial_labels:
            raise ValueError("static and acquiring arms must share the same initial set size")

        if self.acquisition_batch_max < self.acquisition_batch_min:
            raise ValueError("acquisition_batch_max must be >= acquisition_batch_min")
        if len(set(self.ensemble_seeds)) != len(self.ensemble_seeds):
            raise ValueError("ensemble seeds must be distinct")
        return self


def default_phase2a_matrix() -> ExperimentMatrix:
    """The preregistered Phase 2a matrix (plan_phase_2.md §16)."""
    return ExperimentMatrix(
        arms=[
            ArmSpec(
                name=ArmName.ZERO_SHOT,
                initial_labels=0,
                added_label_budget=0,
                acquisition_rounds=0,
                acquisition_policy=AcquisitionPolicy.NONE,
                finetune=False,
                selection_seed=0,
            ),
            ArmSpec(
                name=ArmName.STATIC_FINETUNE,
                initial_labels=32,
                added_label_budget=0,
                acquisition_rounds=0,
                acquisition_policy=AcquisitionPolicy.NONE,
                finetune=True,
                selection_seed=0,
            ),
            ArmSpec(
                name=ArmName.RANDOM,
                initial_labels=32,
                added_label_budget=32,
                acquisition_rounds=3,
                acquisition_policy=AcquisitionPolicy.RANDOM,
                finetune=True,
                selection_seed=1042,
            ),
            ArmSpec(
                name=ArmName.HYBRID,
                initial_labels=32,
                added_label_budget=32,
                acquisition_rounds=3,
                acquisition_policy=AcquisitionPolicy.HYBRID_UNCERTAINTY_DIVERSITY,
                finetune=True,
                selection_seed=2042,
            ),
        ],
        ensemble_seeds=[42, 43, 44],
        acquisition_batch_min=8,
        acquisition_batch_max=16,
    )
