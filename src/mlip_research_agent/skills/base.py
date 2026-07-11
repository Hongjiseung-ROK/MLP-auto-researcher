"""Skill contract: typed inputs/outputs, validated execution, typed failures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.schemas.claims import MINTABLE_CLAIM_CLASSES, Claim
from mlip_research_agent.schemas.failure import FailureClass, RecoveryDecision, Severity

M = TypeVar("M", bound=BaseModel)


class SkillError(Exception):
    """Typed skill failure carrying everything needed to build a FailureRecord."""

    def __init__(
        self,
        message: str,
        *,
        failure_class: FailureClass,
        severity: Severity = Severity.MEDIUM,
        retryable: bool = True,
        likely_causes: list[str] | None = None,
        recommended_action: RecoveryDecision | None = None,
        repair_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.severity = severity
        self.retryable = retryable
        self.likely_causes = likely_causes or []
        self.recommended_action = recommended_action
        # Parameter overrides a REFINE recovery applies on the next attempt.
        # The executor records them so no convergence criterion changes silently.
        self.repair_params = repair_params or {}


@dataclass
class SkillContext:
    """Everything a skill may touch. Skills must not write outside step_dir."""

    run_dir: Path
    step_id: str
    seed: int
    attempt: int
    registry: ArtifactRegistry
    claims: list[Claim] = field(default_factory=list)

    @property
    def step_dir(self) -> Path:
        d = self.run_dir / "steps" / self.step_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def register_claim(self, claim: Claim) -> None:
        if claim.claim_class not in MINTABLE_CLAIM_CLASSES:
            raise SkillError(
                f"Phase 2 cannot mint {claim.claim_class.value}; elevated claims require "
                "replication and human release outside the run",
                failure_class=FailureClass.UNSUPPORTED_CLAIM,
                severity=Severity.HIGH,
                retryable=False,
            )
        self.claims.append(claim)


class Skill(ABC):
    """A bounded, typed capability. Subclasses set the class attributes and
    implement run(); the executor performs input/output validation."""

    name: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]
    cost_class: ClassVar[str] = "trivial"  # trivial | cheap | expensive | gated
    permission_level: ClassVar[str] = "auto"  # auto | human_approval

    @abstractmethod
    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        """Execute with validated inputs; return an instance of output_model."""


def expect_inputs(inputs: BaseModel, model: type[M]) -> M:
    """Narrow the validated input model inside a skill implementation."""
    if not isinstance(inputs, model):
        raise SkillError(
            f"expected {model.__name__}, got {type(inputs).__name__}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    return inputs


_REGISTRY: dict[str, type[Skill]] = {}


def register_skill(cls: type[Skill]) -> type[Skill]:
    if cls.name in _REGISTRY:
        raise ValueError(f"skill {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_skill(name: str) -> type[Skill]:
    _ensure_builtin_skills_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"unknown skill {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_skills() -> dict[str, type[Skill]]:
    _ensure_builtin_skills_loaded()
    return dict(_REGISTRY)


def _ensure_builtin_skills_loaded() -> None:
    # Import for registration side effects; local import avoids a cycle.
    from mlip_research_agent.skills import (  # noqa: F401
        active_learning,
        atomistics,
        compute,
        data,
        dft,
        evaluation,
        external_adapters,
        literature,
        mlip,
        reflection,
        verification,
    )
