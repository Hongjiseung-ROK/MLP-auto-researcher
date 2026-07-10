"""Declarative workflow DAG schema (PARNESS-style, per plan.md)."""

from __future__ import annotations

import hashlib
import json
from graphlib import CycleError, TopologicalSorter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

STEP_ID_PATTERN = r"^[a-z0-9][a-z0-9_]{0,63}$"


class WorkflowStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=STEP_ID_PATTERN)
    skill: str = Field(min_length=1, description="Registered skill name")
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class WorkflowSpec(BaseModel):
    """A validated DAG of skill invocations compiled from a campaign."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1.0"
    name: str
    seed: int = 0
    steps: list[WorkflowStep] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dag(self) -> WorkflowSpec:
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate step ids: {dupes}")
        known = set(ids)
        for step in self.steps:
            missing = [d for d in step.depends_on if d not in known]
            if missing:
                raise ValueError(f"step {step.step_id!r} depends on unknown steps: {missing}")
            if step.step_id in step.depends_on:
                raise ValueError(f"step {step.step_id!r} depends on itself")
        self.topological_order()  # raises on cycles
        return self

    def topological_order(self) -> list[str]:
        """Deterministic topological order (ties broken by declaration order)."""
        sorter: TopologicalSorter[str] = TopologicalSorter()
        for step in self.steps:
            sorter.add(step.step_id, *step.depends_on)
        try:
            order = list(sorter.static_order())
        except CycleError as exc:
            raise ValueError(f"workflow contains a dependency cycle: {exc.args[1]}") from exc
        declared = {s.step_id: i for i, s in enumerate(self.steps)}
        # static_order is stable only per-insertion; enforce full determinism.
        return sorted(order, key=lambda sid: (_depth(self, sid), declared[sid]))

    def step(self, step_id: str) -> WorkflowStep:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        raise KeyError(step_id)

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _depth(spec: WorkflowSpec, step_id: str) -> int:
    step = spec.step(step_id)
    if not step.depends_on:
        return 0
    return 1 + max(_depth(spec, d) for d in step.depends_on)
