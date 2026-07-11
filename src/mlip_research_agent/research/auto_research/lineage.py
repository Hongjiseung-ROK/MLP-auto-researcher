"""ExperimentLineage: the auditable chain every iteration must complete.

objective → proposal → tea_time → legality → mutation → execution →
evaluation → decision → lesson → next proposal. Every node is registered and
content-addressed; the trace grader fails a missing or unverifiable node.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.research.auto_research.validators import (
    SHA256_HEX_LENGTH,
    ContentAddressedModel,
)


class LineageNodeKind(StrEnum):
    OBJECTIVE = "objective"
    PROPOSAL = "proposal"
    TEA_TIME = "tea_time"
    LEGALITY = "legality"
    MUTATION = "mutation"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    DECISION = "decision"
    LESSON = "lesson"
    NEXT_PROPOSAL = "next_proposal"


#: The complete in-iteration chain, in required order. EVALUATION may be
#: legitimately absent only when the iteration died before execution
#: (illegal mutation) — the grader enforces that exception explicitly.
ITERATION_CHAIN_ORDER: tuple[LineageNodeKind, ...] = (
    LineageNodeKind.PROPOSAL,
    LineageNodeKind.TEA_TIME,
    LineageNodeKind.LEGALITY,
    LineageNodeKind.MUTATION,
    LineageNodeKind.EXECUTION,
    LineageNodeKind.EVALUATION,
    LineageNodeKind.DECISION,
    LineageNodeKind.LESSON,
)


class LineageNode(BaseModel):
    """One content-addressed node in the audit chain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: LineageNodeKind
    node_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1, description="Id in the run's artifact registry")
    content_sha256: str = Field(min_length=SHA256_HEX_LENGTH, max_length=SHA256_HEX_LENGTH)


class IterationLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration_id: str = Field(pattern=r"^iteration-\d{3}$")
    parent_iteration_id: str | None
    nodes: list[LineageNode] = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered_chain(self) -> IterationLineage:
        kinds = [n.kind for n in self.nodes]
        if len(set(kinds)) != len(kinds):
            raise ValueError("an iteration cannot repeat a lineage node kind")
        order = {kind: i for i, kind in enumerate(ITERATION_CHAIN_ORDER)}
        positions = [order[k] for k in kinds if k in order]
        if positions != sorted(positions):
            raise ValueError("iteration lineage nodes are out of chain order")
        return self

    def node(self, kind: LineageNodeKind) -> LineageNode | None:
        for n in self.nodes:
            if n.kind is kind:
                return n
        return None


class ExperimentLineage(ContentAddressedModel):
    """Run-level chain: the objective plus every iteration, connected."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=3)
    objective_node: LineageNode
    iterations: list[IterationLineage]

    @model_validator(mode="after")
    def _connected(self) -> ExperimentLineage:
        if self.objective_node.kind is not LineageNodeKind.OBJECTIVE:
            raise ValueError("objective_node must be an objective lineage node")
        previous_id: str | None = None
        for it in self.iterations:
            if it.parent_iteration_id != previous_id:
                raise ValueError(
                    f"{it.iteration_id} does not descend from {previous_id!r}; "
                    "iterations must form one connected chain"
                )
            previous_id = it.iteration_id
        return self
