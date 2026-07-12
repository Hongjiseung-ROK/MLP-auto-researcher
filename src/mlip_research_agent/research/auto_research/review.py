"""Typed host-side scientific reviews for the iteration-one pause."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.research.auto_research.validators import (
    SHA256_HEX_LENGTH,
    ContentAddressedModel,
)

ReviewRole = Literal["mlip_scientist", "active_learning_scientist", "scientific_auditor"]
ReviewStatus = Literal["continue", "refine", "stop", "escalate"]
ReplayMutationKey = Literal[
    "learning_rate",
    "gradient_clip",
    "batch_size",
    "scheduler_patience",
    "force_loss_weight",
    "energy_loss_weight",
    "trainable_layer_policy",
]


class AgentReview(ContentAddressedModel):
    model_config = ConfigDict(extra="forbid")
    role: ReviewRole
    review_status: ReviewStatus
    recommended_mutation_class: ReplayMutationKey
    recommended_direction: str = Field(min_length=2, max_length=200)
    scientific_rationale: str = Field(min_length=10, max_length=2000)
    engineering_rationale: str = Field(min_length=10, max_length=2000)
    risks: list[str]
    evidence_artifact_ids: list[str] = Field(min_length=1)
    review_packet_sha256: str = Field(min_length=64, max_length=64)
    limitations: list[str]
    evidence_class: Literal["agent_text"] = "agent_text"
    claim_eligible: Literal[False] = False


class ReviewSynthesis(ContentAddressedModel):
    model_config = ConfigDict(extra="forbid")
    iteration_id: Literal["iteration-001"] = "iteration-001"
    review_hashes: dict[ReviewRole, str]
    recommended_mutation_class: ReplayMutationKey
    recommended_direction: str = Field(min_length=2, max_length=200)
    rationale: str = Field(min_length=10, max_length=3000)
    accepted_recommendations: list[str]
    rejected_recommendations: list[str]
    iteration_evidence_sha256: dict[str, str]
    review_packet_sha256: str = Field(min_length=64, max_length=64)
    evidence_class: Literal["agent_text"] = "agent_text"
    claim_eligible: Literal[False] = False

    @model_validator(mode="after")
    def _complete(self) -> ReviewSynthesis:
        expected = {
            "mlip_scientist",
            "active_learning_scientist",
            "scientific_auditor",
        }
        if set(self.review_hashes) != expected:
            raise ValueError("synthesis must reference exactly the three approved review roles")
        if any(len(value) != SHA256_HEX_LENGTH for value in self.review_hashes.values()):
            raise ValueError("every review reference must be a SHA-256")
        required = {"proposal", "evaluation", "decision", "lesson"}
        if set(self.iteration_evidence_sha256) != required:
            raise ValueError("synthesis must bind proposal/evaluation/decision/lesson")
        return self


class ReviewPacket(ContentAddressedModel):
    """The only aggregate, label-free payload specialized reviewers may receive."""

    model_config = ConfigDict(extra="forbid")
    iteration_id: Literal["iteration-001"] = "iteration-001"
    artifact_payloads: dict[str, dict[str, object]]
    artifact_file_sha256: dict[str, str]
    allowed_evidence_artifact_ids: list[str] = Field(min_length=1)
    protected_records_included: Literal[False] = False
    claim_eligible: Literal[False] = False

    @model_validator(mode="after")
    def _same_artifacts(self) -> ReviewPacket:
        if set(self.artifact_payloads) != set(self.artifact_file_sha256):
            raise ValueError("review packet payload/hash keys differ")
        if set(self.allowed_evidence_artifact_ids) != set(self.artifact_payloads):
            raise ValueError("review evidence allowlist differs from packet artifacts")
        return self


def build_review_packet(run_dir: Path) -> ReviewPacket:
    """Build the fixed aggregate-only packet after hash-verifying iteration 1."""
    paths = {
        "objective": run_dir / "objective.json",
        "mutation_policy": run_dir / "mutation_policy.json",
        "baseline_evaluation": run_dir / "baseline/evaluation.json",
        "iteration_001_proposal": run_dir / "iteration-001/proposal.json",
        "iteration_001_tea_time": run_dir / "iteration-001/tea_time.json",
        "iteration_001_legality": run_dir / "iteration-001/legality.json",
        "iteration_001_execution": run_dir / "iteration-001/execution.json",
        "iteration_001_evaluation": run_dir / "iteration-001/evaluation.json",
        "iteration_001_decision": run_dir / "iteration-001/decision.json",
        "iteration_001_lesson": run_dir / "iteration-001/lesson.json",
        "environment": run_dir / "environment.json",
        "compute_attestation": run_dir / "compute_attestation.json",
        "dataset_verification": run_dir / "dataset_verification.json",
        "split_verification": run_dir / "split_verification.json",
        "checkpoint_verification": run_dir / "checkpoint_verification.json",
    }
    payloads: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for artifact_id, path in paths.items():
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"review packet artifact is not an object: {artifact_id}")
        payloads[artifact_id] = payload
        hashes[artifact_id] = sha256_file(path)
    return ReviewPacket(
        artifact_payloads=payloads,
        artifact_file_sha256=hashes,
        allowed_evidence_artifact_ids=sorted(payloads),
    ).sealed()


def verify_review_packet(packet: ReviewPacket, run_dir: Path) -> None:
    """Rebind a sealed packet to the exact iteration-one files on disk."""
    expected = build_review_packet(run_dir)
    if not packet.verify_seal() or packet != expected:
        raise ValueError("review packet does not match the current iteration-one artifacts")
