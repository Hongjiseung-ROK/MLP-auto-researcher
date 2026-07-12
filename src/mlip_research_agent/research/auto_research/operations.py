"""Append-only exactly-once receipts for non-idempotent adapter operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from mlip_research_agent.research.auto_research.validators import ContentAddressedModel


class OptimizerOperationRequest(ContentAddressedModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    git_commit: str = Field(min_length=40, max_length=40)
    proposal_fingerprint: str = Field(min_length=64, max_length=64)
    config_fingerprint: str = Field(min_length=64, max_length=64)
    seed: int = Field(ge=0)
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_manifest_sha256: str = Field(min_length=64, max_length=64)
    checkpoint_sha256: str = Field(min_length=64, max_length=64)
    e0_policy: Literal["foundation"] = "foundation"
    max_optimizer_steps: Literal[1] = 1


class OptimizerOperationReceipt(ContentAddressedModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str
    request_sha256: str = Field(min_length=64, max_length=64)
    status: Literal["started", "complete"]
    optimizer_steps: int = Field(ge=0, le=1)
    model_sha256: str | None = None
    checkpoint_sha256: str | None = None
    scientific_status: Literal["infrastructure_only"] = "infrastructure_only"
    claim_eligible: Literal[False] = False


class ExactlyOnceOperation:
    """Fail closed on an ambiguous start; reuse only a completed receipt."""

    def __init__(self, directory: Path, request: OptimizerOperationRequest) -> None:
        self.directory = directory
        self.request = request
        self.request_path = directory / "operation_request.json"
        self.started_path = directory / "operation_started.json"
        self.completed_path = directory / "operation_completed.json"

    def begin(self) -> OptimizerOperationReceipt | None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.completed_path.is_file():
            completed = OptimizerOperationReceipt.model_validate_json(
                self.completed_path.read_text()
            )
            if (
                not completed.verify_seal()
                or completed.request_sha256 != self.request.content_sha256
                or completed.operation_id != self.request.operation_id
            ):
                raise RuntimeError("completed optimizer receipt conflicts with request")
            return completed
        if self.started_path.is_file():
            started = OptimizerOperationReceipt.model_validate_json(self.started_path.read_text())
            if (
                not started.verify_seal()
                or started.request_sha256 != self.request.content_sha256
                or started.operation_id != self.request.operation_id
            ):
                raise RuntimeError("started optimizer receipt conflicts with request")
            raise RuntimeError(
                "ambiguous optimizer operation already started; reconnect or "
                "escalate, never relaunch"
            )
        if self.request_path.exists():
            recorded = OptimizerOperationRequest.model_validate_json(self.request_path.read_text())
            if recorded != self.request:
                raise RuntimeError("optimizer request path contains different immutable content")
        else:
            self.request_path.write_text(self.request.canonical_text())
        started = OptimizerOperationReceipt(
            operation_id=self.request.operation_id,
            request_sha256=self.request.content_sha256,
            status="started",
            optimizer_steps=0,
        ).sealed()
        self.started_path.write_text(started.canonical_text())
        return None

    def complete(self, *, model_sha256: str, checkpoint_sha256: str) -> OptimizerOperationReceipt:
        if self.completed_path.exists():
            raise RuntimeError("optimizer completion receipt already exists")
        if not self.started_path.is_file():
            raise RuntimeError("optimizer completion requires a started receipt")
        started = OptimizerOperationReceipt.model_validate_json(self.started_path.read_text())
        if (
            not started.verify_seal()
            or started.operation_id != self.request.operation_id
            or started.request_sha256 != self.request.content_sha256
        ):
            raise RuntimeError("optimizer started receipt conflicts with completion request")
        receipt = OptimizerOperationReceipt(
            operation_id=self.request.operation_id,
            request_sha256=self.request.content_sha256,
            status="complete",
            optimizer_steps=1,
            model_sha256=model_sha256,
            checkpoint_sha256=checkpoint_sha256,
        ).sealed()
        self.completed_path.write_text(receipt.canonical_text())
        return receipt


def operation_receipts(directory: Path) -> list[dict[str, object]]:
    return [json.loads(path.read_text()) for path in sorted(directory.rglob("operation_*.json"))]
