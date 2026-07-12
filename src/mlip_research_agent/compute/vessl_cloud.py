"""Approval-gated VESSL Cloud provider over the current ``vesslctl`` CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.compute.vessl_cli_transport import JsonValue, VesslCliTransport
from mlip_research_agent.compute.vessl_schemas import (
    VesslCostApproval,
    VesslCostCard,
    VesslJobRequest,
)
from mlip_research_agent.research.auto_research.validators import require_sealed


class VesslApprovalError(RuntimeError):
    pass


_COST_BINDING_FIELDS = (
    "organization",
    "team",
    "cluster",
    "resource_spec_slug",
    "gpu_type",
    "gpu_count",
    "current_hourly_price",
    "current_credit",
    "currency",
    "image",
    "expected_max_duration_minutes",
    "estimated_compute_cost",
    "storage_type",
    "storage_capacity_gb",
    "storage_hourly_rate",
    "storage_duration_hours",
    "estimated_storage_cost",
    "volume_mounts",
    "timeout_behavior",
    "cleanup_action",
)


def validate_cost_approval(
    *,
    approval: VesslCostApproval | None,
    approved_card: VesslCostCard,
    live_card: VesslCostCard,
    request: VesslJobRequest,
    now: datetime | None = None,
) -> None:
    if approval is None:
        raise VesslApprovalError("VESSL Job creation requires a sealed cost approval")
    require_sealed(approval, "VESSL cost approval")
    require_sealed(approved_card, "approved VESSL cost card")
    require_sealed(live_card, "live VESSL cost card")
    require_sealed(request, "VESSL Job request")
    if approval.cost_card_sha256 != approved_card.content_sha256:
        raise VesslApprovalError("cost approval does not bind the approved cost card")
    current = now or datetime.now(UTC)
    if current < approval.approved_at or current > approval.expires_at:
        raise VesslApprovalError("VESSL cost approval is stale or not yet valid")
    approved_payload = approved_card.model_dump(mode="json")
    live_payload = live_card.model_dump(mode="json")
    changed = [
        field
        for field in _COST_BINDING_FIELDS
        if approved_payload[field] != live_payload[field]
    ]
    if changed:
        raise VesslApprovalError(f"live VESSL cost card changed: {changed}")
    request_payload = request.model_dump(mode="json")
    for field in (
        "organization",
        "team",
        "cluster",
        "resource_spec_slug",
        "gpu_type",
        "gpu_count",
        "image",
        "expected_max_duration_minutes",
        "volume_mounts",
        "timeout_behavior",
        "cleanup_action",
    ):
        if request_payload[field] != approved_payload[field]:
            raise VesslApprovalError(f"VESSL Job request changed approved field {field}")


class VesslCloudProvider:
    """Current Cloud control plane; creation is impossible without approval."""

    def __init__(self, transport: VesslCliTransport) -> None:
        self.transport = transport

    def version(self) -> str:
        return self.transport.version()

    def auth_status(self) -> JsonValue:
        return self.transport.read_json(["auth", "status"])

    def config(self) -> JsonValue:
        return self.transport.read_json(["config", "show"])

    def billing(self) -> JsonValue:
        return self.transport.read_json(["billing", "show"])

    def organizations(self) -> JsonValue:
        return self.transport.read_json(["org", "list"])

    def teams(self) -> JsonValue:
        return self.transport.read_json(["team", "list"])

    def clusters(self) -> JsonValue:
        return self.transport.read_json(["cluster", "list"])

    def resource_specs(self) -> JsonValue:
        return self.transport.read_json(["resource-spec", "list", "--usable-only"])

    def storage(self) -> JsonValue:
        return self.transport.read_json(["storage", "list"])

    def volumes(self) -> JsonValue:
        return self.transport.read_json(["volume", "list"])

    def jobs(self) -> JsonValue:
        return self.transport.read_json(["job", "list"])

    def job(self, slug: str) -> JsonValue:
        return self.transport.read_json(["job", "show", slug])

    def job_logs(self, slug: str) -> JsonValue:
        return self.transport.read_json(["job", "logs", slug])

    def create_job(
        self,
        *,
        config_path: Path,
        request: VesslJobRequest,
        approval: VesslCostApproval | None,
        approved_card: VesslCostCard,
        live_card: VesslCostCard,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if not request.job_config_schema_verified:
            raise VesslApprovalError(
                "current vesslctl job-file schema has not been live-verified"
            )
        if sha256_file(config_path) != request.job_config_sha256:
            raise VesslApprovalError("generated VESSL Job config hash mismatch")
        validate_cost_approval(
            approval=approval,
            approved_card=approved_card,
            live_card=live_card,
            request=request,
            now=now,
        )
        payload = self.transport.create_job(config_path, approval_validated=True)
        if not isinstance(payload, dict):
            raise VesslApprovalError("VESSL Job creation did not return an object")
        return payload
