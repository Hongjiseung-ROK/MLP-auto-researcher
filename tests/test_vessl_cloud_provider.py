from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.compute.vessl_cli_transport import (
    VesslCliError,
    VesslCliNotInstalled,
    VesslCliTransport,
)
from mlip_research_agent.compute.vessl_cloud import (
    VesslApprovalError,
    VesslCloudProvider,
    validate_cost_approval,
)
from mlip_research_agent.compute.vessl_schemas import (
    VesslBillingSnapshot,
    VesslCostApproval,
    VesslCostCard,
    VesslJobRequest,
    VesslVolumeMount,
)


class FakeRunner:
    def __init__(self, stdout: bytes = b"{}", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            self.returncode,
            stdout=self.stdout,
            stderr=b"" if self.returncode == 0 else b"access_token=top-secret-value",
        )


NOW = datetime(2026, 7, 12, 2, 0, tzinfo=UTC)


def _mount() -> VesslVolumeMount:
    return VesslVolumeMount(
        volume_type="object",
        volume_slug="mlip-runs",
        mount_path="/runs",
    ).sealed()


def _card(**updates: Any) -> VesslCostCard:
    payload: dict[str, Any] = {
        "organization": "org",
        "team": "team",
        "cluster": "cluster-a",
        "resource_spec_slug": "a100-one",
        "gpu_type": "NVIDIA A100 80GB",
        "gpu_count": 1,
        "current_hourly_price": 2.0,
        "current_credit": 20.0,
        "currency": "USD",
        "image": "registry.example/mlip@sha256:" + "a" * 64,
        "expected_max_duration_minutes": 30,
        "estimated_compute_cost": 1.0,
        "storage_type": "object",
        "storage_capacity_gb": 10.0,
        "storage_hourly_rate": 0.1,
        "storage_duration_hours": 1.0,
        "estimated_storage_cost": 0.1,
        "volume_mounts": [_mount()],
        "timeout_behavior": "Job exits nonzero at the hard wall-clock limit.",
        "cleanup_action": "Verify terminal Job state and report surviving storage.",
        "live_snapshot_sha256": "b" * 64,
        "observed_at": NOW,
    }
    payload.update(updates)
    if "current_hourly_price" in updates and "estimated_compute_cost" not in updates:
        payload["estimated_compute_cost"] = float(payload["current_hourly_price"]) * 0.5
    return VesslCostCard.model_validate(payload).sealed()


def _request(config_sha256: str, **updates: Any) -> VesslJobRequest:
    payload: dict[str, Any] = {
        "phase": "iteration_1",
        "organization": "org",
        "team": "team",
        "cluster": "cluster-a",
        "resource_spec_slug": "a100-one",
        "gpu_type": "NVIDIA A100 80GB",
        "gpu_count": 1,
        "image": "registry.example/mlip@sha256:" + "a" * 64,
        "expected_max_duration_minutes": 30,
        "volume_mounts": [_mount()],
        "timeout_behavior": "Job exits nonzero at the hard wall-clock limit.",
        "cleanup_action": "Verify terminal Job state and report surviving storage.",
        "git_commit": "c" * 40,
        "job_config_sha256": config_sha256,
        "job_config_schema_verified": True,
        "bounded_label_view_sha256": "d" * 64,
        "dataset_sha256": "e" * 64,
        "split_sha256": "f" * 64,
        "checkpoint_sha256": "1" * 64,
        "environment_lock_sha256": "2" * 64,
    }
    payload.update(updates)
    return VesslJobRequest.model_validate(payload).sealed()


def _approval(card: VesslCostCard, **updates: Any) -> VesslCostApproval:
    payload: dict[str, Any] = {
        "cost_card_sha256": card.content_sha256,
        "approval_artifact_sha256": "3" * 64,
        "approved_by": "owner",
        "approved_at": NOW,
        "expires_at": NOW + timedelta(minutes=15),
    }
    payload.update(updates)
    return VesslCostApproval.model_validate(payload).sealed()


def test_cli_not_installed_fails_without_running() -> None:
    transport = VesslCliTransport(executable=None)
    transport.executable = None
    with pytest.raises(VesslCliNotInstalled):
        transport.read_json(["auth", "status"])


def test_read_only_allowlist_uses_argument_array_and_shell_false() -> None:
    runner = FakeRunner(b'{"authenticated": true}')
    transport = VesslCliTransport(executable="/usr/bin/vesslctl", runner=runner)
    assert transport.read_json(["auth", "status"]) == {"authenticated": True}
    command, kwargs = runner.calls[0]
    assert command == ["/usr/bin/vesslctl", "auth", "status", "-o", "json"]
    assert kwargs["shell"] is False
    assert kwargs["capture_output"] is True
    with pytest.raises(VesslCliError, match="not read-only allowlisted"):
        transport.read_json(["workspace", "create"])
    with pytest.raises(VesslCliError, match="not read-only allowlisted"):
        transport.read_json(["volume", "token", "secret-volume"])


def test_malformed_json_and_secret_keys_fail_safe() -> None:
    malformed = VesslCliTransport(
        executable="vesslctl", runner=FakeRunner(b"not-json")
    )
    with pytest.raises(VesslCliError, match="malformed JSON"):
        malformed.read_json(["billing", "show"])
    secret = VesslCliTransport(
        executable="vesslctl",
        runner=FakeRunner(b'{"access_token":"do-not-record","nested":{"password":"x"}}'),
    )
    payload = secret.read_json(["auth", "status"])
    assert payload == {
        "access_token": "[REDACTED]",
        "nested": {"password": "[REDACTED]"},
    }


def test_billing_snapshot_rejects_missing_credit_fields() -> None:
    with pytest.raises(ValidationError):
        VesslBillingSnapshot.model_validate(
            {
                "organization": "org",
                "currency": "USD",
                "observed_at": NOW,
                "raw_payload_sha256": "a" * 64,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_hourly_price", 2.5),
        ("resource_spec_slug", "different-a100"),
        ("image", "different-image"),
        ("volume_mounts", []),
    ],
)
def test_changed_live_cost_binding_invalidates_approval(
    tmp_path: Path, field: str, value: Any
) -> None:
    config = tmp_path / "job.json"
    config.write_text("{}\n")
    card = _card()
    request = _request(sha256_file(config))
    with pytest.raises(VesslApprovalError, match="changed"):
        validate_cost_approval(
            approval=_approval(card),
            approved_card=card,
            live_card=_card(**{field: value}),
            request=request,
            now=NOW + timedelta(minutes=1),
        )


def test_missing_and_stale_approval_block_job_creation(tmp_path: Path) -> None:
    config = tmp_path / "job.json"
    config.write_text(json.dumps({"name": "dry-run"}) + "\n")
    card = _card()
    request = _request(sha256_file(config))
    with pytest.raises(VesslApprovalError, match="requires a sealed cost approval"):
        validate_cost_approval(
            approval=None,
            approved_card=card,
            live_card=card,
            request=request,
            now=NOW,
        )
    with pytest.raises(VesslApprovalError, match="stale"):
        validate_cost_approval(
            approval=_approval(card),
            approved_card=card,
            live_card=card,
            request=request,
            now=NOW + timedelta(hours=1),
        )


def test_non_a100_and_multi_gpu_are_denied() -> None:
    with pytest.raises(ValidationError, match="exactly one live A100"):
        _card(gpu_type="NVIDIA H100 80GB")
    with pytest.raises(ValidationError, match="exactly one live A100"):
        _card(gpu_count=2)
    with pytest.raises(ValidationError):
        _request("a" * 64, gpu_count=2)


def test_provider_job_create_binds_config_and_approval(tmp_path: Path) -> None:
    config = tmp_path / "job.json"
    config.write_text(json.dumps({"name": "dry-run"}) + "\n")
    card = _card()
    request = _request(sha256_file(config))
    runner = FakeRunner(b'{"slug":"job-123","access_token":"not-recorded"}')
    provider = VesslCloudProvider(
        VesslCliTransport(executable="vesslctl", runner=runner)
    )
    result = provider.create_job(
        config_path=config,
        request=request,
        approval=_approval(card),
        approved_card=card,
        live_card=card,
        now=NOW + timedelta(minutes=1),
    )
    assert result == {"slug": "job-123", "access_token": "[REDACTED]"}
    assert runner.calls[0][0] == [
        "vesslctl",
        "job",
        "create",
        "--file",
        str(config),
        "-o",
        "json",
    ]


def test_transport_refuses_ungated_job_create(tmp_path: Path) -> None:
    config = tmp_path / "job.json"
    config.write_text("{}\n")
    transport = VesslCliTransport(executable="vesslctl", runner=FakeRunner())
    with pytest.raises(VesslCliError, match="validated sealed cost approval"):
        transport.create_job(config, approval_validated=False)


def test_provider_refuses_unverified_live_job_file_schema(tmp_path: Path) -> None:
    config = tmp_path / "job.json"
    config.write_text("{}\n")
    card = _card()
    request = _request(sha256_file(config), job_config_schema_verified=False)
    provider = VesslCloudProvider(
        VesslCliTransport(executable="vesslctl", runner=FakeRunner())
    )
    with pytest.raises(VesslApprovalError, match="file schema has not been live-verified"):
        provider.create_job(
            config_path=config,
            request=request,
            approval=_approval(card),
            approved_card=card,
            live_card=card,
            now=NOW,
        )
