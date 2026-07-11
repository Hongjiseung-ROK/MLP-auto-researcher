"""Router tests: routing table, overrides/audit, preflight gate, redaction."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mlip_research_agent.compute.base import ComputePolicyDenied
from mlip_research_agent.compute.colab import (
    ColabProvider,
    MockRemoteTransport,
    mock_l4_observation,
)
from mlip_research_agent.compute.local import LocalProvider
from mlip_research_agent.compute.policy import load_policy
from mlip_research_agent.compute.preflight import PreflightRecord, PreflightStore
from mlip_research_agent.compute.router import ROUTING_TABLE, ComputeRouter, redact_secrets
from mlip_research_agent.compute.schemas import (
    AcceleratorId,
    JobSpec,
    JobState,
    ProviderName,
    WorkloadClass,
)
from mlip_research_agent.compute.vessl import VesslProvider, mock_a100_80gb_observation

REPO_ROOT = Path(__file__).parent.parent
POLICY_PATH = REPO_ROOT / "configs" / "compute_policy.yaml"


def _policy():
    return load_policy(POLICY_PATH)


def _job_spec(**overrides: object) -> JobSpec:
    defaults: dict[str, object] = {
        "name": "router-test-job",
        "workload_class": WorkloadClass.UNIT_TEST,
        "requested_accelerator": AcceleratorId.CPU,
        "command": ["python", "-c", "print('ok')"],
        "max_runtime_minutes": 10,
        "repo_commit": "deadbeef00",
    }
    defaults.update(overrides)
    return JobSpec(**defaults)


def _preflight_record(**overrides: object) -> PreflightRecord:
    defaults: dict[str, object] = {
        "run_id": "preflight-run-1",
        "passed": True,
        "repo_commit": "deadbeef00",
        "dependency_hash": "",
        "mlip_backend": "none",
        "cuda_compat_class": "none",
        "workflow_schema_version": "0.1.0",
        "created_at": datetime.now(UTC).isoformat(),
    }
    defaults.update(overrides)
    return PreflightRecord(**defaults)


def _read_audit_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- 1. Routing table ------------------------------------------------------


@pytest.mark.parametrize(
    "workload_class, expected_provider",
    [
        (WorkloadClass.UNIT_TEST, ProviderName.LOCAL),
        (WorkloadClass.GPU_SMOKE_TEST, ProviderName.COLAB),
        (WorkloadClass.PREFLIGHT, ProviderName.COLAB),
        (WorkloadClass.FAILURE_REPRODUCTION, ProviderName.COLAB),
        (WorkloadClass.DEVELOPMENT_TRAINING, ProviderName.COLAB),
        (WorkloadClass.PRIMARY_TRAINING, ProviderName.VESSL),
        (WorkloadClass.ACTIVE_LEARNING, ProviderName.VESSL),
        (WorkloadClass.BENCHMARK, ProviderName.VESSL),
        (WorkloadClass.PRODUCTION_RESEARCH, ProviderName.VESSL),
    ],
)
def test_routing_table_maps_workload_to_expected_provider(
    workload_class: WorkloadClass, expected_provider: ProviderName
) -> None:
    assert ROUTING_TABLE[workload_class] is expected_provider


# --- 9. Production routing --------------------------------------------------


def test_route_primary_training_with_no_override_routes_to_vessl(tmp_path: Path) -> None:
    router = ComputeRouter(_policy(), providers={}, preflight_store=PreflightStore(tmp_path / "pf"))
    spec = _job_spec(
        workload_class=WorkloadClass.PRIMARY_TRAINING,
        requested_accelerator=AcceleratorId.NVIDIA_A100_80GB,
    )
    assert router.route(spec) is ProviderName.VESSL


# --- 10. Bounded Colab workload (router side) -------------------------------


def test_route_colab_override_denied_when_development_training_bound_exceeded(
    tmp_path: Path,
) -> None:
    router = ComputeRouter(_policy(), providers={}, preflight_store=PreflightStore(tmp_path / "pf"))
    spec = _job_spec(
        workload_class=WorkloadClass.DEVELOPMENT_TRAINING,
        requested_accelerator=AcceleratorId.NVIDIA_L4,
        provider_override=ProviderName.COLAB,
        max_runtime_minutes=600,
    )
    with pytest.raises(ComputePolicyDenied):
        router.route(spec)


# --- 11. Colab full-campaign rejection --------------------------------------


def test_route_denies_full_campaign_override_to_colab_and_audits_denial(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    router = ComputeRouter(
        _policy(),
        providers={},
        preflight_store=PreflightStore(tmp_path / "pf"),
        audit_log_path=audit_path,
    )
    spec = _job_spec(
        name="full-campaign-job",
        workload_class=WorkloadClass.PRIMARY_TRAINING,
        requested_accelerator=AcceleratorId.NVIDIA_A100_80GB,
        provider_override=ProviderName.COLAB,
    )
    with pytest.raises(ComputePolicyDenied):
        router.route(spec)
    events = _read_audit_events(audit_path)
    override_events = [e for e in events if e["event"] == "provider_override"]
    assert len(override_events) == 1
    assert override_events[0]["allowed"] is False


# --- 12. Provider override audit logging (allowed case) --------------------


def test_route_allowed_override_is_audited_as_allowed(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    router = ComputeRouter(
        _policy(),
        providers={},
        preflight_store=PreflightStore(tmp_path / "pf"),
        audit_log_path=audit_path,
    )
    spec = _job_spec(
        name="smoke-test-job",
        workload_class=WorkloadClass.GPU_SMOKE_TEST,
        requested_accelerator=AcceleratorId.NVIDIA_L4,
        provider_override=ProviderName.COLAB,
    )
    result = router.route(spec)
    assert result is ProviderName.COLAB
    events = _read_audit_events(audit_path)
    override_events = [e for e in events if e["event"] == "provider_override"]
    assert len(override_events) == 1
    assert override_events[0]["allowed"] is True


# --- 13. Secret redaction (function-level; provider-level lives in
#         test_compute_providers.py) ----------------------------------------


@pytest.mark.parametrize(
    "raw, must_be_absent",
    [
        ("token=abc123", "abc123"),
        ("API_KEY: xyz", "xyz"),
        ("gho_ABCDEFGHIJKLMNOP1234", "gho_ABCDEFGHIJKLMNOP1234"),  # PLACEHOLDER, not real
        ("sk-abcdefghijklmnop1234", "sk-abcdefghijklmnop1234"),  # PLACEHOLDER, not real
    ],
)
def test_redact_secrets_scrubs_known_patterns(raw: str, must_be_absent: str) -> None:
    result = redact_secrets(raw)
    assert "[REDACTED]" in result
    assert must_be_absent not in result


def test_redact_secrets_leaves_ordinary_text_untouched() -> None:
    text = "this is an ordinary log line with nothing sensitive in it"
    assert redact_secrets(text) == text


# --- 8. Preflight staleness -------------------------------------------------


def _vessl_router(
    tmp_path: Path, preflight_store: PreflightStore, *, matching_gpu: bool = True
) -> ComputeRouter:
    policy = _policy()
    observation = mock_a100_80gb_observation() if matching_gpu else None
    vessl_provider = VesslProvider(
        policy, tmp_path / "artifacts", transport=MockRemoteTransport(assigned_gpu=observation)
    )
    return ComputeRouter(
        policy, providers={ProviderName.VESSL: vessl_provider}, preflight_store=preflight_store
    )


def _primary_training_spec() -> JobSpec:
    return _job_spec(
        name="primary-training-job",
        workload_class=WorkloadClass.PRIMARY_TRAINING,
        requested_accelerator=AcceleratorId.NVIDIA_A100_80GB,
        repo_commit="deadbeef00",
        dependency_hash="",
    )


def test_preflight_stale_record_blocks_vessl_submission(tmp_path: Path) -> None:
    store = PreflightStore(tmp_path / "pf")
    stale_created_at = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
    store.save(_preflight_record(created_at=stale_created_at))
    router = _vessl_router(tmp_path, store)
    with pytest.raises(ComputePolicyDenied):
        router.submit(_primary_training_spec())


def test_preflight_fresh_matching_record_allows_vessl_submission(tmp_path: Path) -> None:
    store = PreflightStore(tmp_path / "pf")
    fresh_created_at = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    store.save(_preflight_record(created_at=fresh_created_at))
    router = _vessl_router(tmp_path, store)
    handle = router.submit(_primary_training_spec())
    provider = router.providers[ProviderName.VESSL]
    assert provider.status(handle).state is JobState.SUCCEEDED


@pytest.mark.parametrize(
    "mismatch_kwargs",
    [
        {"repo_commit": "0123456789abcdef"},
        {"dependency_hash": "some-other-dep-hash"},
    ],
)
def test_preflight_mismatched_record_blocks_vessl_submission(
    tmp_path: Path, mismatch_kwargs: dict[str, str]
) -> None:
    store = PreflightStore(tmp_path / "pf")
    store.save(_preflight_record(**mismatch_kwargs))
    router = _vessl_router(tmp_path, store)
    with pytest.raises(ComputePolicyDenied):
        router.submit(_primary_training_spec())


# --- 15. VESSL submission blocked after FAILED preflight -------------------


def test_vessl_submission_blocked_after_failed_preflight(tmp_path: Path) -> None:
    store = PreflightStore(tmp_path / "pf")
    store.save(_preflight_record(passed=False))
    router = _vessl_router(tmp_path, store)
    with pytest.raises(ComputePolicyDenied):
        router.submit(_primary_training_spec())


# --- 16. Mock end-to-end -----------------------------------------------------


def test_end_to_end_mock_pipeline(tmp_path: Path) -> None:
    policy = _policy()
    artifacts_root = tmp_path / "artifacts"
    preflight_store = PreflightStore(tmp_path / "pf")
    colab_transport = MockRemoteTransport(assigned_gpu=mock_l4_observation())
    vessl_transport = MockRemoteTransport(assigned_gpu=mock_a100_80gb_observation())
    providers = {
        ProviderName.LOCAL: LocalProvider(policy, artifacts_root),
        ProviderName.COLAB: ColabProvider(policy, artifacts_root, transport=colab_transport),
        ProviderName.VESSL: VesslProvider(
            policy,
            artifacts_root,
            transport=vessl_transport,
        ),
    }
    router = ComputeRouter(policy, providers, preflight_store)

    preflight_spec = _job_spec(
        name="preflight-job",
        workload_class=WorkloadClass.PREFLIGHT,
        requested_accelerator=AcceleratorId.NVIDIA_L4,
        command=["true"],
    )
    preflight_handle = router.submit(preflight_spec)
    assert providers[ProviderName.COLAB].status(preflight_handle).state is JobState.SUCCEEDED

    preflight_store.save(
        _preflight_record(
            run_id=preflight_handle.run_id,
            repo_commit=preflight_spec.repo_commit,
            dependency_hash=preflight_spec.dependency_hash,
            mlip_backend=preflight_spec.mlip_backend,
            cuda_compat_class=preflight_spec.cuda_compat_class,
            workflow_schema_version=preflight_spec.workflow_schema_version,
        )
    )

    primary_spec = _job_spec(
        name="primary-job",
        workload_class=WorkloadClass.PRIMARY_TRAINING,
        requested_accelerator=AcceleratorId.NVIDIA_A100_80GB,
        command=["true"],
        repo_commit=preflight_spec.repo_commit,
        dependency_hash=preflight_spec.dependency_hash,
        mlip_backend=preflight_spec.mlip_backend,
        cuda_compat_class=preflight_spec.cuda_compat_class,
        workflow_schema_version=preflight_spec.workflow_schema_version,
    )
    primary_handle = router.submit(primary_spec)
    assert providers[ProviderName.VESSL].status(primary_handle).state is JobState.SUCCEEDED

    local_spec = _job_spec(
        name="unit-job",
        workload_class=WorkloadClass.UNIT_TEST,
        requested_accelerator=AcceleratorId.CPU,
        command=["python", "-c", "print('ok')"],
    )
    local_handle = router.submit(local_spec)
    assert providers[ProviderName.LOCAL].status(local_handle).state is JobState.SUCCEEDED
