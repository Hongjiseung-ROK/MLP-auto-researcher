"""Provider tests: authorization, denial mechanics, redaction, interruption."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlip_research_agent.compute.colab import (
    ColabProvider,
    MockRemoteTransport,
    mock_l4_observation,
)
from mlip_research_agent.compute.policy import load_policy
from mlip_research_agent.compute.schemas import (
    AcceleratorId,
    GpuObservation,
    JobSpec,
    JobState,
    WorkloadClass,
)
from mlip_research_agent.compute.vessl import mock_a100_80gb_observation

REPO_ROOT = Path(__file__).parent.parent
POLICY_PATH = REPO_ROOT / "configs" / "compute_policy.yaml"


def _policy():
    return load_policy(POLICY_PATH)


def _job_spec(**overrides: object) -> JobSpec:
    defaults: dict[str, object] = {
        "name": "provider-test-job",
        "workload_class": WorkloadClass.GPU_SMOKE_TEST,
        "requested_accelerator": AcceleratorId.NVIDIA_L4,
        "command": ["true"],
        "max_runtime_minutes": 30,
        "repo_commit": "deadbeef00",
    }
    defaults.update(overrides)
    return JobSpec(**defaults)


def _attestation_record(artifacts_root: Path, run_id: str) -> dict[str, object]:
    path = artifacts_root / "compute" / run_id / "attestation.json"
    assert path.exists(), f"expected attestation.json at {path}"
    result: dict[str, object] = json.loads(path.read_text())
    return result


# --- 3. Authorization via ColabProvider + MockRemoteTransport ---------------


def test_colab_authorizes_matching_l4(tmp_path: Path) -> None:
    provider = ColabProvider(
        _policy(), tmp_path, transport=MockRemoteTransport(assigned_gpu=mock_l4_observation())
    )
    handle = provider.submit(_job_spec(requested_accelerator=AcceleratorId.NVIDIA_L4))
    assert provider.status(handle).state is JobState.SUCCEEDED


def test_colab_authorizes_matching_a100_40gb(tmp_path: Path) -> None:
    observation = GpuObservation(name="NVIDIA A100-SXM4-40GB")
    transport = MockRemoteTransport(assigned_gpu=observation)
    provider = ColabProvider(_policy(), tmp_path, transport=transport)
    handle = provider.submit(_job_spec(requested_accelerator=AcceleratorId.NVIDIA_A100_40GB))
    assert provider.status(handle).state is JobState.SUCCEEDED


def test_colab_authorizes_matching_a100_80gb(tmp_path: Path) -> None:
    transport = MockRemoteTransport(assigned_gpu=mock_a100_80gb_observation())
    provider = ColabProvider(_policy(), tmp_path, transport=transport)
    handle = provider.submit(_job_spec(requested_accelerator=AcceleratorId.NVIDIA_A100_80GB))
    assert provider.status(handle).state is JobState.SUCCEEDED


# --- 4/5. Unauthorized / unknown accelerators denied ------------------------


@pytest.mark.parametrize("observed_name", ["Tesla T4", "NVIDIA H100 80GB HBM3", "garbage"])
def test_colab_denies_unauthorized_or_unknown_gpu_and_writes_deny_attestation(
    tmp_path: Path, observed_name: str
) -> None:
    provider = ColabProvider(
        _policy(),
        tmp_path,
        transport=MockRemoteTransport(assigned_gpu=GpuObservation(name=observed_name)),
    )
    spec = _job_spec(requested_accelerator=AcceleratorId.NVIDIA_L4)
    handle = provider.submit(spec)
    assert provider.status(handle).state is JobState.POLICY_DENIED
    record = _attestation_record(tmp_path, handle.run_id)
    assert record["policy_decision"] == "deny"


# --- 6. Missing attestation --------------------------------------------------


def test_colab_denies_missing_attestation(tmp_path: Path) -> None:
    provider = ColabProvider(_policy(), tmp_path, transport=MockRemoteTransport(assigned_gpu=None))
    spec = _job_spec(requested_accelerator=AcceleratorId.NVIDIA_L4)
    handle = provider.submit(spec)
    status = provider.status(handle)
    assert status.state is JobState.POLICY_DENIED
    assert "attestation" in status.message.lower()
    record = _attestation_record(tmp_path, handle.run_id)
    assert record["policy_decision"] == "deny"


# --- 7. Requested-vs-observed mismatch --------------------------------------


def test_colab_denies_requested_observed_mismatch(tmp_path: Path) -> None:
    provider = ColabProvider(
        _policy(), tmp_path, transport=MockRemoteTransport(assigned_gpu=mock_l4_observation())
    )
    spec = _job_spec(requested_accelerator=AcceleratorId.NVIDIA_A100_80GB)
    handle = provider.submit(spec)
    assert provider.status(handle).state is JobState.POLICY_DENIED
    record = _attestation_record(tmp_path, handle.run_id)
    assert record["policy_decision"] == "deny"


# --- 13. Provider log redaction ---------------------------------------------


def test_colab_provider_redacts_secrets_in_stream_logs(tmp_path: Path) -> None:
    # NB: redact_secrets matches "token"/"secret"/etc. as a whole word bounded
    # by non-word characters; an underscore (as in HF_TOKEN) does not create a
    # boundary, so a hyphenated variable name is used here to exercise the
    # actual provider-log redaction path end to end.
    provider = ColabProvider(
        _policy(),
        tmp_path,
        transport=MockRemoteTransport(
            assigned_gpu=mock_l4_observation(),
            log_lines=["export HF-TOKEN=supersecret123"],
        ),
    )
    spec = _job_spec(requested_accelerator=AcceleratorId.NVIDIA_L4)
    handle = provider.submit(spec)
    combined = "\n".join(event.message for event in provider.stream_logs(handle))
    assert "[REDACTED]" in combined
    assert "supersecret123" not in combined


# --- 14. Interrupted Colab session recovery ---------------------------------


def test_colab_interrupted_session_recovers_via_resubmission(tmp_path: Path) -> None:
    interrupted_transport = MockRemoteTransport(
        assigned_gpu=mock_l4_observation(),
        interrupt=True,
        produced_files={"partial.json": "{}"},
    )
    provider = ColabProvider(_policy(), tmp_path, transport=interrupted_transport)
    spec = _job_spec(requested_accelerator=AcceleratorId.NVIDIA_L4)
    handle = provider.submit(spec)

    assert provider.status(handle).state is JobState.INTERRUPTED
    manifest = provider.collect_artifacts(handle)
    assert any(artifact.path == "partial.json" for artifact in manifest.artifacts)

    # Recovery by resubmission: a fresh provider/transport for the same spec
    # completes cleanly, independent of the prior interrupted session.
    fresh_provider = ColabProvider(
        _policy(), tmp_path, transport=MockRemoteTransport(assigned_gpu=mock_l4_observation())
    )
    new_handle = fresh_provider.submit(spec)
    assert fresh_provider.status(new_handle).state is JobState.SUCCEEDED
