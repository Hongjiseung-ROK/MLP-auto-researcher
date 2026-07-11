"""Policy tests: GPU name normalization and fail-closed evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlip_research_agent.compute.policy import load_policy, normalize_gpu_name
from mlip_research_agent.compute.schemas import (
    AcceleratorId,
    JobSpec,
    ProviderName,
    WorkloadClass,
)

REPO_ROOT = Path(__file__).parent.parent
POLICY_PATH = REPO_ROOT / "configs" / "compute_policy.yaml"


def _policy():
    return load_policy(POLICY_PATH)


def _job_spec(**overrides: object) -> JobSpec:
    defaults: dict[str, object] = {
        "name": "policy-test-job",
        "workload_class": WorkloadClass.UNIT_TEST,
        "requested_accelerator": AcceleratorId.CPU,
        "command": ["python", "-c", "print('ok')"],
        "max_runtime_minutes": 10,
        "repo_commit": "deadbeef00",
    }
    defaults.update(overrides)
    return JobSpec(**defaults)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("NVIDIA L4", AcceleratorId.NVIDIA_L4),
        ("NVIDIA A100-SXM4-40GB", AcceleratorId.NVIDIA_A100_40GB),
        ("NVIDIA A100-SXM4-80GB", AcceleratorId.NVIDIA_A100_80GB),
        ("NVIDIA A100 80GB PCIe", AcceleratorId.NVIDIA_A100_80GB),
        ("  nvidia   l4 ", AcceleratorId.NVIDIA_L4),
    ],
)
def test_normalize_gpu_name_known_aliases(raw: str, expected: AcceleratorId) -> None:
    assert normalize_gpu_name(raw) is expected


@pytest.mark.parametrize("raw", ["Tesla T4", "NVIDIA H100 80GB HBM3", "garbage"])
def test_normalize_gpu_name_unknown_returns_none(raw: str) -> None:
    assert normalize_gpu_name(raw) is None


def test_load_policy_reads_expected_structure() -> None:
    policy = _policy()
    assert policy.version == 1
    assert policy.routing.primary_provider is ProviderName.VESSL
    assert policy.routing.validation_provider is ProviderName.COLAB
    assert policy.routing.local_provider is ProviderName.LOCAL
    assert set(policy.providers) == {ProviderName.LOCAL, ProviderName.COLAB, ProviderName.VESSL}


def test_evaluate_bounded_colab_development_training_within_bound_allows() -> None:
    # observed_gpu_name=None always carries a "requires runtime attestation"
    # reason for colab (require_runtime_attestation=true); that reason alone
    # is not about the runtime bound. Every other (binding) reason -- in
    # particular any "must be bounded" denial -- must be absent when the
    # requested runtime is within the policy bound.
    policy = _policy()
    spec = _job_spec(
        workload_class=WorkloadClass.DEVELOPMENT_TRAINING,
        requested_accelerator=AcceleratorId.NVIDIA_L4,
        max_runtime_minutes=60,
    )
    decision = policy.evaluate(ProviderName.COLAB, spec, observed_gpu_name=None)
    non_attestation_reasons = [r for r in decision.reasons if "runtime attestation" not in r]
    assert not non_attestation_reasons


def test_evaluate_bounded_colab_development_training_exceeding_bound_denies() -> None:
    policy = _policy()
    spec = _job_spec(
        workload_class=WorkloadClass.DEVELOPMENT_TRAINING,
        requested_accelerator=AcceleratorId.NVIDIA_L4,
        max_runtime_minutes=600,
    )
    decision = policy.evaluate(ProviderName.COLAB, spec, observed_gpu_name=None)
    assert not decision.allowed
    assert any("bounded" in reason for reason in decision.reasons)
