"""Correctness tests for the colab_preflight skill (mock transport only)."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.compute.schemas import JobState, WorkloadClass
from mlip_research_agent.schemas.failure import FailureClass
from mlip_research_agent.skills.base import SkillContext, SkillError
from mlip_research_agent.skills.compute.colab.implementation import ColabPreflightSkill
from mlip_research_agent.skills.compute.colab.schema import (
    ColabPreflightInput,
    ColabPreflightOutput,
)

REPO_ROOT = Path(__file__).parents[6]
POLICY = str(REPO_ROOT / "configs" / "compute_policy.yaml")


def make_ctx(tmp_path: Path) -> SkillContext:
    return SkillContext(
        run_dir=tmp_path,
        step_id="preflight",
        seed=7,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )


def run_skill(tmp_path: Path, **overrides: object) -> BaseModel:
    inputs = ColabPreflightInput.model_validate({"policy_path": POLICY, **overrides})
    return ColabPreflightSkill().run(inputs, make_ctx(tmp_path))


def test_mock_l4_preflight_allows_and_saves_record(tmp_path: Path) -> None:
    out = run_skill(tmp_path)
    assert isinstance(out, ColabPreflightOutput)
    assert out.job_state is JobState.SUCCEEDED
    assert out.canonical_accelerator is not None
    assert out.canonical_accelerator.value == "nvidia-l4"
    assert out.preflight_record_saved
    assert out.n_remote_artifacts >= 1


def test_unauthorized_device_denied_with_evidence(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    inputs = ColabPreflightInput.model_validate(
        {"policy_path": POLICY, "mock_observed_gpu": "Tesla T4"}
    )
    with pytest.raises(SkillError) as excinfo:
        ColabPreflightSkill().run(inputs, ctx)
    assert excinfo.value.failure_class is FailureClass.TOOL_ERROR
    assert excinfo.value.retryable
    # Evidence was registered before the failure raised.
    kinds = {a.kind for a in ctx.registry.all()}
    assert {"attestation", "preflight_result"} <= kinds
    attestation_artifact = next(a for a in ctx.registry.all() if a.kind == "attestation")
    attestation = json.loads((tmp_path / attestation_artifact.relative_path).read_text())
    assert attestation["policy_decision"] == "deny"
    # No preflight record was saved for the VESSL gate.
    result = next(a for a in ctx.registry.all() if a.kind == "preflight_result")
    payload = json.loads((tmp_path / result.relative_path).read_text())
    assert payload["preflight_record_saved"] is False


def test_cpu_accelerator_rejected(tmp_path: Path) -> None:
    with pytest.raises(SkillError) as excinfo:
        run_skill(tmp_path, requested_accelerator="cpu")
    assert excinfo.value.failure_class is FailureClass.VALIDATION_ERROR


def test_production_workload_rejected(tmp_path: Path) -> None:
    with pytest.raises(SkillError) as excinfo:
        run_skill(tmp_path, workload_class=WorkloadClass.PRIMARY_TRAINING.value)
    assert excinfo.value.failure_class is FailureClass.VALIDATION_ERROR
    assert not excinfo.value.retryable


def test_real_mode_requires_reviewed_cli(tmp_path: Path) -> None:
    with pytest.raises(SkillError) as excinfo:
        run_skill(tmp_path, use_mock=False)
    assert excinfo.value.failure_class is FailureClass.TOOL_ERROR
    assert "gcloud ADC" in str(excinfo.value)
