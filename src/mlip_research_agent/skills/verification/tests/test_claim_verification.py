"""Correctness tests for the claim_verification skill."""

import json
from pathlib import Path

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.schemas.claims import (
    Claim,
    ClaimClass,
    ClaimStatus,
    ScientificEvidenceTier,
)
from mlip_research_agent.schemas.failure import FailureClass
from mlip_research_agent.skills.base import SkillContext, SkillError
from mlip_research_agent.skills.verification.implementation import ClaimVerificationSkill
from mlip_research_agent.skills.verification.schema import VerificationInput, VerificationOutput


def setup_run(tmp_path: Path, artifact_refs: list[str], register: bool = True) -> ArtifactRegistry:
    registry = ArtifactRegistry(tmp_path)
    if register:
        step_dir = tmp_path / "steps" / "evaluation"
        step_dir.mkdir(parents=True)
        metrics = step_dir / "metrics.json"
        metrics.write_text(json.dumps({"energy_mae": 0.5}))
        registry.register(metrics, kind="metrics", step_id="evaluation")
    claim = Claim(
        claim_id="energy_mae_holdout",
        statement="MAE claim for testing",
        value=0.5,
        units="eV",
        created_by_step="evaluation",
        artifact_references=artifact_refs,
    )
    (tmp_path / "claims.json").write_text(json.dumps([claim.model_dump(mode="json")]))
    return registry


def verify(tmp_path: Path, registry: ArtifactRegistry, strict: bool = True) -> VerificationOutput:
    ctx = SkillContext(
        run_dir=tmp_path, step_id="verification", seed=0, attempt=0, registry=registry
    )
    out = ClaimVerificationSkill().run(VerificationInput(strict=strict), ctx)
    assert isinstance(out, VerificationOutput)
    return out


def test_backed_claim_verified(tmp_path: Path) -> None:
    registry = setup_run(tmp_path, ["evaluation:metrics.json"])
    out = verify(tmp_path, registry)
    assert (out.n_verified, out.n_rejected) == (1, 0)
    audited = json.loads((tmp_path / out.verified_claims_path).read_text())
    assert audited[0]["status"] == ClaimStatus.VERIFIED.value
    assert out.counts_by_claim_class == {ClaimClass.INFRASTRUCTURE.value: 1}
    assert out.counts_by_evidence_tier == {ScientificEvidenceTier.NON_SCIENTIFIC.value: 1}
    assert out.counts_by_verification_status == {ClaimStatus.VERIFIED.value: 1}


def test_unregistered_reference_rejected_strict(tmp_path: Path) -> None:
    registry = setup_run(tmp_path, ["evaluation:does_not_exist.json"])
    with pytest.raises(SkillError) as excinfo:
        verify(tmp_path, registry)
    assert excinfo.value.failure_class is FailureClass.UNSUPPORTED_CLAIM
    assert not excinfo.value.retryable
    # Evidence artifacts were registered before the failure was raised.
    report = tmp_path / "steps" / "verification" / "verification_report.json"
    assert report.is_file()
    assert json.loads(report.read_text())["rejected_claim_ids"] == ["energy_mae_holdout"]


def test_corrupted_artifact_rejected(tmp_path: Path) -> None:
    registry = setup_run(tmp_path, ["evaluation:metrics.json"])
    (tmp_path / "steps" / "evaluation" / "metrics.json").write_text("{\"tampered\": true}")
    out = verify(tmp_path, registry, strict=False)
    assert (out.n_verified, out.n_rejected) == (0, 1)


def test_non_strict_counts_without_failing(tmp_path: Path) -> None:
    registry = setup_run(tmp_path, ["evaluation:missing.json"], register=False)
    out = verify(tmp_path, registry, strict=False)
    assert (out.n_claims, out.n_verified, out.n_rejected) == (1, 0, 1)
    assert out.rejection_reason_categories == {"unregistered_artifact": 1}
    rejected = json.loads((tmp_path / out.verified_claims_path).read_text())[0]
    assert rejected["rejection_evidence"] == ["unregistered:evaluation:missing.json"]


def test_agent_prose_cannot_verify_a_claim(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    step_dir = tmp_path / "steps" / "reflection"
    step_dir.mkdir(parents=True)
    prose = step_dir / "reflection.md"
    prose.write_text("A persuasive but non-evidentiary narrative.")
    artifact = registry.register(prose, kind="reflection", step_id="reflection")
    claim = Claim(
        claim_id="prose_is_not_evidence",
        statement="Narrative claim",
        value="unsupported",
        created_by_step="reflection",
        artifact_references=[artifact.artifact_id],
    )
    (tmp_path / "claims.json").write_text(json.dumps([claim.model_dump(mode="json")]))
    out = verify(tmp_path, registry, strict=False)
    assert out.rejection_reason_categories == {"agent_prose_as_evidence": 1}


def test_missing_claims_file_rejected(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    with pytest.raises(SkillError) as excinfo:
        verify(tmp_path, registry)
    assert excinfo.value.failure_class is FailureClass.VALIDATION_ERROR
