"""Phase 2 claim migration and scientific-evidence invariants."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.schemas.claims import (
    Claim,
    ClaimClass,
    ClaimStatus,
    EvaluationPartition,
    MetricValueBinding,
    ScientificEvidenceTier,
)
from mlip_research_agent.schemas.failure import FailureClass
from mlip_research_agent.skills.base import SkillContext, SkillError

ARTIFACTS = [
    "evaluation:metrics.json",
    "data:dataset_manifest.json",
    "split:split_manifest.json",
    "model:model_manifest.json",
    "approval:h5.json",
]


def scientific_kwargs() -> dict[str, object]:
    return {
        "claim_id": "force_mae_test",
        "statement": "Frozen-test force MAE",
        "value": 0.12,
        "units": "eV/angstrom",
        "created_by_step": "evaluation",
        "artifact_references": ARTIFACTS[:4],
        "metric_artifact_references": [ARTIFACTS[0]],
        "metric_value_binding": MetricValueBinding(
            metric_artifact_reference=ARTIFACTS[0],
            json_pointer="/aggregate/force_component_mae_ev_per_a",
        ),
        "dataset_manifest_reference": ARTIFACTS[1],
        "split_manifest_reference": ARTIFACTS[2],
        "model_manifest_reference": ARTIFACTS[3],
        "evaluation_partition": EvaluationPartition.FROZEN_TEST,
        "derived_from_model_metrics": True,
    }


def test_legacy_phase1_claim_migrates_without_scientific_upgrade() -> None:
    legacy = {
        "claim_id": "energy_mae_holdout",
        "statement": "Mock holdout MAE",
        "value": 0.5,
        "units": "eV",
        "created_by_step": "evaluation",
        "artifact_references": ["evaluation:metrics.json"],
        "status": ClaimStatus.REGISTERED.value,
        "rejection_reason": None,
    }
    claim = Claim.model_validate(legacy)
    assert claim.claim_class is ClaimClass.INFRASTRUCTURE
    assert claim.scientific_evidence_tier is ScientificEvidenceTier.NON_SCIENTIFIC
    assert claim.human_approval_references == []


def test_pilot_claim_is_pilot_only() -> None:
    claim = Claim(
        **scientific_kwargs(),
        claim_class=ClaimClass.PILOT_SCIENTIFIC,
        scientific_evidence_tier=ScientificEvidenceTier.PILOT_ONLY,
        source_run_references=["colab-run-1"],
        limitations=["single-seed Colab integration pilot"],
        seed_metadata=[42],
    )
    assert claim.scientific_evidence_tier is ScientificEvidenceTier.PILOT_ONLY

    with pytest.raises(ValidationError, match="requires evidence tier pilot_only"):
        Claim(
            **scientific_kwargs(),
            claim_class=ClaimClass.PILOT_SCIENTIFIC,
            scientific_evidence_tier=ScientificEvidenceTier.REPLICATED,
        )


def test_replicated_claim_requires_multiple_runs() -> None:
    with pytest.raises(ValidationError, match="at least two source runs"):
        Claim(
            **scientific_kwargs(),
            claim_class=ClaimClass.REPLICATED_SCIENTIFIC,
            scientific_evidence_tier=ScientificEvidenceTier.REPLICATED,
            source_run_references=["vessl-run-1"],
        )
    replicated = Claim(
        **scientific_kwargs(),
        claim_class=ClaimClass.REPLICATED_SCIENTIFIC,
        scientific_evidence_tier=ScientificEvidenceTier.REPLICATED,
        source_run_references=["vessl-run-1", "vessl-run-2", "vessl-run-3"],
    )
    assert len(replicated.source_run_references) == 3


def test_publication_claim_requires_replication_and_human_approval() -> None:
    publication_kwargs = scientific_kwargs()
    publication_kwargs["artifact_references"] = ARTIFACTS
    with pytest.raises(ValidationError, match="replicated claim references"):
        Claim(
            **publication_kwargs,
            claim_class=ClaimClass.PUBLICATION,
            scientific_evidence_tier=ScientificEvidenceTier.PUBLICATION_ELIGIBLE,
            source_run_references=["vessl-run-1", "vessl-run-2"],
            human_approval_references=[ARTIFACTS[4]],
        )
    publication = Claim(
        **publication_kwargs,
        claim_class=ClaimClass.PUBLICATION,
        scientific_evidence_tier=ScientificEvidenceTier.PUBLICATION_ELIGIBLE,
        source_run_references=["vessl-run-1", "vessl-run-2"],
        replicated_claim_references=["force_mae_replication"],
        human_approval_references=[ARTIFACTS[4]],
    )
    assert publication.scientific_evidence_tier is ScientificEvidenceTier.PUBLICATION_ELIGIBLE


def test_numerical_scientific_claim_requires_metrics_and_dataset() -> None:
    kwargs = scientific_kwargs()
    kwargs["metric_artifact_references"] = []
    with pytest.raises(ValidationError, match="metric artifact"):
        Claim(
            **kwargs,
            claim_class=ClaimClass.PILOT_SCIENTIFIC,
            scientific_evidence_tier=ScientificEvidenceTier.PILOT_ONLY,
        )
    kwargs = scientific_kwargs()
    kwargs["metric_value_binding"] = None
    with pytest.raises(ValidationError, match="exact metric value binding"):
        Claim(
            **kwargs,
            claim_class=ClaimClass.PILOT_SCIENTIFIC,
            scientific_evidence_tier=ScientificEvidenceTier.PILOT_ONLY,
        )
    kwargs = scientific_kwargs()
    kwargs["dataset_manifest_reference"] = None
    with pytest.raises(ValidationError, match="dataset manifest"):
        Claim(
            **kwargs,
            claim_class=ClaimClass.PILOT_SCIENTIFIC,
            scientific_evidence_tier=ScientificEvidenceTier.PILOT_ONLY,
        )


def test_frozen_test_and_model_metrics_require_manifests() -> None:
    kwargs = scientific_kwargs()
    kwargs["split_manifest_reference"] = None
    with pytest.raises(ValidationError, match="frozen-test"):
        Claim(
            **kwargs,
            claim_class=ClaimClass.PILOT_SCIENTIFIC,
            scientific_evidence_tier=ScientificEvidenceTier.PILOT_ONLY,
        )
    kwargs = scientific_kwargs()
    kwargs["model_manifest_reference"] = None
    with pytest.raises(ValidationError, match="model-derived"):
        Claim(
            **kwargs,
            claim_class=ClaimClass.PILOT_SCIENTIFIC,
            scientific_evidence_tier=ScientificEvidenceTier.PILOT_ONLY,
        )


def test_run_context_cannot_mint_elevated_claims(tmp_path: Path) -> None:
    claim = Claim(
        **scientific_kwargs(),
        claim_class=ClaimClass.REPLICATED_SCIENTIFIC,
        scientific_evidence_tier=ScientificEvidenceTier.REPLICATED,
        source_run_references=["run-a", "run-b"],
    )
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="evaluation",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )
    with pytest.raises(SkillError) as excinfo:
        ctx.register_claim(claim)
    assert excinfo.value.failure_class is FailureClass.UNSUPPORTED_CLAIM
