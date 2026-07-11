"""WP0 gates: preregistration hashing, freeze validation, approvals, matrix
fairness, honest pilot completion."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_research_agent.research import (
    ApprovalRecord,
    HumanGate,
    PilotStage,
    PilotStatus,
    default_phase2a_matrix,
    load_pilot_config,
    require_approval,
)
from mlip_research_agent.research.experiment_matrix import (
    AcquisitionPolicy,
    ArmName,
    ArmSpec,
    ExperimentMatrix,
)
from mlip_research_agent.research.pilot_status import STAGE_ORDER
from mlip_research_agent.research.preregistration import (
    ApprovalMissingError,
    append_approval,
    load_approvals,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG = REPO_ROOT / "configs" / "research" / "cu_mace_al_pilot.yaml"


def test_pilot_config_loads_and_hash_is_deterministic() -> None:
    cfg1 = load_pilot_config(PILOT_CONFIG)
    cfg2 = load_pilot_config(PILOT_CONFIG)
    assert cfg1.preregistration.content_hash() == cfg2.preregistration.content_hash()
    assert len(cfg1.preregistration.content_hash()) == 64


def test_draft_preregistration_reports_pending_fields() -> None:
    prereg = load_pilot_config(PILOT_CONFIG).preregistration
    pending = prereg.pending_fields()
    assert "preregistration.dataset.qualified_manifest_sha256" in pending
    assert "preregistration.model_section.checkpoint_sha256" in pending
    assert not prereg.is_frozen_candidate


def test_result_run_refused_while_draft() -> None:
    prereg = load_pilot_config(PILOT_CONFIG).preregistration
    with pytest.raises(ApprovalMissingError, match="pending fields"):
        prereg.assert_result_run_allowed([])


def test_result_run_refused_without_h2_even_when_frozen() -> None:
    prereg = load_pilot_config(PILOT_CONFIG).preregistration
    frozen = prereg.model_copy(deep=True)
    for path, value in (
        ("dataset.qualified_manifest_sha256", "a" * 64),
        ("dataset.license_spdx", "BSD-3-Clause"),
        ("dataset.level_of_theory", "VASP PBE PAW"),
        ("dataset.overlap_risk_statement", "assessed low"),
        ("split.grouping_rule", "source_group"),
        ("split.split_manifest_sha256", "b" * 64),
        ("model_section.torch_version", "2.13.0"),
        ("model_section.checkpoint_id", "mace-mp-0-small"),
        ("model_section.checkpoint_sha256", "c" * 64),
        ("model_section.checkpoint_license", "MIT"),
        ("finetune.optimizer", "adam"),
        ("finetune.learning_rate", "0.01"),
    ):
        section_name, field = path.split(".")
        setattr(getattr(frozen, section_name), field, value)
    assert frozen.is_frozen_candidate
    with pytest.raises(ApprovalMissingError, match="no H2 approval"):
        frozen.assert_result_run_allowed([])

    approval = ApprovalRecord.create(
        HumanGate.H2_PREREGISTRATION,
        approved=True,
        approved_by="owner",
        subject_sha256=frozen.content_hash(),
    )
    frozen.assert_result_run_allowed([approval])  # does not raise


def test_rejected_approval_blocks() -> None:
    rejection = ApprovalRecord.create(
        HumanGate.H1_DATASET_PROMOTION,
        approved=False,
        approved_by="owner",
        subject_sha256="d" * 64,
    )
    with pytest.raises(ApprovalMissingError, match="rejected"):
        require_approval(HumanGate.H1_DATASET_PROMOTION, "d" * 64, [rejection])


def test_approval_hash_mismatch_blocks() -> None:
    approval = ApprovalRecord.create(
        HumanGate.H3_REMOTE_EXECUTION,
        approved=True,
        approved_by="owner",
        subject_sha256="e" * 64,
    )
    with pytest.raises(ApprovalMissingError):
        require_approval(HumanGate.H3_REMOTE_EXECUTION, "f" * 64, [approval])


def test_approvals_roundtrip_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "approvals.jsonl"
    r1 = ApprovalRecord.create(
        HumanGate.H1_DATASET_PROMOTION,
        approved=True,
        approved_by="owner",
        subject_sha256="a" * 64,
    )
    r2 = ApprovalRecord.create(
        HumanGate.H2_PREREGISTRATION,
        approved=True,
        approved_by="owner",
        subject_sha256="b" * 64,
        notes="frozen 2026-07-11",
    )
    append_approval(path, r1)
    append_approval(path, r2)
    assert load_approvals(path) == [r1, r2]
    assert load_approvals(tmp_path / "missing.jsonl") == []


def test_matrix_fairness_enforced() -> None:
    matrix = default_phase2a_matrix()
    by_name = {arm.name: arm for arm in matrix.arms}
    random_arm = by_name[ArmName.RANDOM]
    hybrid = by_name[ArmName.HYBRID]
    assert random_arm.added_label_budget == hybrid.added_label_budget
    assert random_arm.acquisition_rounds == hybrid.acquisition_rounds

    unfair = [arm.model_copy(deep=True) for arm in matrix.arms]
    for arm in unfair:
        if arm.name is ArmName.HYBRID:
            arm.added_label_budget += 8
    with pytest.raises(ValidationError, match="equal added_label_budget"):
        ExperimentMatrix(
            arms=unfair,
            ensemble_seeds=matrix.ensemble_seeds,
            acquisition_batch_min=matrix.acquisition_batch_min,
            acquisition_batch_max=matrix.acquisition_batch_max,
        )


def test_matrix_requires_all_four_arms() -> None:
    matrix = default_phase2a_matrix()
    three = [a for a in matrix.arms if a.name is not ArmName.ZERO_SHOT]
    with pytest.raises(ValidationError, match="missing"):
        ExperimentMatrix(
            arms=three,
            ensemble_seeds=matrix.ensemble_seeds,
            acquisition_batch_min=8,
            acquisition_batch_max=16,
        )


def test_non_acquiring_arm_cannot_carry_budget() -> None:
    with pytest.raises(ValidationError, match="zero budget"):
        ArmSpec(
            name=ArmName.STATIC_FINETUNE,
            initial_labels=32,
            added_label_budget=8,
            acquisition_rounds=0,
            acquisition_policy=AcquisitionPolicy.NONE,
            finetune=True,
            selection_seed=0,
        )


def test_pilot_status_partial_cannot_be_complete(tmp_path: Path) -> None:
    status = PilotStatus(
        run_id="r1",
        campaign_name="cu-mace-al-phase2-pilot",
        preregistration_sha256="a" * 64,
        repo_commit="0123abc",
    )
    status.mark_stage_complete(PilotStage.PREREG_VERIFIED)
    with pytest.raises(ValidationError, match="unfinished stages"):
        PilotStatus(
            run_id="r1",
            campaign_name="c",
            preregistration_sha256="a" * 64,
            repo_commit="0123abc",
            stages_completed=[PilotStage.PREREG_VERIFIED],
            complete=True,
        )

    for stage in STAGE_ORDER:
        status.mark_stage_complete(stage)
    status.complete = True
    saved = status.save(tmp_path)
    assert saved.name == "completion_status.json"
    assert PilotStatus.load(tmp_path) == status
