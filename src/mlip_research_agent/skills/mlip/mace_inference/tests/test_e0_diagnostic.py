"""D0-only energy-reference diagnostic tests."""

from pathlib import Path

import pytest

from mlip_research_agent.data.manifests import LabeledConfiguration, NormalizedDataset
from mlip_research_agent.data.registry import NormalizedManifest
from mlip_research_agent.data.split import SplitManifest, SplitSpec, build_split_manifest
from mlip_research_agent.schemas.predictions import PredictionBatch, PredictionRecord
from mlip_research_agent.skills.mlip.mace_inference.diagnostics import (
    compute_e0_diagnostic,
)


def _fixture(tmp_path: Path) -> tuple[NormalizedDataset, SplitManifest]:
    records = [
        LabeledConfiguration(
            config_id=f"record-{index}",
            source_id=f"source[{index}]",
            top_group=f"top-{index % 2}",
            group_id=f"group-{index}",
            split_unit_id=f"unit-{index}",
            symbols=["Cu"],
            positions=[[0.0, 0.0, 0.0]],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            pbc=[True, True, True],
            energy_ev=float(index),
            forces_ev_per_a=[[0.0, 0.0, 0.0]],
            level_of_theory="synthetic",
            source_record_sha256=f"{index + 1:064x}",
        )
        for index in range(8)
    ]
    dataset = NormalizedDataset(
        dataset_id="e0-fixture", level_of_theory="synthetic", configurations=records
    )
    dataset_path = tmp_path / "dataset.json"
    dataset_sha = dataset.save(dataset_path)
    manifest = NormalizedManifest.from_dataset(dataset, dataset_sha)
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    split = build_split_manifest(
        manifest,
        SplitSpec(
            initial_labeled=2,
            acquisition_pool=2,
            validation=2,
            frozen_test=2,
            seed=7,
            max_partition_size_deviation_fraction=0.0,
        ),
        qualified_manifest_path=manifest_path,
        qualified_manifest_reference="manifest.json",
    )
    return dataset, split


def _d0_view(dataset: NormalizedDataset, split: SplitManifest) -> NormalizedDataset:
    by_id = dataset.by_id()
    return NormalizedDataset(
        dataset_id=dataset.dataset_id,
        level_of_theory=dataset.level_of_theory,
        configurations=[by_id[record_id] for record_id in split.record_ids["initial_labeled"]],
    )


def _batch(
    dataset: NormalizedDataset,
    split: SplitManifest,
    model_id: str,
    offsets: list[float],
) -> PredictionBatch:
    record_ids = split.record_ids["initial_labeled"]
    by_id = dataset.by_id()
    return PredictionBatch(
        dataset_id=dataset.dataset_id,
        dataset_content_sha256=split.qualified_dataset_content_sha256,
        split_semantic_sha256=split.semantic_hash(),
        model_id=model_id,
        model_manifest_artifact=f"{model_id}:model.json",
        checkpoint_sha256=("a" if model_id == "a" else "b") * 64,
        energy_unit="eV",
        force_unit="eV/angstrom",
        structure_set_sha256="f" * 64,
        predictions=[
            PredictionRecord(
                record_id=record_id,
                energy_ev=by_id[record_id].energy_ev + offset,
                forces_ev_per_a=[[0.0, 0.0, 0.0]],
            )
            for record_id, offset in zip(record_ids, offsets, strict=True)
        ],
    )


def test_raw_offsets_are_hand_calculated_and_never_select(tmp_path: Path) -> None:
    dataset, split = _fixture(tmp_path)
    d0 = _d0_view(dataset, split)
    report = compute_e0_diagnostic(
        d0,
        dataset.content_hash(),
        split,
        [_batch(dataset, split, "a", [1.0, -1.0]), _batch(dataset, split, "b", [2.0, 2.0])],
        {"a": -3.2, "b": -0.6},
    )
    a, b = report.candidates
    assert a.mean_offset_ev_per_atom == pytest.approx(0.0)
    assert a.std_offset_ev_per_atom == pytest.approx(1.0)
    assert a.mae_offset_ev_per_atom == pytest.approx(1.0)
    assert a.p95_abs_offset_ev_per_atom == pytest.approx(1.0)
    assert b.mean_offset_ev_per_atom == pytest.approx(2.0)
    assert all(candidate.no_offset_fit_applied for candidate in report.candidates)
    assert all(
        candidate.eligibility_status == "threshold_not_frozen"
        for candidate in report.candidates
    )
    assert report.selection_status == "evidence_only_not_h2_selected"


def test_non_d0_or_wrong_provenance_is_rejected(tmp_path: Path) -> None:
    dataset, split = _fixture(tmp_path)
    d0 = _d0_view(dataset, split)
    batch = _batch(dataset, split, "a", [0.0, 0.0])
    wrong = batch.model_copy(update={"dataset_content_sha256": "0" * 64})
    with pytest.raises(ValueError, match="content hash mismatch"):
        compute_e0_diagnostic(d0, dataset.content_hash(), split, [wrong], {"a": -3.2})

    extra = batch.model_copy(
        update={
            "predictions": [
                *batch.predictions,
                batch.predictions[0].model_copy(update={"record_id": "not-d0"}),
            ]
        }
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        compute_e0_diagnostic(d0, dataset.content_hash(), split, [extra], {"a": -3.2})
