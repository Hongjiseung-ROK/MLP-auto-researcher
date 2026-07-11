"""WP1 gates: mlearn parsing, normalization determinism, qualification
fail-closed behavior, registry promotion guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlip_research_agent.data.manifests import (
    LabeledConfiguration,
    NormalizedDataset,
    to_extxyz,
)
from mlip_research_agent.data.mlearn import (
    MlearnParseError,
    classify_record,
    parse_mlearn_file,
)
from mlip_research_agent.data.qualification import qualify_dataset
from mlip_research_agent.data.registry import (
    ChecksumMismatchError,
    NormalizedManifest,
    RawFileRecord,
    SourceDescriptor,
    verify_raw_files,
)
from mlip_research_agent.research.preregistration import ApprovalMissingError

FIXTURE = Path(__file__).parent / "fixtures" / "mlearn_cu_sample.json"


def load_fixture_configs() -> list[LabeledConfiguration]:
    return parse_mlearn_file(FIXTURE, dataset_id="cu_fixture")


def test_fixture_parses_with_expected_groups() -> None:
    configs = load_fixture_configs()
    assert len(configs) == 4
    by_group = {c.top_group: c for c in configs}
    assert by_group["Elastic"].group_id == "elastic_mode_2"
    assert by_group["Elastic"].split_unit_id == "elastic_mode_2"
    # Snapshot 20 with block size 10 -> zero-based block 1.
    assert by_group["AIMD-NVT"].group_id == "aimd_nvt_300k"
    assert by_group["AIMD-NVT"].split_unit_id == "aimd_nvt_300k_block1"
    # Snapshot 1 with block size 5 -> block 0.
    assert by_group["Vacancy"].split_unit_id == "vacancy_300k_block0"
    assert by_group["Surface"].split_unit_id == "surface_1_1_0"
    for cfg in configs:
        assert cfg.pbc == [True, True, True]
        assert cfg.n_atoms == len(cfg.forces_ev_per_a)
        assert cfg.config_id.startswith("cu_fixture-")


def test_classify_rejects_unknown_group_and_bad_descriptions() -> None:
    with pytest.raises(MlearnParseError, match="unknown mlearn group"):
        classify_record("Molten", "whatever")
    with pytest.raises(MlearnParseError, match="unrecognized AIMD"):
        classify_record("AIMD-NVT", "Snapshot ??? garbled")
    with pytest.raises(MlearnParseError, match="unrecognized Surface"):
        classify_record("Surface", "no miller indices here")


def test_malformed_record_rejected(tmp_path: Path) -> None:
    records = json.loads(FIXTURE.read_text())
    del records[0]["outputs"]["forces"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(records))
    with pytest.raises(MlearnParseError, match="malformed record"):
        parse_mlearn_file(bad, dataset_id="cu_fixture")


def test_nonfinite_labels_fail_qualification(tmp_path: Path) -> None:
    configs = load_fixture_configs()
    poisoned = configs[0].model_copy(update={"energy_ev": float("nan")})
    dataset = NormalizedDataset(
        dataset_id="cu_fixture",
        level_of_theory="test",
        configurations=[poisoned, *configs[1:]],
    )
    report = qualify_dataset(dataset)
    assert not report.passed
    failed = {c.name for c in report.checks if not c.passed}
    assert "labels_finite" in failed


def test_duplicate_content_detected() -> None:
    configs = load_fixture_configs()
    clone = configs[0].model_copy(update={"config_id": "cu_fixture-clone", "source_id": "x[9]"})
    dataset = NormalizedDataset(
        dataset_id="cu_fixture",
        level_of_theory="test",
        configurations=[*configs, clone],
    )
    report = qualify_dataset(dataset)
    assert report.exact_duplicate_count == 1
    assert not report.passed


def test_dataset_serialization_is_byte_deterministic(tmp_path: Path) -> None:
    configs = load_fixture_configs()
    dataset = NormalizedDataset(
        dataset_id="cu_fixture", level_of_theory="test", configurations=configs
    )
    sha_a = dataset.save(tmp_path / "a.json")
    sha_b = dataset.save(tmp_path / "b.json")
    assert sha_a == sha_b
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    reloaded = NormalizedDataset.load(tmp_path / "a.json")
    assert reloaded.content_hash() == dataset.content_hash()


def test_extxyz_export_deterministic_and_parseable_by_ase(tmp_path: Path) -> None:
    configs = load_fixture_configs()
    text_a = to_extxyz(configs)
    text_b = to_extxyz(configs)
    assert text_a == text_b
    path = tmp_path / "sample.extxyz"
    path.write_text(text_a)
    ase_io = pytest.importorskip("ase.io")
    atoms_list = ase_io.read(path, index=":")
    assert len(atoms_list) == 4
    first = atoms_list[0]
    assert first.get_chemical_symbols() == configs[0].symbols
    # REF_* keys must round-trip untouched (ASE would capture plain
    # energy/forces into a calculator, hiding them from MACE's reader).
    assert abs(first.info["REF_energy"] - configs[0].energy_ev) < 1e-8
    assert first.arrays["REF_forces"].shape == (configs[0].n_atoms, 3)
    assert abs(first.arrays["REF_forces"][0][0] - configs[0].forces_ev_per_a[0][0]) < 1e-8


def test_checksum_mismatch_aborts(tmp_path: Path) -> None:
    payload = tmp_path / "raw.json"
    payload.write_text("[]")
    source = SourceDescriptor(
        dataset_id="cu_fixture",
        citation="test",
        canonical_repo="https://example.invalid/repo",
        pinned_commit="0123456789ab",
        license_spdx="BSD-3-Clause",
        redistribution_allowed=True,
        files=[
            RawFileRecord(
                url="https://example.invalid/raw.json",
                local_path="raw.json",
                sha256="0" * 64,
                size_bytes=2,
            )
        ],
    )
    with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
        verify_raw_files(source, tmp_path)
    source_missing = source.model_copy(deep=True)
    source_missing.files[0].local_path = "missing.json"
    with pytest.raises(ChecksumMismatchError, match="missing"):
        verify_raw_files(source_missing, tmp_path)


def test_qualified_registry_round_trip_requires_h1(tmp_path: Path) -> None:
    from mlip_research_agent.data.qualification import QUALIFICATION_REPORT_NAME
    from mlip_research_agent.data.registry import (
        NORMALIZED_DATASET_NAME,
        NORMALIZED_MANIFEST_NAME,
        DatasetNotQualifiedError,
        load_qualified_dataset,
    )
    from mlip_research_agent.research.preregistration import ApprovalRecord, HumanGate

    configs = load_fixture_configs()
    dataset = NormalizedDataset(
        dataset_id="cu_fixture", level_of_theory="test", configurations=configs
    )
    report = qualify_dataset(dataset)
    assert report.passed

    dataset_dir = tmp_path / "cu_fixture"
    file_sha = dataset.save(dataset_dir / NORMALIZED_DATASET_NAME)
    NormalizedManifest.from_dataset(dataset, file_sha).save(
        dataset_dir / NORMALIZED_MANIFEST_NAME
    )
    report.save(dataset_dir / QUALIFICATION_REPORT_NAME)

    # No H1 approval -> refused.
    with pytest.raises(ApprovalMissingError):
        load_qualified_dataset(dataset_dir, [])

    approval = ApprovalRecord.create(
        HumanGate.H1_DATASET_PROMOTION,
        approved=True,
        approved_by="owner",
        subject_sha256=dataset.content_hash(),
    )
    loaded = load_qualified_dataset(dataset_dir, [approval])
    assert loaded.content_hash() == dataset.content_hash()

    # Tampered dataset bytes -> refused even with approval.
    dataset_path = dataset_dir / NORMALIZED_DATASET_NAME
    dataset_path.write_text(dataset_path.read_text().replace("cu_fixture-", "cu_fixtura-", 1))
    with pytest.raises(DatasetNotQualifiedError, match="do not match"):
        load_qualified_dataset(dataset_dir, [approval])
