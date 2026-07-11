"""WP2 leakage-safe split invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlip_research_agent.data.manifests import LabeledConfiguration, NormalizedDataset
from mlip_research_agent.data.registry import ConfigLineage, NormalizedManifest
from mlip_research_agent.data.split import (
    PartitionName,
    SplitError,
    SplitManifest,
    SplitSpec,
    build_split_manifest,
)


def make_manifest(n_groups: int = 12, records_per_group: int = 2) -> NormalizedManifest:
    records: list[ConfigLineage] = []
    for group_index in range(n_groups):
        for record_index in range(records_per_group):
            record_id = f"fixture-{group_index:02d}-{record_index:02d}"
            records.append(
                ConfigLineage(
                    config_id=record_id,
                    source_id=f"source-{group_index // 3}.json[{record_index}]",
                    top_group=f"family-{group_index // 4}",
                    group_id=f"trajectory-{group_index:02d}",
                    split_unit_id=f"unit-{group_index:02d}",
                    n_atoms=4,
                    content_sha256=f"{group_index * records_per_group + record_index + 1:064x}",
                )
            )
    return NormalizedManifest(
        dataset_id="split_fixture",
        format_version="2.0",
        level_of_theory="synthetic",
        n_configurations=len(records),
        dataset_content_sha256="a" * 64,
        dataset_file_sha256="b" * 64,
        configurations=records,
    )


def write_manifest(manifest: NormalizedManifest, path: Path) -> None:
    manifest.save(path)


def split_spec(seed: int = 7) -> SplitSpec:
    return SplitSpec(
        initial_labeled=4,
        acquisition_pool=10,
        validation=4,
        frozen_test=6,
        seed=seed,
    )


def build(tmp_path: Path, *, seed: int = 7) -> tuple[NormalizedManifest, SplitManifest]:
    manifest = make_manifest()
    path = tmp_path / "normalized_manifest.json"
    write_manifest(manifest, path)
    split = build_split_manifest(
        manifest,
        split_spec(seed),
        qualified_manifest_path=path,
        qualified_manifest_reference="registry/normalized_manifest.json",
    )
    return manifest, split


def test_deterministic_reproduction_and_hash_stability(tmp_path: Path) -> None:
    _, first = build(tmp_path)
    _, second = build(tmp_path)
    assert first == second
    assert first.semantic_hash() == second.semantic_hash()
    assert first.save(tmp_path / "a.json") == second.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_changed_seed_changes_eligible_allocation(tmp_path: Path) -> None:
    _, first = build(tmp_path, seed=7)
    _, second = build(tmp_path, seed=8)
    assert first.record_ids != second.record_ids


def test_record_and_source_groups_are_disjoint_and_complete(tmp_path: Path) -> None:
    manifest, split = build(tmp_path)
    record_sets = [set(ids) for ids in split.record_ids.values()]
    group_sets = [set(ids) for ids in split.source_group_ids.values()]
    for index, records in enumerate(record_sets):
        assert all(records.isdisjoint(other) for other in record_sets[index + 1 :])
    for index, groups in enumerate(group_sets):
        assert all(groups.isdisjoint(other) for other in group_sets[index + 1 :])
    assert set().union(*record_sets) == {record.config_id for record in manifest.configurations}
    assert split.actual_partition_sizes == split.requested_partition_sizes


def test_frozen_test_and_validation_never_enter_learning_partitions(tmp_path: Path) -> None:
    _, split = build(tmp_path)
    learning = set(split.record_ids[PartitionName.INITIAL_LABELED.value]) | set(
        split.record_ids[PartitionName.ACQUISITION_POOL.value]
    )
    assert learning.isdisjoint(split.record_ids[PartitionName.FROZEN_TEST.value])
    assert learning.isdisjoint(split.record_ids[PartitionName.VALIDATION.value])


def test_unknown_record_and_partition_hash_tampering_rejected(tmp_path: Path) -> None:
    manifest, split = build(tmp_path)
    payload = split.model_dump(mode="json")
    payload["record_ids"][PartitionName.ACQUISITION_POOL.value].append("unknown-id")
    payload["record_ids"][PartitionName.ACQUISITION_POOL.value].sort()
    payload["actual_partition_sizes"][PartitionName.ACQUISITION_POOL.value] += 1
    # Recompute is deliberately omitted: schema catches hash/count drift.
    with pytest.raises(ValueError, match="partition hash mismatch"):
        SplitManifest.model_validate(payload)

    valid_payload = split.model_dump(mode="json")
    unknown = valid_payload["record_ids"][PartitionName.ACQUISITION_POOL.value][0]
    manifest_without_one = manifest.model_copy(
        update={
            "n_configurations": manifest.n_configurations - 1,
            "configurations": [r for r in manifest.configurations if r.config_id != unknown],
        }
    )
    with pytest.raises(SplitError, match="coverage mismatch"):
        split.validate_against(manifest_without_one)


def test_insufficient_groups_fail_clearly(tmp_path: Path) -> None:
    manifest = make_manifest(n_groups=3, records_per_group=4)
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    spec = SplitSpec(
        initial_labeled=2,
        acquisition_pool=4,
        validation=2,
        frozen_test=4,
        seed=1,
        grouping_priority=["split_unit_id"],
    )
    with pytest.raises(SplitError, match="no reliable grouping field"):
        build_split_manifest(
            manifest,
            spec,
            qualified_manifest_path=path,
            qualified_manifest_reference="manifest.json",
        )


def test_malformed_or_mismatched_manifest_fails(tmp_path: Path) -> None:
    manifest = make_manifest()
    path = tmp_path / "manifest.json"
    path.write_text("{not-json")
    with pytest.raises(SplitError, match="malformed qualified manifest"):
        build_split_manifest(
            manifest,
            split_spec(),
            qualified_manifest_path=path,
            qualified_manifest_reference="manifest.json",
        )

    write_manifest(manifest, path)
    altered = manifest.model_copy(update={"level_of_theory": "changed"})
    with pytest.raises(SplitError, match="does not match"):
        build_split_manifest(
            altered,
            split_spec(),
            qualified_manifest_path=path,
            qualified_manifest_reference="manifest.json",
        )


def test_semantic_content_contains_no_timestamp(tmp_path: Path) -> None:
    _, split = build(tmp_path)
    serialized = json.dumps(split.model_dump(mode="json"), sort_keys=True)
    assert "timestamp" not in serialized


def test_forged_record_to_group_mapping_is_rejected(tmp_path: Path) -> None:
    manifest, split = build(tmp_path)
    payload = split.model_dump(mode="json")
    pool = PartitionName.ACQUISITION_POOL.value
    test = PartitionName.FROZEN_TEST.value
    pool_id = payload["record_ids"][pool][0]
    test_id = payload["record_ids"][test][0]
    payload["record_ids"][pool][0] = test_id
    payload["record_ids"][test][0] = pool_id
    payload["record_ids"][pool].sort()
    payload["record_ids"][test].sort()
    from mlip_research_agent.data.split import _partition_hash

    payload["partition_hashes"][pool] = _partition_hash(
        payload["record_ids"][pool], payload["source_group_ids"][pool]
    )
    payload["partition_hashes"][test] = _partition_hash(
        payload["record_ids"][test], payload["source_group_ids"][test]
    )
    forged = SplitManifest.model_validate(payload)
    with pytest.raises(SplitError, match="record/group mapping mismatch"):
        forged.validate_against(manifest)


def test_grossly_uneven_groups_fail_deviation_gate(tmp_path: Path) -> None:
    sizes = [100, 1, 1, 1, 1]
    records: list[ConfigLineage] = []
    index = 0
    for group_index, size in enumerate(sizes):
        for _ in range(size):
            records.append(
                ConfigLineage(
                    config_id=f"uneven-{index:03d}",
                    source_id=f"source.json[{index}]",
                    top_group=f"family-{group_index}",
                    group_id=f"group-{group_index}",
                    split_unit_id=f"unit-{group_index}",
                    n_atoms=1,
                    content_sha256=f"{index + 1:064x}",
                )
            )
            index += 1
    manifest = NormalizedManifest(
        dataset_id="uneven_fixture",
        format_version="2.0",
        level_of_theory="synthetic",
        n_configurations=len(records),
        dataset_content_sha256="c" * 64,
        dataset_file_sha256="d" * 64,
        configurations=records,
    )
    path = tmp_path / "uneven.json"
    write_manifest(manifest, path)
    with pytest.raises(SplitError, match="deviates beyond"):
        build_split_manifest(
            manifest,
            SplitSpec(
                initial_labeled=26,
                acquisition_pool=26,
                validation=26,
                frozen_test=26,
                seed=1,
            ),
            qualified_manifest_path=path,
            qualified_manifest_reference="uneven.json",
        )


def test_structural_fingerprint_fallback_is_explicit(tmp_path: Path) -> None:
    configs: list[LabeledConfiguration] = []
    for index in range(8):
        cluster = index // 2
        configs.append(
            LabeledConfiguration(
                config_id=f"fingerprint-{index}",
                source_id=f"unknown.json[{index}]",
                top_group="unknown",
                group_id="unknown",
                split_unit_id="unknown",
                symbols=["Cu", "Cu"],
                positions=[[0.0, 0.0, 0.0], [1.0 + 0.2 * cluster, 0.0, 0.0]],
                cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                pbc=[True, True, True],
                energy_ev=-7.0 + index * 0.01,
                forces_ev_per_a=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                level_of_theory="synthetic",
                source_record_sha256=f"{index + 1:064x}",
            )
        )
    dataset = NormalizedDataset(
        dataset_id="fingerprint_fixture",
        level_of_theory="synthetic",
        configurations=configs,
    )
    path = tmp_path / "fingerprint.json"
    dataset_path = tmp_path / "fingerprint_dataset.json"
    dataset_sha = dataset.save(dataset_path)
    manifest = NormalizedManifest.from_dataset(dataset, dataset_sha)
    write_manifest(manifest, path)
    spec = SplitSpec(
        initial_labeled=2,
        acquisition_pool=2,
        validation=2,
        frozen_test=2,
        seed=2,
        grouping_priority=["structural_fingerprint"],
    )
    split = build_split_manifest(
        manifest,
        spec,
        qualified_manifest_path=path,
        qualified_manifest_reference="fingerprint.json",
        structural_dataset=dataset,
    )
    assert split.grouping_fields_used == ["structural_fingerprint"]
    assert "geometry_cluster_v1" in split.fallback_grouping_decisions[-1]
    from mlip_research_agent.data.oracle import build_acquisition_view

    assert len(build_acquisition_view(dataset, split).candidates) == 2
    with pytest.raises(SplitError, match="no reliable grouping field"):
        build_split_manifest(
            manifest,
            spec,
            qualified_manifest_path=path,
            qualified_manifest_reference="fingerprint.json",
        )


def test_real_cu_split_keeps_broader_groups_wholly_isolated() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = NormalizedManifest.load(
        root / "data_registry/datasets/cu_phase2/normalized_manifest.json"
    )
    split = SplitManifest.load(root / "data_registry/datasets/cu_phase2/split_manifest.json")
    assert split.grouping_fields_used == ["group_id"]
    split.validate_against(manifest)
    partition_by_id = {
        record_id: partition
        for partition, record_ids in split.record_ids.items()
        for record_id in record_ids
    }
    group_partitions: dict[str, set[str]] = {}
    for record in manifest.configurations:
        group_partitions.setdefault(record.group_id, set()).add(
            partition_by_id[record.config_id]
        )
    assert all(len(partitions) == 1 for partitions in group_partitions.values())
