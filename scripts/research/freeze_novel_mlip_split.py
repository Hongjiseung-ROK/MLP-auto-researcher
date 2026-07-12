#!/usr/bin/env python3
"""Create the outcome-blind, whole-group split for the novel MLIP campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mlip_research_agent.data.registry import NormalizedManifest, sha256_file
from mlip_research_agent.data.split import PartitionName, SplitManifest


def partition_hash(record_ids: list[str], group_ids: list[str]) -> str:
    payload = {"record_ids": sorted(record_ids), "source_group_ids": sorted(group_ids)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_split(manifest_path: Path) -> SplitManifest:
    manifest = NormalizedManifest.load(manifest_path)
    records_by_group: dict[str, list[str]] = {}
    for record in manifest.configurations:
        records_by_group.setdefault(record.group_id, []).append(record.config_id)
    surfaces = sorted(group for group in records_by_group if group.startswith("surface_"))
    if len(surfaces) != 12:
        raise ValueError(f"expected 12 Cu surface groups; found {len(surfaces)}")

    groups = {
        PartitionName.INITIAL_LABELED.value: [
            "elastic_ground_state",
            "elastic_mode_1",
            *surfaces[:11],
        ],
        PartitionName.ACQUISITION_POOL.value: [
            "aimd_nvt_1000k",
            "vacancy_3000k",
            "elastic_mode_2",
            "elastic_mode_3",
            "elastic_mode_4",
            "elastic_mode_5",
            surfaces[11],
        ],
        PartitionName.VALIDATION.value: ["aimd_nvt_300k"],
        PartitionName.FROZEN_TEST.value: ["aimd_nvt_3000k", "vacancy_300k"],
        PartitionName.STRESS_TEST.value: ["elastic_mode_6"],
    }
    expected_groups = {group for partition in groups.values() for group in partition}
    if expected_groups != set(records_by_group):
        missing = sorted(set(records_by_group) - expected_groups)
        unknown = sorted(expected_groups - set(records_by_group))
        raise ValueError(f"group coverage mismatch; missing={missing}, unknown={unknown}")

    record_ids = {
        partition: sorted(
            record_id for group in group_ids for record_id in records_by_group[group]
        )
        for partition, group_ids in groups.items()
    }
    source_group_ids = {
        partition: sorted(group_ids) for partition, group_ids in groups.items()
    }
    actual_sizes = {partition: len(ids) for partition, ids in record_ids.items()}
    expected_sizes = {
        PartitionName.INITIAL_LABELED.value: 32,
        PartitionName.ACQUISITION_POOL.value: 141,
        PartitionName.VALIDATION.value: 40,
        PartitionName.FROZEN_TEST.value: 60,
        PartitionName.STRESS_TEST.value: 20,
    }
    if actual_sizes != expected_sizes:
        raise ValueError(f"unexpected partition sizes: {actual_sizes}")
    split = SplitManifest(
        dataset_id=manifest.dataset_id,
        qualified_manifest_reference=manifest_path.as_posix(),
        qualified_manifest_sha256=sha256_file(manifest_path),
        qualified_dataset_content_sha256=manifest.dataset_content_sha256,
        split_algorithm="prespecified_whole_group_partition",
        split_algorithm_version="1.0.0",
        random_seed=0,
        grouping_fields_used=["group_id"],
        requested_partition_sizes=expected_sizes,
        actual_partition_sizes=actual_sizes,
        record_ids=record_ids,
        source_group_ids=source_group_ids,
        partition_hashes={
            partition: partition_hash(record_ids[partition], source_group_ids[partition])
            for partition in record_ids
        },
        warnings=[
            "Each held-out stratum is one trajectory; frames and atoms are nested, not "
            "independent strategy replicates.",
            "elastic_mode_6 is reserved as a protected physical stress-test set.",
        ],
        fallback_grouping_decisions=[
            "No stochastic splitter was used; named whole group_id assignments were frozen "
            "before new outcome-bearing experiments."
        ],
    )
    split.validate_against(manifest)
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_registry/datasets/cu_phase2/normalized_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/research_spec/novel_mlip_ralph_split.json"),
    )
    args = parser.parse_args()
    split = build_split(args.manifest)
    split.save(args.output)
    audit_path = args.output.with_name("novel_mlip_ralph_split_audit.json")
    audit = {
        "schema_version": "1.0.0",
        "label_access": "none; normalized manifest metadata only",
        "split_path": args.output.as_posix(),
        "split_semantic_sha256": split.semantic_hash(),
        "partition_sizes": split.actual_partition_sizes,
        "partition_groups": split.source_group_ids,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
