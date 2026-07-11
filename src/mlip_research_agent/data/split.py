"""Deterministic, group-aware partitioning for qualified Phase 2 datasets.

The splitter consumes the committed :class:`NormalizedManifest`, not label
values.  Related records are assigned as an indivisible group, and protected
partitions are selected before the acquisition pool.  No timestamp is stored:
the complete manifest is byte-reproducible under D-P2-4.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.data.manifests import LabeledConfiguration, NormalizedDataset
from mlip_research_agent.data.registry import ConfigLineage, NormalizedManifest, sha256_file

SPLIT_SCHEMA_VERSION = "2.0.0"
SPLIT_ALGORITHM = "group_subset_partition"
SPLIT_ALGORITHM_VERSION = "1.0.0"
STRUCTURAL_FINGERPRINT_ALGORITHM_VERSION = "geometry_cluster_v1"


class PartitionName(StrEnum):
    INITIAL_LABELED = "initial_labeled"
    ACQUISITION_POOL = "acquisition_pool"
    VALIDATION = "validation"
    FROZEN_TEST = "frozen_test"
    STRESS_TEST = "stress_test"


class GroupingField(StrEnum):
    SPLIT_UNIT = "split_unit_id"
    GROUP = "group_id"
    TOP_GROUP = "top_group"
    PROVENANCE_SOURCE = "provenance_source"
    STRUCTURAL_FINGERPRINT = "structural_fingerprint"


class SplitError(ValueError):
    """The manifest cannot be partitioned without violating the contract."""


class SplitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_labeled: int = Field(gt=0)
    acquisition_pool: int = Field(gt=0)
    validation: int = Field(gt=0)
    frozen_test: int = Field(gt=0)
    stress_test: int = Field(default=0, ge=0)
    seed: int = Field(ge=0)
    grouping_priority: list[GroupingField] = Field(
        default_factory=lambda: [
            GroupingField.GROUP,
            GroupingField.SPLIT_UNIT,
            GroupingField.TOP_GROUP,
            GroupingField.PROVENANCE_SOURCE,
            GroupingField.STRUCTURAL_FINGERPRINT,
        ],
        min_length=1,
    )
    initial_labeled_min: int = Field(default=1, ge=1)
    acquisition_pool_min: int = Field(default=1, ge=1)
    validation_min: int = Field(default=1, ge=1)
    frozen_test_min: int = Field(default=1, ge=1)
    stress_test_min: int = Field(default=0, ge=0)
    max_partition_size_deviation_fraction: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _unique_grouping_priority(self) -> SplitSpec:
        if len(set(self.grouping_priority)) != len(self.grouping_priority):
            raise ValueError("grouping_priority contains duplicates")
        requested = self.requested_sizes()
        minimums = self.minimum_sizes()
        too_small = {
            name: (requested[name], minimum)
            for name, minimum in minimums.items()
            if requested[name] < minimum
        }
        if too_small:
            raise ValueError(f"requested sizes are below declared minima: {too_small}")
        return self

    def requested_sizes(self) -> dict[str, int]:
        return {
            PartitionName.INITIAL_LABELED.value: self.initial_labeled,
            PartitionName.ACQUISITION_POOL.value: self.acquisition_pool,
            PartitionName.VALIDATION.value: self.validation,
            PartitionName.FROZEN_TEST.value: self.frozen_test,
            PartitionName.STRESS_TEST.value: self.stress_test,
        }

    def minimum_sizes(self) -> dict[str, int]:
        return {
            PartitionName.INITIAL_LABELED.value: self.initial_labeled_min,
            PartitionName.ACQUISITION_POOL.value: self.acquisition_pool_min,
            PartitionName.VALIDATION.value: self.validation_min,
            PartitionName.FROZEN_TEST.value: self.frozen_test_min,
            PartitionName.STRESS_TEST.value: self.stress_test_min,
        }


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SPLIT_SCHEMA_VERSION
    dataset_id: str
    qualified_manifest_reference: str
    qualified_manifest_sha256: str = Field(min_length=64, max_length=64)
    qualified_dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_algorithm: str = SPLIT_ALGORITHM
    split_algorithm_version: str = SPLIT_ALGORITHM_VERSION
    random_seed: int = Field(ge=0)
    grouping_fields_used: list[str] = Field(min_length=1)
    requested_partition_sizes: dict[str, int]
    actual_partition_sizes: dict[str, int]
    record_ids: dict[str, list[str]]
    source_group_ids: dict[str, list[str]]
    partition_hashes: dict[str, str]
    warnings: list[str] = Field(default_factory=list)
    fallback_grouping_decisions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _internal_consistency(self) -> SplitManifest:
        expected = {p.value for p in PartitionName}
        mappings = (
            self.requested_partition_sizes,
            self.actual_partition_sizes,
            self.record_ids,
            self.source_group_ids,
            self.partition_hashes,
        )
        if any(set(mapping) != expected for mapping in mappings):
            raise ValueError("every partition mapping must contain exactly the canonical keys")

        seen_records: set[str] = set()
        seen_groups: set[str] = set()
        for partition in PartitionName:
            name = partition.value
            records = self.record_ids[name]
            groups = self.source_group_ids[name]
            if records != sorted(records) or len(records) != len(set(records)):
                raise ValueError(f"{name}: record ids must be sorted and unique")
            if groups != sorted(groups) or len(groups) != len(set(groups)):
                raise ValueError(f"{name}: source group ids must be sorted and unique")
            if seen_records.intersection(records):
                raise ValueError(f"{name}: record ids cross partition boundaries")
            if seen_groups.intersection(groups):
                raise ValueError(f"{name}: protected source groups cross boundaries")
            seen_records.update(records)
            seen_groups.update(groups)
            if self.actual_partition_sizes[name] != len(records):
                raise ValueError(f"{name}: actual size does not match record count")
            expected_hash = _partition_hash(records, groups)
            if self.partition_hashes[name] != expected_hash:
                raise ValueError(f"{name}: partition hash mismatch")
        return self

    def semantic_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def save(self, path: Path) -> str:
        text = json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> SplitManifest:
        return cls.model_validate(json.loads(path.read_text()))

    def validate_against(
        self,
        manifest: NormalizedManifest,
        *,
        structural_fingerprints: dict[str, str] | None = None,
    ) -> None:
        if manifest.dataset_id != self.dataset_id:
            raise SplitError("split dataset id does not match the qualified manifest")
        if manifest.dataset_content_sha256 != self.qualified_dataset_content_sha256:
            raise SplitError("qualified dataset content hash does not match split provenance")
        expected_ids = {record.config_id for record in manifest.configurations}
        actual_ids = {record for ids in self.record_ids.values() for record in ids}
        missing = sorted(expected_ids - actual_ids)
        unknown = sorted(actual_ids - expected_ids)
        if missing or unknown:
            raise SplitError(
                f"split coverage mismatch; missing={missing[:5]}, unknown={unknown[:5]}"
            )
        try:
            grouping_field = GroupingField(self.grouping_fields_used[0])
        except (ValueError, IndexError) as exc:
            raise SplitError("split declares an unknown grouping field") from exc
        partition_by_record = {
            record_id: partition
            for partition, record_ids in self.record_ids.items()
            for record_id in record_ids
        }
        declared_group_partition = {
            group_id: partition
            for partition, group_ids in self.source_group_ids.items()
            for group_id in group_ids
        }
        for record in manifest.configurations:
            expected_group = _group_value(record, grouping_field, structural_fingerprints)
            partition = partition_by_record[record.config_id]
            if declared_group_partition.get(expected_group) != partition:
                raise SplitError(
                    f"record/group mapping mismatch for {record.config_id}: "
                    f"expected group {expected_group!r} in {partition!r}"
                )


def _provenance_source(record: ConfigLineage) -> str:
    """Stable raw-file provenance group (the prefix before the record index)."""
    return record.source_id.partition("[")[0]


def _determinant3(cell: list[list[float]]) -> float:
    return (
        cell[0][0] * (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1])
        - cell[0][1] * (cell[1][0] * cell[2][2] - cell[1][2] * cell[2][0])
        + cell[0][2] * (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0])
    )


def _geometry_cluster_key(record: LabeledConfiguration) -> str:
    """Coarse, geometry-only cluster; labels and record identity are excluded.

    The quantized features are translation/rotation/atom-order invariant.  It
    is deliberately a cluster key rather than an exact structure hash so the
    fallback cannot silently degenerate into frame-level random splitting.
    """
    nearest: list[float] = []
    for index, position in enumerate(record.positions):
        distances = [
            math.dist(position, other)
            for other_index, other in enumerate(record.positions)
            if other_index != index
        ]
        nearest.append(min(distances) if distances else 0.0)
    nearest_mean = sum(nearest) / len(nearest)
    nearest_variance = sum((value - nearest_mean) ** 2 for value in nearest) / len(nearest)
    cell_lengths = sorted(math.sqrt(sum(value * value for value in row)) for row in record.cell)
    payload = {
        "algorithm": STRUCTURAL_FINGERPRINT_ALGORITHM_VERSION,
        "species_counts": sorted(Counter(record.symbols).items()),
        "n_atoms": record.n_atoms,
        "pbc": record.pbc,
        "volume_per_atom_bin": round(abs(_determinant3(record.cell)) / record.n_atoms, 1),
        "nearest_mean_bin": round(nearest_mean, 1),
        "nearest_std_bin": round(math.sqrt(nearest_variance), 1),
        "cell_length_bins": [round(value, 1) for value in cell_lengths],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"geometry-{digest[:20]}"


def derive_structural_fingerprint_clusters(dataset: NormalizedDataset) -> dict[str, str]:
    """Return deterministic geometry-only cluster ids for fallback grouping."""
    return {
        record.config_id: _geometry_cluster_key(record)
        for record in dataset.configurations
    }


def _group_value(
    record: ConfigLineage,
    field: GroupingField,
    structural_fingerprints: dict[str, str] | None = None,
) -> str:
    if field is GroupingField.PROVENANCE_SOURCE:
        return _provenance_source(record)
    if field is GroupingField.STRUCTURAL_FINGERPRINT:
        if structural_fingerprints is None or record.config_id not in structural_fingerprints:
            raise SplitError(
                "structural-fingerprint fallback requires a geometry-only fingerprint "
                f"for every record; missing {record.config_id}"
            )
        return structural_fingerprints[record.config_id]
    return str(getattr(record, field.value))


def _select_grouping(
    records: list[ConfigLineage],
    spec: SplitSpec,
    required_partitions: int,
    structural_fingerprints: dict[str, str] | None,
) -> tuple[GroupingField, dict[str, list[ConfigLineage]], list[str]]:
    decisions: list[str] = []
    for field in spec.grouping_priority:
        if field is GroupingField.STRUCTURAL_FINGERPRINT and structural_fingerprints is None:
            decisions.append(
                "skipped structural_fingerprint: geometry-only fingerprints were not supplied"
            )
            continue
        grouped: dict[str, list[ConfigLineage]] = defaultdict(list)
        invalid = False
        for record in records:
            value = _group_value(record, field, structural_fingerprints).strip()
            if not value:
                invalid = True
                break
            grouped[value].append(record)
        if invalid:
            decisions.append(f"skipped {field.value}: one or more records lack the field")
            continue
        if len(grouped) < required_partitions:
            decisions.append(
                f"skipped {field.value}: {len(grouped)} groups cannot populate "
                f"{required_partitions} non-empty partitions"
            )
            continue
        if field is GroupingField.STRUCTURAL_FINGERPRINT and len(grouped) == len(records):
            decisions.append(
                "skipped structural_fingerprint: every record formed a singleton cluster"
            )
            continue
        return field, dict(grouped), decisions
    raise SplitError(
        "no reliable grouping field can populate all requested partitions; "
        f"fallback audit: {decisions}"
    )


def _partition_hash(record_ids: list[str], group_ids: list[str]) -> str:
    payload = {"record_ids": sorted(record_ids), "source_group_ids": sorted(group_ids)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _candidate_signature(
    basis: str, partition: str, group_ids: tuple[str, ...]
) -> str:
    payload = f"{basis}|{partition}|{'|'.join(sorted(group_ids))}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _choose_group_subset(
    groups: dict[str, list[ConfigLineage]],
    *,
    target: int,
    minimum: int,
    max_groups: int,
    basis: str,
    partition: str,
) -> set[str]:
    """Choose the closest-size subset with a seed-derived deterministic tie break."""
    if target <= 0:
        return set()
    ordered = sorted(
        groups,
        key=lambda group_id: hashlib.sha256(f"{basis}|{group_id}".encode()).hexdigest(),
    )
    # (record_count, group_count) -> tuple of selected group ids.  Keeping one
    # seed-ranked candidate per state bounds this exact subset search by
    # O(n_records * n_groups^2), tiny for the qualified Cu registry.
    states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
    for group_id in ordered:
        size = len(groups[group_id])
        additions: dict[tuple[int, int], tuple[str, ...]] = {}
        for (total, count), selected in states.items():
            if count >= max_groups:
                continue
            key = (total + size, count + 1)
            candidate = (*selected, group_id)
            existing = states.get(key) or additions.get(key)
            if existing is None or _candidate_signature(
                basis, partition, candidate
            ) < _candidate_signature(basis, partition, existing):
                additions[key] = candidate
        states.update(additions)

    eligible = [
        (total, selected)
        for (total, count), selected in states.items()
        if 0 < count <= max_groups and total >= minimum
    ]
    if not eligible:
        raise SplitError(f"cannot allocate any protected group to {partition}")
    _, best = min(
        eligible,
        key=lambda item: (
            abs(item[0] - target),
            _candidate_signature(basis, partition, item[1]),
        ),
    )
    return set(best)


def build_split_manifest(
    manifest: NormalizedManifest,
    spec: SplitSpec,
    *,
    qualified_manifest_path: Path,
    qualified_manifest_reference: str,
    structural_dataset: NormalizedDataset | None = None,
) -> SplitManifest:
    """Partition every manifest record exactly once without splitting a group."""
    try:
        on_disk_manifest = NormalizedManifest.load(qualified_manifest_path)
    except (OSError, ValueError) as exc:
        raise SplitError(f"malformed qualified manifest: {exc}") from exc
    if on_disk_manifest != manifest:
        raise SplitError("qualified manifest argument does not match the referenced file")
    requested = spec.requested_sizes()
    if sum(requested.values()) != manifest.n_configurations:
        raise SplitError(
            "requested partition sizes must cover the qualified dataset exactly: "
            f"requested {sum(requested.values())}, dataset has {manifest.n_configurations}"
        )
    if manifest.n_configurations != len(manifest.configurations):
        raise SplitError("malformed manifest: n_configurations does not match records")
    ids = [record.config_id for record in manifest.configurations]
    if len(ids) != len(set(ids)):
        raise SplitError("malformed manifest: duplicate config ids")

    structural_fingerprints: dict[str, str] | None = None
    if structural_dataset is not None:
        if structural_dataset.dataset_id != manifest.dataset_id:
            raise SplitError("structural fallback dataset id does not match the manifest")
        if structural_dataset.content_hash() != manifest.dataset_content_sha256:
            raise SplitError("structural fallback dataset content does not match the manifest")
        structural_fingerprints = derive_structural_fingerprint_clusters(structural_dataset)

    non_empty = sum(size > 0 for size in requested.values())
    grouping_field, remaining, fallback_decisions = _select_grouping(
        manifest.configurations, spec, non_empty, structural_fingerprints
    )
    if grouping_field is GroupingField.STRUCTURAL_FINGERPRINT:
        fallback_decisions.append(
            "selected deterministic geometry-only clustering algorithm "
            f"{STRUCTURAL_FINGERPRINT_ALGORITHM_VERSION}"
        )
    manifest_sha = sha256_file(qualified_manifest_path)
    basis_payload = {
        "qualified_manifest_sha256": manifest_sha,
        "spec": spec.model_dump(mode="json"),
        "seed": spec.seed,
        "grouping_field": grouping_field.value,
        "algorithm_version": SPLIT_ALGORITHM_VERSION,
    }
    basis = hashlib.sha256(json.dumps(basis_payload, sort_keys=True).encode()).hexdigest()

    allocation_order = [
        PartitionName.INITIAL_LABELED,
        PartitionName.FROZEN_TEST,
        PartitionName.VALIDATION,
        PartitionName.STRESS_TEST,
    ]
    groups_by_partition: dict[str, set[str]] = {
        partition.value: set() for partition in PartitionName
    }
    remaining_non_empty = non_empty
    for partition in allocation_order:
        target = requested[partition.value]
        if target == 0:
            continue
        remaining_non_empty -= 1
        selected = _choose_group_subset(
            remaining,
            target=target,
            minimum=spec.minimum_sizes()[partition.value],
            max_groups=len(remaining) - remaining_non_empty,
            basis=basis,
            partition=partition.value,
        )
        groups_by_partition[partition.value] = selected
        remaining = {key: value for key, value in remaining.items() if key not in selected}

    if not remaining:
        raise SplitError("acquisition pool received no source groups")
    groups_by_partition[PartitionName.ACQUISITION_POOL.value] = set(remaining)

    source_groups: dict[str, list[str]] = {}
    record_ids: dict[str, list[str]] = {}
    all_groups: dict[str, list[ConfigLineage]] = defaultdict(list)
    for record in manifest.configurations:
        all_groups[_group_value(record, grouping_field, structural_fingerprints)].append(record)
    for partition in PartitionName:
        name = partition.value
        source_groups[name] = sorted(groups_by_partition[name])
        record_ids[name] = sorted(
            record.config_id
            for group_id in source_groups[name]
            for record in all_groups[group_id]
        )

    actual = {name: len(values) for name, values in record_ids.items()}
    minimums = spec.minimum_sizes()
    below_minimum = {
        name: (actual[name], minimum)
        for name, minimum in minimums.items()
        if actual[name] < minimum
    }
    if below_minimum:
        raise SplitError(f"group-aware allocation falls below partition minima: {below_minimum}")
    excessive_deviation = {
        name: (requested[name], actual[name])
        for name in requested
        if requested[name] > 0
        and abs(actual[name] - requested[name]) / requested[name]
        > spec.max_partition_size_deviation_fraction
    }
    if excessive_deviation:
        raise SplitError(
            "group-aware allocation deviates beyond the declared tolerance: "
            f"{excessive_deviation}"
        )
    warnings = [
        f"{name}: requested {requested[name]}, group-aware allocation produced {actual[name]}"
        for name in sorted(requested)
        if requested[name] != actual[name]
    ]
    if grouping_field is GroupingField.SPLIT_UNIT:
        group_partition: dict[str, set[str]] = defaultdict(set)
        by_id = {record.config_id: record for record in manifest.configurations}
        for partition_name, ids_in_partition in record_ids.items():
            for record_id in ids_in_partition:
                group_partition[by_id[record_id].group_id].add(partition_name)
        divided_families = sorted(
            group_id for group_id, partitions in group_partition.items() if len(partitions) > 1
        )
        if divided_families:
            warnings.append(
                "split_unit_id protects contiguous/source units, but broader group_id "
                f"families span partitions: {divided_families}"
            )
    partition_hashes = {
        name: _partition_hash(record_ids[name], source_groups[name])
        for name in record_ids
    }
    split = SplitManifest(
        dataset_id=manifest.dataset_id,
        qualified_manifest_reference=qualified_manifest_reference,
        qualified_manifest_sha256=manifest_sha,
        qualified_dataset_content_sha256=manifest.dataset_content_sha256,
        random_seed=spec.seed,
        grouping_fields_used=[grouping_field.value],
        requested_partition_sizes=requested,
        actual_partition_sizes=actual,
        record_ids=record_ids,
        source_group_ids=source_groups,
        partition_hashes=partition_hashes,
        warnings=warnings,
        fallback_grouping_decisions=fallback_decisions,
    )
    split.validate_against(manifest, structural_fingerprints=structural_fingerprints)
    return split
