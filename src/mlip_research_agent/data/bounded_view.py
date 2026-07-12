"""Least-privilege labeled data package for infrastructure replays."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.data.manifests import LabeledConfiguration
from mlip_research_agent.data.registry import NormalizedManifest, sha256_file
from mlip_research_agent.data.split import PartitionName, SplitManifest

EXPECTED_DATASET_CONTENT_SHA256 = "bc9b78ca5e3b95f91bf34bbc3641a3d6e3f92338b4e3d97065165157848cfc48"
EXPECTED_NORMALIZED_MANIFEST_SHA256 = (
    "ac3655e41ce4327ba18e2a603866b97b00b839ff04f16bb2cb86a3d8ba8a70d3"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = "80d9b95083ebba9d8f988a461525b326ba807c79da834b066d1917afc9d8a15e"
EXPECTED_BOUNDED_VIEW_FILE_SHA256 = (
    "90ef76ca4c927401ac59ee645236af23b6d49d229ec9da53cae0f16dd27e2be7"
)
EXPECTED_INITIAL_COUNT = 32
EXPECTED_VALIDATION_COUNT = 40


class ReplayLabeledConfiguration(BaseModel):
    """Energy/force-only record; stress has no representable field."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    config_id: str
    source_id: str
    top_group: str
    group_id: str
    split_unit_id: str
    symbols: list[str] = Field(min_length=1)
    positions: list[list[float]]
    cell: list[list[float]]
    pbc: list[bool]
    energy_ev: float
    forces_ev_per_a: list[list[float]]
    level_of_theory: str
    source_record_sha256: str = Field(min_length=64, max_length=64)

    @classmethod
    def from_record(cls, record: LabeledConfiguration) -> ReplayLabeledConfiguration:
        return cls.model_validate(record.model_dump(mode="json", exclude={"virial_stress_kbar"}))

    def to_training_record(self) -> LabeledConfiguration:
        return LabeledConfiguration(**self.model_dump(mode="json"), virial_stress_kbar=None)


class BoundedLabelView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0.0"] = "1.0.0"
    scientific_status: Literal["infrastructure_only"] = "infrastructure_only"
    claim_eligible: Literal[False] = False
    dataset_id: Literal["cu_phase2"] = "cu_phase2"
    source_dataset_content_sha256: str = Field(min_length=64, max_length=64)
    normalized_manifest_sha256: str = Field(min_length=64, max_length=64)
    split_manifest_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    energy_unit: Literal["eV"] = "eV"
    force_unit: Literal["eV/angstrom"] = "eV/angstrom"
    initial_labeled: list[ReplayLabeledConfiguration]
    validation: list[ReplayLabeledConfiguration]

    @model_validator(mode="after")
    def _strict_partitions(self) -> BoundedLabelView:
        if self.source_dataset_content_sha256 != EXPECTED_DATASET_CONTENT_SHA256:
            raise ValueError("bounded view source dataset identity mismatch")
        if self.normalized_manifest_sha256 != EXPECTED_NORMALIZED_MANIFEST_SHA256:
            raise ValueError("bounded view normalized manifest identity mismatch")
        if self.split_manifest_sha256 != EXPECTED_SPLIT_MANIFEST_SHA256:
            raise ValueError("bounded view split manifest identity mismatch")
        if len(self.initial_labeled) != EXPECTED_INITIAL_COUNT:
            raise ValueError(
                f"initial_labeled must contain exactly {EXPECTED_INITIAL_COUNT} records"
            )
        if len(self.validation) != EXPECTED_VALIDATION_COUNT:
            raise ValueError(f"validation must contain exactly {EXPECTED_VALIDATION_COUNT} records")
        initial_ids = [record.config_id for record in self.initial_labeled]
        validation_ids = [record.config_id for record in self.validation]
        if initial_ids != sorted(set(initial_ids)):
            raise ValueError("initial_labeled record ids must be sorted and unique")
        if validation_ids != sorted(set(validation_ids)):
            raise ValueError("validation record ids must be sorted and unique")
        if set(initial_ids).intersection(validation_ids):
            raise ValueError("initial and validation views overlap")
        return self

    def content_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, path: Path) -> str:
        text = json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> BoundedLabelView:
        return cls.model_validate_json(path.read_text())


def project_bounded_label_view(
    *,
    dataset_path: Path,
    normalized_manifest_path: Path,
    split_manifest_path: Path,
    data_steward_authorized: bool = False,
) -> BoundedLabelView:
    """Data-steward-only projection; the replay must consume a preprojected view."""
    if not data_steward_authorized:
        raise PermissionError(
            "projection touches the protected source container and requires separate "
            "data-steward authorization"
        )
    if sha256_file(normalized_manifest_path) != EXPECTED_NORMALIZED_MANIFEST_SHA256:
        raise ValueError("normalized manifest SHA-256 mismatch")
    if sha256_file(split_manifest_path) != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise ValueError("split manifest SHA-256 mismatch")
    manifest = NormalizedManifest.load(normalized_manifest_path)
    split = SplitManifest.load(split_manifest_path)
    if manifest.dataset_content_sha256 != EXPECTED_DATASET_CONTENT_SHA256:
        raise ValueError("normalized dataset content SHA-256 mismatch")
    if sha256_file(dataset_path) != manifest.dataset_file_sha256:
        raise ValueError("normalized dataset bytes do not match the committed manifest")
    split.validate_against(manifest)
    initial_ids = sorted(split.record_ids[PartitionName.INITIAL_LABELED.value])
    validation_ids = sorted(split.record_ids[PartitionName.VALIDATION.value])
    selected_ids = set(initial_ids) | set(validation_ids)
    selected_indices = {
        index
        for index, entry in enumerate(manifest.configurations)
        if entry.config_id in selected_ids
    }
    by_id: dict[str, LabeledConfiguration] = {}
    for raw_record in _selected_array_objects(
        dataset_path, key="configurations", selected_indices=selected_indices
    ):
        stress_free = _remove_stress_without_decoding(raw_record)
        record = LabeledConfiguration.model_validate_json(stress_free)
        by_id[record.config_id] = record
    if set(by_id) != selected_ids:
        raise ValueError("streaming projection did not recover exactly the approved record ids")
    metadata = {entry.config_id: entry for entry in manifest.configurations}
    for config_id, record in by_id.items():
        entry = metadata[config_id]
        if (
            record.source_id != entry.source_id
            or record.top_group != entry.top_group
            or record.group_id != entry.group_id
            or record.split_unit_id != entry.split_unit_id
            or record.n_atoms != entry.n_atoms
        ):
            raise ValueError(f"selected record lineage differs from manifest: {config_id}")
    return BoundedLabelView(
        source_dataset_content_sha256=EXPECTED_DATASET_CONTENT_SHA256,
        normalized_manifest_sha256=EXPECTED_NORMALIZED_MANIFEST_SHA256,
        split_manifest_sha256=EXPECTED_SPLIT_MANIFEST_SHA256,
        split_semantic_sha256=split.semantic_hash(),
        initial_labeled=[ReplayLabeledConfiguration.from_record(by_id[i]) for i in initial_ids],
        validation=[ReplayLabeledConfiguration.from_record(by_id[i]) for i in validation_ids],
    )


def verify_bounded_view_membership(view: BoundedLabelView, split: SplitManifest) -> None:
    """Bind the preprojected label view to the committed ID-only split."""
    expected_initial = sorted(split.record_ids[PartitionName.INITIAL_LABELED.value])
    expected_validation = sorted(split.record_ids[PartitionName.VALIDATION.value])
    if [record.config_id for record in view.initial_labeled] != expected_initial:
        raise ValueError("bounded view initial IDs differ from the committed split")
    if [record.config_id for record in view.validation] != expected_validation:
        raise ValueError("bounded view validation IDs differ from the committed split")


def _selected_array_objects(
    path: Path, *, key: str, selected_indices: set[int]
) -> Iterator[str]:
    """Stream a JSON array and materialize only selected object indices.

    Bytes for non-selected records are traversed only to find JSON boundaries;
    their objects and labels are never decoded or retained.
    """
    marker = f'"{key}"'
    with path.open() as handle:
        window = ""
        while marker not in window:
            char = handle.read(1)
            if not char:
                raise ValueError(f"JSON key {key!r} not found")
            window = (window + char)[-len(marker) :]
        char = handle.read(1)
        while char and char.isspace():
            char = handle.read(1)
        if char != ":":
            raise ValueError(f"JSON key {key!r} has no value")
        char = handle.read(1)
        while char and char.isspace():
            char = handle.read(1)
        if char != "[":
            raise ValueError(f"JSON key {key!r} is not an array")

        index = 0
        while True:
            char = handle.read(1)
            while char and (char.isspace() or char == ","):
                char = handle.read(1)
            if char == "]":
                break
            if char != "{":
                raise ValueError(f"{key}[{index}] is not an object")
            keep = index in selected_indices
            pieces = [char] if keep else []
            depth = 1
            in_string = False
            escaped = False
            while depth:
                char = handle.read(1)
                if not char:
                    raise ValueError(f"unterminated object at {key}[{index}]")
                if keep:
                    pieces.append(char)
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char in "[{":
                    depth += 1
                elif char in "]}":
                    depth -= 1
            if keep:
                yield "".join(pieces)
            index += 1
        if selected_indices and max(selected_indices) >= index:
            raise ValueError("selected record index exceeds the dataset array")


def _remove_stress_without_decoding(raw_record: str) -> str:
    """Erase the flat stress field before JSON decoding (stress is forbidden)."""
    pattern = r',\s*"virial_stress_kbar"\s*:\s*(?:null|\[[^\[\]]*\])'
    stress_free, count = re.subn(pattern, "", raw_record, count=1, flags=re.DOTALL)
    if count != 1:
        raise ValueError("selected normalized record has no removable stress field")
    return stress_free
