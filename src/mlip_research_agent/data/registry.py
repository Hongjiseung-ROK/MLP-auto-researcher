"""Dataset registry: raw-source verification and qualified promotion.

Layout per dataset (plan_phase_2.md §4.3):

    data_registry/datasets/<dataset_id>/
      source.json                # committed — provenance + pinned URLs/hashes
      raw_manifest.json          # committed — raw file hashes (fail-closed)
      normalized_manifest.json   # committed — per-config lineage + hashes
      qualification_report.json  # committed
      dataset_card.md            # committed
      license.txt                # committed
      normalized_dataset.json    # NOT committed (full data; content-addressed)

Promotion to claim-eligible requires BOTH a passing qualification report and
a recorded H1 approval whose subject hash is the normalized dataset's
content hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mlip_research_agent.data.manifests import NormalizedDataset
from mlip_research_agent.data.qualification import (
    QUALIFICATION_REPORT_NAME,
    QualificationReport,
)
from mlip_research_agent.research.preregistration import (
    ApprovalRecord,
    HumanGate,
    require_approval,
)

SOURCE_NAME = "source.json"
RAW_MANIFEST_NAME = "raw_manifest.json"
NORMALIZED_MANIFEST_NAME = "normalized_manifest.json"
NORMALIZED_DATASET_NAME = "normalized_dataset.json"


class RawFileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    local_path: str = Field(description="Path relative to the repository root")
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)


class SourceDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    citation: str
    canonical_repo: str
    pinned_commit: str = Field(min_length=7)
    license_spdx: str
    redistribution_allowed: bool
    files: list[RawFileRecord] = Field(min_length=1)


class ChecksumMismatchError(Exception):
    """Raw data differ from the pinned hashes. ABORT (plan §13.1)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw_files(source: SourceDescriptor, repo_root: Path) -> None:
    """Recompute every raw hash; any mismatch or missing file aborts."""
    for record in source.files:
        path = repo_root / record.local_path
        if not path.is_file():
            raise ChecksumMismatchError(f"raw file missing: {record.local_path}")
        actual = sha256_file(path)
        if actual != record.sha256:
            raise ChecksumMismatchError(
                f"checksum mismatch for {record.local_path}: "
                f"expected {record.sha256}, got {actual}"
            )


class ConfigLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_id: str
    source_id: str
    top_group: str
    group_id: str
    split_unit_id: str
    n_atoms: int = Field(gt=0)
    content_sha256: str = Field(min_length=64, max_length=64)


class NormalizedManifest(BaseModel):
    """Committed summary of the (uncommitted) full normalized dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    format_version: str
    level_of_theory: str
    n_configurations: int = Field(gt=0)
    dataset_content_sha256: str = Field(
        min_length=64, max_length=64, description="NormalizedDataset.content_hash()"
    )
    dataset_file_sha256: str = Field(
        min_length=64, max_length=64, description="SHA-256 of normalized_dataset.json bytes"
    )
    configurations: list[ConfigLineage] = Field(min_length=1)

    @classmethod
    def from_dataset(cls, dataset: NormalizedDataset, file_sha256: str) -> NormalizedManifest:
        return cls(
            dataset_id=dataset.dataset_id,
            format_version=dataset.format_version,
            level_of_theory=dataset.level_of_theory,
            n_configurations=len(dataset.configurations),
            dataset_content_sha256=dataset.content_hash(),
            dataset_file_sha256=file_sha256,
            configurations=[
                ConfigLineage(
                    config_id=c.config_id,
                    source_id=c.source_id,
                    top_group=c.top_group,
                    group_id=c.group_id,
                    split_unit_id=c.split_unit_id,
                    n_atoms=c.n_atoms,
                    content_sha256=c.content_hash(),
                )
                for c in dataset.configurations
            ],
        )

    def save(self, path: Path) -> str:
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=1) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> NormalizedManifest:
        return cls.model_validate(json.loads(path.read_text()))


class DatasetNotQualifiedError(Exception):
    """Fail-closed guard for claim-eligible dataset use."""


def load_qualified_dataset(
    dataset_dir: Path, approvals: list[ApprovalRecord]
) -> NormalizedDataset:
    """The only sanctioned path to a claim-eligible dataset.

    Verifies: qualification report exists and passed; the stored dataset file
    matches the committed manifest hashes; H1 approval matches the dataset
    content hash. Any failure raises.
    """
    report_path = dataset_dir / QUALIFICATION_REPORT_NAME
    if not report_path.is_file():
        raise DatasetNotQualifiedError(f"missing {QUALIFICATION_REPORT_NAME} in {dataset_dir}")
    report = QualificationReport.load(report_path)
    if not report.passed:
        failed = [c.name for c in report.checks if not c.passed]
        raise DatasetNotQualifiedError(f"qualification failed: {failed}")

    manifest = NormalizedManifest.load(dataset_dir / NORMALIZED_MANIFEST_NAME)
    dataset_path = dataset_dir / NORMALIZED_DATASET_NAME
    if not dataset_path.is_file():
        raise DatasetNotQualifiedError(
            f"normalized dataset file missing: {dataset_path} "
            "(re-run scripts/data/qualify_cu_benchmark.py)"
        )
    actual_file_hash = sha256_file(dataset_path)
    if actual_file_hash != manifest.dataset_file_sha256:
        raise DatasetNotQualifiedError(
            "normalized dataset bytes do not match the committed manifest"
        )
    dataset = NormalizedDataset.load(dataset_path)
    if dataset.content_hash() != manifest.dataset_content_sha256:
        raise DatasetNotQualifiedError("dataset content hash mismatch vs manifest")

    require_approval(HumanGate.H1_DATASET_PROMOTION, manifest.dataset_content_sha256, approvals)
    return dataset
