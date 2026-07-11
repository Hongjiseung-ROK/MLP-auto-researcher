"""Fail-closed data, lineage, and artifact checks for MACE fine-tuning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mlip_research_agent.artifacts.registry import Artifact, ArtifactRegistry
from mlip_research_agent.data.manifests import LabeledConfiguration, NormalizedDataset
from mlip_research_agent.data.oracle import TrainingLineageManifest
from mlip_research_agent.data.registry import NormalizedManifest
from mlip_research_agent.data.split import PartitionName, SplitManifest
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError
from mlip_research_agent.skills.mlip.mace_finetune.schema import (
    FineTuneSubsetManifest,
    MACEFineTuneInput,
)


@dataclass(frozen=True)
class FineTuneDataBundle:
    dataset: NormalizedDataset
    dataset_manifest: NormalizedManifest
    split: SplitManifest
    train_subset: FineTuneSubsetManifest
    validation_subset: FineTuneSubsetManifest
    lineage: TrainingLineageManifest
    train_records: list[LabeledConfiguration]
    validation_records: list[LabeledConfiguration]
    artifacts: dict[str, Artifact]


def validation_error(message: str) -> SkillError:
    return SkillError(
        message,
        failure_class=FailureClass.VALIDATION_ERROR,
        severity=Severity.CRITICAL,
        retryable=False,
    )


def registered_artifact(
    registry: ArtifactRegistry,
    artifact_id: str,
    expected_kind: str,
) -> tuple[Artifact, Path]:
    artifact = registry.get(artifact_id)
    if artifact is None:
        raise validation_error(f"fine-tune input is not registered: {artifact_id}")
    if artifact.kind != expected_kind:
        raise validation_error(
            f"fine-tune artifact {artifact_id} has kind {artifact.kind!r}; "
            f"expected {expected_kind!r}"
        )
    if not registry.verify(artifact_id):
        raise validation_error(f"fine-tune artifact failed integrity check: {artifact_id}")
    path = (registry.run_dir / artifact.relative_path).resolve()
    try:
        path.relative_to(registry.run_dir.resolve())
    except ValueError as exc:
        raise validation_error(f"fine-tune artifact escapes run directory: {artifact_id}") from exc
    return artifact, path


def _finite_records(records: list[LabeledConfiguration], role: str) -> None:
    for record in records:
        values = [
            record.energy_ev,
            *(value for force in record.forces_ev_per_a for value in force),
        ]
        if not all(math.isfinite(value) for value in values):
            raise validation_error(f"{role} contains non-finite labels: {record.config_id}")


def load_and_validate_data(
    params: MACEFineTuneInput,
    registry: ArtifactRegistry,
) -> FineTuneDataBundle:
    requested = {
        "dataset": (params.dataset_artifact, "normalized_dataset"),
        "dataset_manifest": (
            params.dataset_manifest_artifact,
            "normalized_dataset_manifest",
        ),
        "split": (params.split_manifest_artifact, "split_manifest"),
        "train_subset": (
            params.train_subset_manifest_artifact,
            "fine_tune_subset_manifest",
        ),
        "validation_subset": (
            params.validation_subset_manifest_artifact,
            "fine_tune_subset_manifest",
        ),
        "lineage": (params.training_lineage_artifact, "training_lineage_manifest"),
    }
    artifacts: dict[str, Artifact] = {}
    paths: dict[str, Path] = {}
    for role, (artifact_id, kind) in requested.items():
        artifact, path = registered_artifact(registry, artifact_id, kind)
        artifacts[role] = artifact
        paths[role] = path
    try:
        dataset = NormalizedDataset.load(paths["dataset"])
        dataset_manifest = NormalizedManifest.load(paths["dataset_manifest"])
        split = SplitManifest.load(paths["split"])
        train_subset = FineTuneSubsetManifest.model_validate_json(
            paths["train_subset"].read_text()
        )
        validation_subset = FineTuneSubsetManifest.model_validate_json(
            paths["validation_subset"].read_text()
        )
        lineage = TrainingLineageManifest.model_validate(
            json.loads(paths["lineage"].read_text())
        )
    except (OSError, ValueError) as exc:
        raise validation_error(f"fine-tune input parsing failed: {exc}") from exc

    if dataset_manifest.dataset_file_sha256 != artifacts["dataset"].sha256:
        raise validation_error("fine-tune dataset bytes do not match its manifest")
    derived_manifest = NormalizedManifest.from_dataset(dataset, artifacts["dataset"].sha256)
    if derived_manifest != dataset_manifest:
        raise validation_error("fine-tune dataset content/lineage does not match its manifest")
    if split.qualified_manifest_sha256 != artifacts["dataset_manifest"].sha256:
        raise validation_error("fine-tune split does not reference the registered manifest")
    try:
        split.validate_against(dataset_manifest)
    except ValueError as exc:
        raise validation_error(f"fine-tune split validation failed: {exc}") from exc

    identities = (train_subset, validation_subset)
    for subset in identities:
        if subset.dataset_id != dataset.dataset_id:
            raise validation_error(f"{subset.role} subset dataset id mismatch")
        if subset.dataset_content_sha256 != dataset.content_hash():
            raise validation_error(f"{subset.role} subset dataset content hash mismatch")
        if subset.split_semantic_sha256 != split.semantic_hash():
            raise validation_error(f"{subset.role} subset split identity mismatch")
    if train_subset.role != "train" or validation_subset.role != "validation":
        raise validation_error("fine-tune subset roles must be train and validation")
    if train_subset.training_lineage_artifact != params.training_lineage_artifact:
        raise validation_error("training subset lineage artifact mismatch")

    if lineage.dataset_id != dataset.dataset_id:
        raise validation_error("training lineage dataset id mismatch")
    if lineage.dataset_content_sha256 != dataset.content_hash():
        raise validation_error("training lineage dataset content hash mismatch")
    if lineage.split_semantic_sha256 != split.semantic_hash():
        raise validation_error("training lineage split identity mismatch")
    initial_ids = set(split.record_ids[PartitionName.INITIAL_LABELED.value])
    if set(lineage.initial_record_ids) != initial_ids:
        raise validation_error("training lineage does not preserve the frozen initial set")
    if train_subset.record_ids != lineage.training_record_ids:
        raise validation_error("training subset must exactly match the registered lineage head")

    train_ids = set(train_subset.record_ids)
    validation_ids = set(validation_subset.record_ids)
    allowed_train = initial_ids | set(split.record_ids[PartitionName.ACQUISITION_POOL.value])
    protected = (
        set(split.record_ids[PartitionName.VALIDATION.value])
        | set(split.record_ids[PartitionName.FROZEN_TEST.value])
        | set(split.record_ids[PartitionName.STRESS_TEST.value])
    )
    if not train_ids.issubset(allowed_train) or train_ids.intersection(protected):
        raise validation_error("training subset crosses a protected partition boundary")
    if validation_ids != set(split.record_ids[PartitionName.VALIDATION.value]):
        raise validation_error("validation subset must equal the frozen validation partition")
    if train_ids.intersection(validation_ids):
        raise validation_error("training and validation subsets overlap")

    by_id = dataset.by_id()
    try:
        train_records = [by_id[record_id] for record_id in train_subset.record_ids]
        validation_records = [by_id[record_id] for record_id in validation_subset.record_ids]
    except KeyError as exc:
        raise validation_error(f"fine-tune subset references an unknown record: {exc}") from exc
    _finite_records(train_records, "training subset")
    _finite_records(validation_records, "validation subset")
    return FineTuneDataBundle(
        dataset=dataset,
        dataset_manifest=dataset_manifest,
        split=split,
        train_subset=train_subset,
        validation_subset=validation_subset,
        lineage=lineage,
        train_records=train_records,
        validation_records=validation_records,
        artifacts=artifacts,
    )
