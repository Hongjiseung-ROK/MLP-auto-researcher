"""Golden values and fail-closed boundaries for independent MLIP evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_research_agent.artifacts.registry import ArtifactRegistry, sha256_file
from mlip_research_agent.data.manifests import LabeledConfiguration, NormalizedDataset
from mlip_research_agent.data.registry import NormalizedManifest
from mlip_research_agent.data.split import (
    PartitionName,
    SplitManifest,
    SplitSpec,
    _partition_hash,
    build_split_manifest,
)
from mlip_research_agent.skills.base import SkillContext, SkillError, get_skill
from mlip_research_agent.skills.evaluation.mlip_metrics.implementation import (
    METRICS_FILE,
    MLIPMetricsSkill,
    _group_aggregates,
    _RecordErrors,
)
from mlip_research_agent.skills.evaluation.mlip_metrics.schema import (
    ArmComparisonSpec,
    LearningCurvePoint,
    LearningCurveSpec,
    MetricsArtifact,
    MLIPMetricsInput,
    MLIPMetricsOutput,
    PostRevealCalibrationSpec,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@dataclass
class Fixture:
    root: Path
    dataset: NormalizedDataset
    manifest: NormalizedManifest
    split: SplitManifest
    paths: dict[str, Path]
    params: MLIPMetricsInput

    def registry(self) -> ArtifactRegistry:
        registry = ArtifactRegistry(self.root)
        registry.register(self.paths["dataset"], "normalized_dataset", "qualify")
        registry.register(
            self.paths["dataset_manifest"], "normalized_dataset_manifest", "qualify"
        )
        registry.register(self.paths["split_manifest"], "split_manifest", "split")
        registry.register(self.paths["predictions"], "mlip_predictions", "predict")
        registry.register(self.paths["model_manifest"], "model_manifest", "predict")
        if "evaluation_authorization" in self.paths:
            registry.register(
                self.paths["evaluation_authorization"],
                "evaluation_authorization",
                "approve",
            )
        return registry

    def context(self, registry: ArtifactRegistry, step_id: str = "evaluate") -> SkillContext:
        return SkillContext(
            run_dir=self.root,
            step_id=step_id,
            seed=0,
            attempt=0,
            registry=registry,
        )


def make_fixture(root: Path, *, partition: str = "validation") -> Fixture:
    records = [
        LabeledConfiguration(
            config_id=f"metric-{index}",
            source_id=f"fixture.json[{index}]",
            top_group="fixture-group",
            group_id=f"group-{index}",
            split_unit_id=f"unit-{index}",
            symbols=["Cu"],
            positions=[[float(index), 0.0, 0.0]],
            cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            pbc=[True, True, True],
            energy_ev=float(index),
            forces_ev_per_a=[[0.0, 0.0, 0.0]],
            level_of_theory="synthetic",
            source_record_sha256=f"{index + 1:064x}",
        )
        for index in range(8)
    ]
    dataset = NormalizedDataset(
        dataset_id="metric_fixture",
        level_of_theory="synthetic",
        configurations=records,
    )
    dataset_path = root / "inputs" / "normalized_dataset.json"
    dataset_sha = dataset.save(dataset_path)
    manifest = NormalizedManifest.from_dataset(dataset, dataset_sha)
    manifest_path = root / "inputs" / "normalized_manifest.json"
    manifest.save(manifest_path)
    split = build_split_manifest(
        manifest,
        SplitSpec(
            initial_labeled=2,
            acquisition_pool=2,
            validation=2,
            frozen_test=2,
            seed=17,
            max_partition_size_deviation_fraction=0.0,
        ),
        qualified_manifest_path=manifest_path,
        qualified_manifest_reference="inputs/normalized_manifest.json",
    )
    split_path = root / "inputs" / "split_manifest.json"
    split.save(split_path)

    selected_ids = sorted(split.record_ids[partition])
    offsets = [1.0, 2.0]
    by_id = dataset.by_id()
    prediction_payload = {
        "schema_version": "2.0.0",
        "dataset_id": dataset.dataset_id,
        "dataset_content_sha256": dataset.content_hash(),
        "split_semantic_sha256": split.semantic_hash(),
        "model_id": "fixture-model",
        "model_manifest_artifact": "predict:model_manifest.json",
        "checkpoint_sha256": "c" * 64,
        "energy_unit": "eV",
        "force_unit": "eV/angstrom",
        "structure_set_sha256": "f" * 64,
        "predictions": [
            {
                "record_id": record_id,
                "energy_ev": by_id[record_id].energy_ev + offset,
                "forces_ev_per_a": [
                    [offset, 0.0, 0.0] if index == 0 else [0.0, offset, 0.0]
                ],
            }
            for index, (record_id, offset) in enumerate(
                zip(selected_ids, offsets, strict=True)
            )
        ],
    }
    predictions_path = root / "inputs" / "predictions.json"
    _write_json(predictions_path, prediction_payload)
    model_path = root / "inputs" / "model_manifest.json"
    _write_json(
        model_path,
        {
            "schema_version": "2.0.0",
            "model_id": "fixture-model",
            "checkpoint": {"sha256": "c" * 64},
            "predictions_artifact": "predict:predictions.json",
            "structure_set_sha256": "f" * 64,
        },
    )
    paths = {
        "dataset": dataset_path,
        "dataset_manifest": manifest_path,
        "split_manifest": split_path,
        "predictions": predictions_path,
        "model_manifest": model_path,
    }
    authorization_artifact: str | None = None
    if partition in {"frozen_test", "stress_test"}:
        authorization_path = root / "inputs" / "evaluation_authorization.json"
        _write_json(
            authorization_path,
            {
                "schema_version": "1.0.0",
                "authorization_id": f"authorize-{partition}",
                "partition": partition,
                "dataset_content_sha256": dataset.content_hash(),
                "split_semantic_sha256": split.semantic_hash(),
                "predictions_artifact": "predict:predictions.json",
                "model_manifest_artifact": "predict:model_manifest.json",
                "purpose": "final_aggregate_evaluation",
                "approved_by": "fixture-owner",
            },
        )
        paths["evaluation_authorization"] = authorization_path
        authorization_artifact = "approve:evaluation_authorization.json"
    params = MLIPMetricsInput(
        predictions_artifact="predict:predictions.json",
        dataset_artifact="qualify:normalized_dataset.json",
        dataset_manifest_artifact="qualify:normalized_manifest.json",
        split_manifest_artifact="split:split_manifest.json",
        model_manifest_artifact="predict:model_manifest.json",
        evaluation_authorization_artifact=authorization_artifact,
        partition=partition,  # type: ignore[arg-type]
        high_error_threshold_ev_per_a=1.5,
    )
    return Fixture(root, dataset, manifest, split, paths, params)


def _run(
    fixture: Fixture, *, step_id: str = "evaluate"
) -> tuple[MLIPMetricsOutput, dict[str, Any]]:
    registry = fixture.registry()
    output = MLIPMetricsSkill().run(fixture.params, fixture.context(registry, step_id))
    assert isinstance(output, MLIPMetricsOutput)
    return output, json.loads((fixture.root / output.metrics_path).read_text())


def _mutate_json(path: Path, mutator: Callable[[dict[str, Any]], None]) -> None:
    payload: dict[str, Any] = json.loads(path.read_text())
    mutator(payload)
    _write_json(path, payload)


def test_hand_calculated_golden_metrics_groups_and_claim_metadata(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    registry = fixture.registry()
    ctx = fixture.context(registry)
    output = MLIPMetricsSkill().run(fixture.params, ctx)
    assert isinstance(output, MLIPMetricsOutput)
    assert output.aggregate.energy_mae_ev_per_atom == pytest.approx(1.5)
    assert output.aggregate.force_component_mae_ev_per_a == pytest.approx(0.5)
    assert output.aggregate.force_component_rmse_ev_per_a == pytest.approx(
        math.sqrt(5.0 / 6.0)
    )
    assert output.aggregate.force_vector_error_p95_ev_per_a == pytest.approx(1.95)
    assert output.aggregate.high_error_structure_fraction == pytest.approx(0.5)
    assert output.aggregate.n_structures == 2
    assert output.aggregate.n_atoms == 2
    assert output.n_groups == 0

    artifact_path = tmp_path / output.metrics_path
    artifact = MetricsArtifact.model_validate_json(artifact_path.read_text())
    assert artifact.groups == []
    assert artifact.suppressed_group_structure_count == 2
    assert artifact.minimum_group_size == 3
    assert artifact.grouping_field == "top_group"
    assert len(artifact.evaluation_code_sha256) == 64

    claim_metadata = artifact.claim_reference_metadata
    assert claim_metadata.metric_artifact_references == [output.metrics_artifact]
    assert claim_metadata.dataset_manifest_reference == fixture.params.dataset_manifest_artifact
    assert claim_metadata.split_manifest_reference == fixture.params.split_manifest_artifact
    assert claim_metadata.model_manifest_reference == fixture.params.model_manifest_artifact
    assert claim_metadata.metric_value_paths["force_component_mae_ev_per_a"] == (
        "/aggregate/force_component_mae_ev_per_a"
    )
    assert set(claim_metadata.required_artifact_references) == {
        output.metrics_artifact,
        fixture.params.predictions_artifact,
        fixture.params.dataset_artifact,
        fixture.params.dataset_manifest_artifact,
        fixture.params.split_manifest_artifact,
        fixture.params.model_manifest_artifact,
    }
    assert registry.verify(output.metrics_artifact)
    assert registry.get(output.metrics_artifact).kind == "mlip_metrics"  # type: ignore[union-attr]
    assert ctx.claims == []


def test_output_contains_no_record_level_labels_errors_or_rankings(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path, partition="frozen_test")
    output, payload = _run(fixture)
    assert output.partition == "frozen_test"

    keys: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            keys.update(str(key) for key in value)
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(payload)
    assert keys.isdisjoint(
        {
            "record_id",
            "record_ids",
            "energy_ev",
            "forces_ev_per_a",
            "labels",
            "per_record_errors",
            "rankings",
        }
    )
    serialized = json.dumps(payload, sort_keys=True)
    assert all(record_id not in serialized for record_id in fixture.split.record_ids["frozen_test"])


def test_group_metrics_use_coarse_groups_and_suppress_small_cells() -> None:
    records = [
        _RecordErrors(
            group_id="bulk",
            n_atoms=1,
            energy_error_per_atom=float(index + 1),
            force_component_errors=(1.0, 0.0, 0.0),
            force_vector_errors=(1.0,),
            structure_force_vector_error=1.0,
        )
        for index in range(3)
    ]
    records.append(
        _RecordErrors(
            group_id="singleton-surface",
            n_atoms=1,
            energy_error_per_atom=99.0,
            force_component_errors=(99.0, 0.0, 0.0),
            force_vector_errors=(99.0,),
            structure_force_vector_error=99.0,
        )
    )
    groups, suppressed = _group_aggregates(records, threshold=2.0)
    assert [group.group_name for group in groups] == ["bulk"]
    assert groups[0].n_structures == 3
    assert groups[0].energy_mae_ev_per_atom == pytest.approx(2.0)
    assert suppressed == 1


def test_artifact_bytes_are_deterministic(tmp_path: Path) -> None:
    first = make_fixture(tmp_path / "first")
    second = make_fixture(tmp_path / "second")
    first_output, _ = _run(first)
    second_output, _ = _run(second)
    first_path = first.root / first_output.metrics_path
    second_path = second.root / second_output.metrics_path
    assert first_path.read_bytes() == second_path.read_bytes()
    assert sha256_file(first_path) == sha256_file(second_path)


def test_prediction_order_does_not_change_aggregate_bytes(tmp_path: Path) -> None:
    first = make_fixture(tmp_path / "first")
    second = make_fixture(tmp_path / "second")
    _mutate_json(
        second.paths["predictions"],
        lambda payload: payload["predictions"].reverse(),
    )
    _, first_payload = _run(first)
    _, second_payload = _run(second)
    assert first_payload["aggregate"] == second_payload["aggregate"]
    assert first_payload["groups"] == second_payload["groups"]


def test_skill_is_registered() -> None:
    assert get_skill("mlip_metrics") is MLIPMetricsSkill


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda payload: payload["predictions"].pop(), "coverage mismatch"),
        (
            lambda payload: payload["predictions"].append(
                {**payload["predictions"][0], "record_id": "metric-unknown"}
            ),
            "unknown record ids",
        ),
        (
            lambda payload: payload["predictions"].append(payload["predictions"][0]),
            "record ids must be unique",
        ),
        (
            lambda payload: payload["predictions"][0].update(
                {"forces_ev_per_a": [[1.0, 2.0]]}
            ),
            "n_atoms x 3",
        ),
        (
            lambda payload: payload.update({"force_unit": "hartree/bohr"}),
            "force_unit",
        ),
        (
            lambda payload: payload["predictions"][0].update({"energy_ev": float("nan")}),
            "finite number",
        ),
    ],
)
def test_prediction_schema_and_id_rejections(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    fixture = make_fixture(tmp_path)
    _mutate_json(fixture.paths["predictions"], mutator)
    with pytest.raises(SkillError, match=match):
        MLIPMetricsSkill().run(fixture.params, fixture.context(fixture.registry()))


def test_extra_record_from_another_partition_is_rejected(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    extra_id = fixture.split.record_ids[PartitionName.ACQUISITION_POOL.value][0]
    target = fixture.dataset.by_id()[extra_id]

    def add_extra(payload: dict[str, Any]) -> None:
        payload["predictions"].append(
            {
                "record_id": extra_id,
                "energy_ev": target.energy_ev,
                "forces_ev_per_a": target.forces_ev_per_a,
            }
        )

    _mutate_json(fixture.paths["predictions"], add_extra)
    with pytest.raises(SkillError, match="extra="):
        MLIPMetricsSkill().run(fixture.params, fixture.context(fixture.registry()))


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda payload: payload.update({"dataset_id": "other"}), "dataset id mismatch"),
        (
            lambda payload: payload.update({"dataset_content_sha256": "0" * 64}),
            "dataset content hash mismatch",
        ),
        (
            lambda payload: payload.update({"split_semantic_sha256": "0" * 64}),
            "split identity mismatch",
        ),
        (lambda payload: payload.update({"model_id": "other"}), "model id mismatch"),
        (
            lambda payload: payload.update({"checkpoint_sha256": "0" * 64}),
            "checkpoint hash mismatch",
        ),
        (
            lambda payload: payload.update({"model_manifest_artifact": "other:model.json"}),
            "model-manifest artifact mismatch",
        ),
    ],
)
def test_prediction_provenance_mismatches_rejected(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    fixture = make_fixture(tmp_path)
    _mutate_json(fixture.paths["predictions"], mutator)
    with pytest.raises(SkillError, match=match):
        MLIPMetricsSkill().run(fixture.params, fixture.context(fixture.registry()))


def test_model_manifest_prediction_mismatch_rejected(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    _mutate_json(
        fixture.paths["model_manifest"],
        lambda payload: payload.update({"predictions_artifact": "other:predictions.json"}),
    )
    with pytest.raises(SkillError, match="does not reference"):
        MLIPMetricsSkill().run(fixture.params, fixture.context(fixture.registry()))


def test_target_unit_mismatch_is_rejected_even_when_hashes_are_consistent(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path)
    payload = fixture.dataset.model_dump(mode="json")
    payload["units"]["energy"] = "hartree"
    payload["units"]["forces"] = "hartree/bohr"
    _write_json(fixture.paths["dataset"], payload)
    changed = NormalizedDataset.load(fixture.paths["dataset"])
    changed_manifest = NormalizedManifest.from_dataset(
        changed, sha256_file(fixture.paths["dataset"])
    )
    changed_manifest.save(fixture.paths["dataset_manifest"])
    split_payload = fixture.split.model_dump(mode="json")
    split_payload["qualified_dataset_content_sha256"] = changed.content_hash()
    split_payload["qualified_manifest_sha256"] = sha256_file(
        fixture.paths["dataset_manifest"]
    )
    changed_split = SplitManifest.model_validate(split_payload)
    changed_split.save(fixture.paths["split_manifest"])
    _mutate_json(
        fixture.paths["predictions"],
        lambda prediction: prediction.update(
            {
                "dataset_content_sha256": changed.content_hash(),
                "split_semantic_sha256": changed_split.semantic_hash(),
            }
        ),
    )
    with pytest.raises(SkillError, match="target dataset units"):
        MLIPMetricsSkill().run(fixture.params, fixture.context(fixture.registry()))


def test_dataset_manifest_and_target_nonfinite_rejected(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    payload = fixture.dataset.model_dump(mode="json")
    selected = fixture.split.record_ids[PartitionName.VALIDATION.value][0]
    for record in payload["configurations"]:
        if record["config_id"] == selected:
            record["energy_ev"] = float("nan")
    _write_json(fixture.paths["dataset"], payload)
    # Keep the manifest internally consistent so the explicit finite-label gate is reached.
    changed = NormalizedDataset.load(fixture.paths["dataset"])
    changed_manifest = NormalizedManifest.from_dataset(
        changed, sha256_file(fixture.paths["dataset"])
    )
    changed_manifest.save(fixture.paths["dataset_manifest"])
    split_payload = fixture.split.model_dump(mode="json")
    split_payload["qualified_dataset_content_sha256"] = changed.content_hash()
    split_payload["qualified_manifest_sha256"] = sha256_file(fixture.paths["dataset_manifest"])
    changed_split = SplitManifest.model_validate(split_payload)
    changed_split.save(fixture.paths["split_manifest"])
    _mutate_json(
        fixture.paths["predictions"],
        lambda prediction: prediction.update(
            {
                "dataset_content_sha256": changed.content_hash(),
                "split_semantic_sha256": changed_split.semantic_hash(),
            }
        ),
    )
    with pytest.raises(SkillError, match="non-finite values in target energy"):
        MLIPMetricsSkill().run(fixture.params, fixture.context(fixture.registry()))


def test_forged_record_group_mapping_rejected(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    payload = fixture.split.model_dump(mode="json")
    validation = PartitionName.VALIDATION.value
    frozen = PartitionName.FROZEN_TEST.value
    payload["record_ids"][validation][0], payload["record_ids"][frozen][0] = (
        payload["record_ids"][frozen][0],
        payload["record_ids"][validation][0],
    )
    for partition in (validation, frozen):
        payload["record_ids"][partition].sort()
        payload["partition_hashes"][partition] = _partition_hash(
            payload["record_ids"][partition], payload["source_group_ids"][partition]
        )
    SplitManifest.model_validate(payload).save(fixture.paths["split_manifest"])
    with pytest.raises(SkillError, match="record/group mapping mismatch"):
        MLIPMetricsSkill().run(fixture.params, fixture.context(fixture.registry()))


def test_frozen_test_access_is_selected_internally(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    frozen_params = fixture.params.model_copy(update={"partition": "frozen_test"})
    with pytest.raises(SkillError, match="coverage mismatch"):
        MLIPMetricsSkill().run(
            frozen_params,
            fixture.context(fixture.registry()),
        )
    assert not (fixture.context(fixture.registry()).step_dir / METRICS_FILE).exists()


def test_protected_authorization_is_exact_and_one_shot(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path, partition="frozen_test")
    registry = fixture.registry()
    first_ctx = fixture.context(registry, step_id="evaluate-first")
    MLIPMetricsSkill().run(fixture.params, first_ctx)
    with pytest.raises(SkillError, match="already been consumed"):
        MLIPMetricsSkill().run(
            fixture.params,
            fixture.context(registry, step_id="evaluate-second"),
        )

    _mutate_json(
        fixture.paths["evaluation_authorization"],
        lambda payload: payload.update({"predictions_artifact": "other:predictions.json"}),
    )
    with pytest.raises(SkillError, match="predictions mismatch"):
        MLIPMetricsSkill().run(
            fixture.params,
            fixture.context(fixture.registry(), step_id="evaluate-forged"),
        )


def test_empty_stress_partition_rejected(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    stress = fixture.params.model_copy(update={"partition": "stress_test"})
    with pytest.raises(SkillError, match="is empty"):
        MLIPMetricsSkill().run(stress, fixture.context(fixture.registry()))


def test_unregistered_wrong_kind_and_tampered_artifacts_rejected(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    registry = fixture.registry()
    fixture.paths["predictions"].write_text("{}")
    with pytest.raises(SkillError, match="integrity verification failed"):
        MLIPMetricsSkill().run(fixture.params, fixture.context(registry))

    missing = fixture.params.model_copy(update={"predictions_artifact": "missing:artifact"})
    with pytest.raises(SkillError, match="not registered"):
        MLIPMetricsSkill().run(missing, fixture.context(fixture.registry()))

    wrong_registry = ArtifactRegistry(tmp_path)
    wrong_registry.register(fixture.paths["predictions"], "wrong", "predict")
    wrong_registry.register(fixture.paths["dataset"], "normalized_dataset", "qualify")
    wrong_registry.register(
        fixture.paths["dataset_manifest"], "normalized_dataset_manifest", "qualify"
    )
    wrong_registry.register(fixture.paths["split_manifest"], "split_manifest", "split")
    wrong_registry.register(fixture.paths["model_manifest"], "model_manifest", "predict")
    with pytest.raises(SkillError, match="expected 'mlip_predictions'"):
        MLIPMetricsSkill().run(fixture.params, fixture.context(wrong_registry))


def test_non_evaluation_partition_rejected_by_schema() -> None:
    with pytest.raises(ValidationError, match=r"validation|frozen_test|stress_test"):
        MLIPMetricsInput(
            predictions_artifact="p",
            dataset_artifact="d",
            dataset_manifest_artifact="dm",
            split_manifest_artifact="s",
            model_manifest_artifact="m",
            partition="acquisition_pool",  # type: ignore[arg-type]
            high_error_threshold_ev_per_a=1.0,
        )


def test_learning_curve_calibration_and_arm_comparison_interfaces_are_typed() -> None:
    curve = LearningCurveSpec(
        arm="hybrid",
        points=[
            LearningCurvePoint(
                labels_available=32, metrics_artifact="round0:metrics.json"
            ),
            LearningCurvePoint(
                labels_available=40, metrics_artifact="round1:metrics.json"
            ),
        ],
    )
    assert [point.labels_available for point in curve.points] == [32, 40]
    with pytest.raises(ValidationError, match="strictly increasing"):
        LearningCurveSpec.model_validate(
            {
                "arm": "hybrid",
                "points": [
                    {"labels_available": 40, "metrics_artifact": "round1:metrics.json"},
                    {"labels_available": 32, "metrics_artifact": "round0:metrics.json"},
                ],
            }
        )
    calibration = PostRevealCalibrationSpec(
        selection_artifact="select:selection.json",
        revealed_label_batch_artifact="oracle:labels.json",
        uncertainty_scores_artifact="uq:scores.json",
    )
    assert calibration.metric == "spearman_disagreement_vs_error"
    comparison = ArmComparisonSpec(
        left_arm="random",
        right_arm="hybrid",
        left_metrics_artifact="random:metrics.json",
        right_metrics_artifact="hybrid:metrics.json",
        equal_label_budget=32,
    )
    assert comparison.equal_label_budget == 32
