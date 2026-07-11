"""Independent aggregate evaluator for registered MLIP prediction artifacts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from mlip_research_agent.artifacts.registry import Artifact, sha256_file
from mlip_research_agent.data.manifests import ENERGY_UNIT, FORCE_UNIT, NormalizedDataset
from mlip_research_agent.data.registry import NormalizedManifest
from mlip_research_agent.data.split import (
    GroupingField,
    SplitManifest,
    derive_structural_fingerprint_clusters,
)
from mlip_research_agent.schemas.predictions import PredictionBatch
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.evaluation.mlip_metrics.schema import (
    AggregateMetrics,
    ClaimReferenceMetadata,
    EvaluationAuthorization,
    EvaluationModelManifest,
    GroupMetrics,
    InputArtifactReference,
    MetricsArtifact,
    MLIPMetricsInput,
    MLIPMetricsOutput,
)
from mlip_research_agent.skills.evaluation.mlip_metrics.validators import (
    registered_artifact,
    require_finite,
    validation_error,
)

METRICS_FILE = "mlip_metrics.json"
MINIMUM_GROUP_SIZE = 3

_ARTIFACT_KINDS = {
    "predictions": "mlip_predictions",
    "dataset": "normalized_dataset",
    "dataset_manifest": "normalized_dataset_manifest",
    "split_manifest": "split_manifest",
    "model_manifest": "model_manifest",
}


@dataclass(frozen=True)
class _RecordErrors:
    group_id: str
    n_atoms: int
    energy_error_per_atom: float
    force_component_errors: tuple[float, ...]
    force_vector_errors: tuple[float, ...]
    structure_force_vector_error: float


def _linear_percentile(values: list[float], quantile: float) -> float:
    """Type-7 linear percentile, equivalent to NumPy's default method."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _aggregate(records: list[_RecordErrors], threshold: float) -> AggregateMetrics:
    if not records:
        raise ValueError("cannot aggregate an empty evaluation partition")
    component_errors = [
        error for record in records for error in record.force_component_errors
    ]
    vector_errors = [error for record in records for error in record.force_vector_errors]
    return AggregateMetrics(
        n_structures=len(records),
        n_atoms=sum(record.n_atoms for record in records),
        energy_mae_ev_per_atom=(
            sum(record.energy_error_per_atom for record in records) / len(records)
        ),
        force_component_mae_ev_per_a=sum(component_errors) / len(component_errors),
        force_component_rmse_ev_per_a=math.sqrt(
            sum(error * error for error in component_errors) / len(component_errors)
        ),
        force_vector_error_p95_ev_per_a=_linear_percentile(vector_errors, 0.95),
        high_error_structure_fraction=(
            sum(record.structure_force_vector_error > threshold for record in records)
            / len(records)
        ),
    )


def _group_aggregates(
    records: list[_RecordErrors],
    threshold: float,
) -> tuple[list[GroupMetrics], int]:
    by_group: dict[str, list[_RecordErrors]] = defaultdict(list)
    for record in records:
        by_group[record.group_id].append(record)
    groups = [
        GroupMetrics(
            group_name=group_name,
            **_aggregate(group_records, threshold).model_dump(),
        )
        for group_name, group_records in sorted(by_group.items())
        if len(group_records) >= MINIMUM_GROUP_SIZE
    ]
    suppressed = sum(
        len(group_records)
        for group_records in by_group.values()
        if len(group_records) < MINIMUM_GROUP_SIZE
    )
    return groups, suppressed


def _artifact_reference(artifact: Artifact) -> InputArtifactReference:
    return InputArtifactReference(
        artifact_id=artifact.artifact_id,
        sha256=artifact.sha256,
        kind=artifact.kind,
    )


@register_skill
class MLIPMetricsSkill(Skill):
    """Evaluate one protected partition and emit aggregates only."""

    name = "mlip_metrics"
    input_model = MLIPMetricsInput
    output_model = MLIPMetricsOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, MLIPMetricsInput)
        requested_ids = {
            "predictions": params.predictions_artifact,
            "dataset": params.dataset_artifact,
            "dataset_manifest": params.dataset_manifest_artifact,
            "split_manifest": params.split_manifest_artifact,
            "model_manifest": params.model_manifest_artifact,
        }
        if params.evaluation_authorization_artifact is not None:
            requested_ids["evaluation_authorization"] = (
                params.evaluation_authorization_artifact
            )
        artifacts: dict[str, Artifact] = {}
        paths: dict[str, Path] = {}
        for role, artifact_id in requested_ids.items():
            artifact, path = registered_artifact(
                ctx.registry,
                artifact_id,
                expected_kind=(
                    "evaluation_authorization"
                    if role == "evaluation_authorization"
                    else _ARTIFACT_KINDS[role]
                ),
            )
            artifacts[role] = artifact
            paths[role] = path

        try:
            predictions = PredictionBatch.model_validate_json(
                paths["predictions"].read_text()
            )
            dataset = NormalizedDataset.load(paths["dataset"])
            dataset_manifest = NormalizedManifest.load(paths["dataset_manifest"])
            split = SplitManifest.load(paths["split_manifest"])
            model_manifest = EvaluationModelManifest.model_validate_json(
                paths["model_manifest"].read_text()
            )
            self._validate_provenance(
                predictions,
                dataset,
                dataset_manifest,
                split,
                model_manifest,
                artifacts,
                params,
            )
            self._validate_authorization(
                params,
                predictions,
                dataset,
                split,
                artifacts,
                paths,
                ctx,
            )
            record_errors = self._record_errors(predictions, dataset, split, params)
            aggregate = _aggregate(record_errors, params.high_error_threshold_ev_per_a)
            groups, suppressed_group_structure_count = _group_aggregates(
                record_errors,
                params.high_error_threshold_ev_per_a,
            )
        except (OSError, ValueError) as exc:
            raise validation_error(f"MLIP metric evaluation rejected its inputs: {exc}") from exc

        expected_metric_id = f"{ctx.step_id}:{METRICS_FILE}"
        input_references = {
            role: _artifact_reference(artifacts[role])
            for role in (
                "predictions",
                "dataset",
                "dataset_manifest",
                "split_manifest",
                "model_manifest",
            )
        }
        required_claim_refs = [
            expected_metric_id,
            params.predictions_artifact,
            params.dataset_artifact,
            params.dataset_manifest_artifact,
            params.split_manifest_artifact,
            params.model_manifest_artifact,
        ]
        if params.evaluation_authorization_artifact is not None:
            required_claim_refs.append(params.evaluation_authorization_artifact)
        metric_value_paths = {
            name: f"/aggregate/{name}"
            for name in (
                "energy_mae_ev_per_atom",
                "force_component_mae_ev_per_a",
                "force_component_rmse_ev_per_a",
                "force_vector_error_p95_ev_per_a",
                "high_error_structure_fraction",
            )
        }
        metric_payload = MetricsArtifact(
            evaluation_code_sha256=sha256_file(Path(__file__)),
            partition=params.partition,
            dataset_id=dataset.dataset_id,
            dataset_content_sha256=dataset.content_hash(),
            split_semantic_sha256=split.semantic_hash(),
            model_id=model_manifest.model_id,
            checkpoint_sha256=model_manifest.checkpoint.sha256,
            units={
                "energy_mae": "eV/atom",
                "force_component_mae": FORCE_UNIT,
                "force_component_rmse": FORCE_UNIT,
                "force_vector_error_p95": FORCE_UNIT,
                "high_error_threshold": FORCE_UNIT,
                "prediction_energy": ENERGY_UNIT,
            },
            high_error_threshold_ev_per_a=params.high_error_threshold_ev_per_a,
            metric_definitions={
                "energy_mae_ev_per_atom": (
                    "mean over structures of absolute total-energy error divided by n_atoms"
                ),
                "force_component_mae_ev_per_a": (
                    "mean absolute error over all Cartesian force components"
                ),
                "force_component_rmse_ev_per_a": (
                    "root mean square error over all Cartesian force components"
                ),
                "force_vector_error_p95_ev_per_a": (
                    "type-7 95th percentile of per-atom Euclidean force-vector errors"
                ),
                "high_error_structure_fraction": (
                    "fraction whose mean per-atom force-vector error is strictly above threshold"
                ),
            },
            input_artifacts=input_references,
            aggregate=aggregate,
            groups=groups,
            minimum_group_size=MINIMUM_GROUP_SIZE,
            suppressed_group_structure_count=suppressed_group_structure_count,
            evaluation_authorization_reference=(
                params.evaluation_authorization_artifact
            ),
            claim_reference_metadata=ClaimReferenceMetadata(
                metric_artifact_references=[expected_metric_id],
                dataset_manifest_reference=params.dataset_manifest_artifact,
                split_manifest_reference=params.split_manifest_artifact,
                model_manifest_reference=params.model_manifest_artifact,
                required_artifact_references=required_claim_refs,
                metric_value_paths=metric_value_paths,
                evaluation_partition=params.partition,
            ),
        )
        output_path = ctx.step_dir / METRICS_FILE
        output_path.write_text(
            json.dumps(metric_payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        metric_artifact = ctx.registry.register(
            output_path,
            kind="mlip_metrics",
            step_id=ctx.step_id,
        )
        if metric_artifact.artifact_id != expected_metric_id:
            raise validation_error("metric artifact identity differs from claim-reference metadata")
        return MLIPMetricsOutput(
            metrics_artifact=metric_artifact.artifact_id,
            metrics_path=metric_artifact.relative_path,
            partition=params.partition,
            aggregate=aggregate,
            n_groups=len(groups),
        )

    @staticmethod
    def _validate_provenance(
        predictions: PredictionBatch,
        dataset: NormalizedDataset,
        dataset_manifest: NormalizedManifest,
        split: SplitManifest,
        model_manifest: EvaluationModelManifest,
        artifacts: dict[str, Artifact],
        params: MLIPMetricsInput,
    ) -> None:
        if dataset_manifest.dataset_file_sha256 != artifacts["dataset"].sha256:
            raise ValueError("dataset bytes do not match the dataset manifest")
        derived_manifest = NormalizedManifest.from_dataset(
            dataset, artifacts["dataset"].sha256
        )
        if derived_manifest != dataset_manifest:
            raise ValueError("dataset content/lineage does not match its manifest")
        if split.qualified_manifest_sha256 != artifacts["dataset_manifest"].sha256:
            raise ValueError("split provenance does not match the dataset manifest artifact")
        structural_fingerprints = (
            derive_structural_fingerprint_clusters(dataset)
            if split.grouping_fields_used == [GroupingField.STRUCTURAL_FINGERPRINT.value]
            else None
        )
        split.validate_against(
            dataset_manifest,
            structural_fingerprints=structural_fingerprints,
        )
        if dataset.units.energy != ENERGY_UNIT or dataset.units.forces != FORCE_UNIT:
            raise ValueError(
                "target dataset units must be exactly eV and eV/angstrom"
            )
        if predictions.dataset_id != dataset.dataset_id:
            raise ValueError("prediction dataset id mismatch")
        if predictions.dataset_content_sha256 != dataset.content_hash():
            raise ValueError("prediction dataset content hash mismatch")
        if predictions.split_semantic_sha256 != split.semantic_hash():
            raise ValueError("prediction split identity mismatch")
        if predictions.model_manifest_artifact != params.model_manifest_artifact:
            raise ValueError("prediction model-manifest artifact mismatch")
        if model_manifest.predictions_artifact != params.predictions_artifact:
            raise ValueError("model manifest does not reference the prediction artifact")
        if predictions.model_id != model_manifest.model_id:
            raise ValueError("prediction model id mismatch")
        if predictions.checkpoint_sha256 != model_manifest.checkpoint.sha256:
            raise ValueError("prediction checkpoint hash mismatch")
        if predictions.structure_set_sha256 != model_manifest.structure_set_sha256:
            raise ValueError("prediction/model structure-set hash mismatch")

    @staticmethod
    def _validate_authorization(
        params: MLIPMetricsInput,
        predictions: PredictionBatch,
        dataset: NormalizedDataset,
        split: SplitManifest,
        artifacts: dict[str, Artifact],
        paths: dict[str, Path],
        ctx: SkillContext,
    ) -> None:
        if params.evaluation_authorization_artifact is None:
            return
        try:
            authorization = EvaluationAuthorization.model_validate_json(
                paths["evaluation_authorization"].read_text()
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"protected evaluation authorization is invalid: {exc}") from exc
        if authorization.partition != params.partition:
            raise ValueError("evaluation authorization partition mismatch")
        if authorization.dataset_content_sha256 != dataset.content_hash():
            raise ValueError("evaluation authorization dataset mismatch")
        if authorization.split_semantic_sha256 != split.semantic_hash():
            raise ValueError("evaluation authorization split mismatch")
        if authorization.predictions_artifact != params.predictions_artifact:
            raise ValueError("evaluation authorization predictions mismatch")
        if authorization.model_manifest_artifact != params.model_manifest_artifact:
            raise ValueError("evaluation authorization model mismatch")
        for artifact in ctx.registry.all():
            if artifact.kind != "mlip_metrics" or not ctx.registry.verify(artifact.artifact_id):
                continue
            payload = json.loads((ctx.run_dir / artifact.relative_path).read_text())
            if (
                payload.get("evaluation_authorization_reference")
                == params.evaluation_authorization_artifact
            ):
                raise ValueError(
                    "evaluation authorization has already been consumed in this run"
                )
        if artifacts["evaluation_authorization"].artifact_id != (
            params.evaluation_authorization_artifact
        ):
            raise ValueError("evaluation authorization artifact identity mismatch")

    @staticmethod
    def _record_errors(
        predictions: PredictionBatch,
        dataset: NormalizedDataset,
        split: SplitManifest,
        params: MLIPMetricsInput,
    ) -> list[_RecordErrors]:
        expected_ids = set(split.record_ids[params.partition])
        if not expected_ids:
            raise ValueError(f"evaluation partition {params.partition!r} is empty")
        prediction_by_id = {record.record_id: record for record in predictions.predictions}
        predicted_ids = set(prediction_by_id)
        dataset_by_id = dataset.by_id()
        unknown = sorted(predicted_ids - set(dataset_by_id))
        missing = sorted(expected_ids - predicted_ids)
        extra = sorted(predicted_ids - expected_ids)
        if unknown:
            raise ValueError(f"prediction contains unknown record ids: {unknown[:5]}")
        if missing or extra:
            raise ValueError(
                f"prediction partition coverage mismatch; missing={missing[:5]}, extra={extra[:5]}"
            )

        evaluated: list[_RecordErrors] = []
        for record_id in sorted(expected_ids):
            target = dataset_by_id[record_id]
            prediction = prediction_by_id[record_id]
            if len(prediction.forces_ev_per_a) != target.n_atoms:
                raise ValueError(
                    f"{record_id}: predicted force rows do not match target atom count"
                )
            require_finite([target.energy_ev], f"target energy for {record_id}")
            require_finite(
                [value for row in target.forces_ev_per_a for value in row],
                f"target forces for {record_id}",
            )
            component_errors: list[float] = []
            vector_errors: list[float] = []
            for predicted_force, target_force in zip(
                prediction.forces_ev_per_a,
                target.forces_ev_per_a,
                strict=True,
            ):
                differences = [
                    predicted - reference
                    for predicted, reference in zip(
                        predicted_force, target_force, strict=True
                    )
                ]
                component_errors.extend(abs(value) for value in differences)
                vector_errors.append(math.sqrt(sum(value * value for value in differences)))
            evaluated.append(
                _RecordErrors(
                    group_id=target.top_group,
                    n_atoms=target.n_atoms,
                    energy_error_per_atom=(
                        abs(prediction.energy_ev - target.energy_ev) / target.n_atoms
                    ),
                    force_component_errors=tuple(component_errors),
                    force_vector_errors=tuple(vector_errors),
                    structure_force_vector_error=sum(vector_errors) / len(vector_errors),
                )
            )
        return evaluated
