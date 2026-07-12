"""Extract fixed-foundation MACE descriptors without exposing oracle labels."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from typing import Any

import numpy as np
from ase import Atoms
from pydantic import BaseModel

from mlip_research_agent.data.oracle import AcquisitionView, assert_no_hidden_labels
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.active_learning.diversity_select.schema import DescriptorSet
from mlip_research_agent.skills.active_learning.mace_descriptors.schema import (
    MACEDescriptorInput,
    MACEDescriptorOutput,
)
from mlip_research_agent.skills.active_learning.mace_descriptors.validators import (
    validate_num_layers,
    validate_view,
)
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.mlip.mace_inference.checkpoint import (
    MACECheckpointManifest,
    resolve_checkpoint,
)
from mlip_research_agent.skills.mlip.mace_inference.implementation import _make_calculator
from mlip_research_agent.skills.mlip.mace_inference.validators import local_file

DESCRIPTOR_FILENAME = "mace_descriptors.json"


def _invalid(message: str) -> SkillError:
    return SkillError(
        message,
        failure_class=FailureClass.VALIDATION_ERROR,
        severity=Severity.HIGH,
        retryable=False,
    )


def standardize_descriptors(
    raw: dict[str, np.ndarray], *, min_feature_std: float, l2_normalize: bool
) -> tuple[dict[str, list[float]], int, int, list[int]]:
    """Pool-only population standardization followed by optional row L2 scaling."""
    ids = sorted(raw)
    if not ids:
        raise ValueError("descriptor pool is empty")
    matrix = np.stack([np.asarray(raw[c], dtype=float) for c in ids])
    if matrix.ndim != 2 or matrix.shape[1] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("raw MACE descriptors must be a finite two-dimensional matrix")
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    retained = np.flatnonzero(stds > min_feature_std)
    if retained.size == 0:
        raise ValueError("all MACE descriptor dimensions are constant in the pool")
    normalized = (matrix[:, retained] - means[retained]) / stds[retained]
    if l2_normalize:
        norms = np.linalg.norm(normalized, axis=1)
        nonzero = norms > min_feature_std
        normalized[nonzero] = normalized[nonzero] / norms[nonzero, None]
    vectors = {
        candidate_id: [float(value) for value in row]
        for candidate_id, row in zip(ids, normalized, strict=True)
    }
    return vectors, int(matrix.shape[1]), int(normalized.shape[1]), [int(i) for i in retained]


@register_skill
class MACEDescriptorSkill(Skill):
    name = "mace_descriptors"
    input_model = MACEDescriptorInput
    output_model = MACEDescriptorOutput
    cost_class = "expensive"
    permission_level = "human_approval"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, MACEDescriptorInput)
        artifact = ctx.registry.get(params.acquisition_view_artifact)
        if artifact is None or artifact.kind != "acquisition_view":
            raise _invalid("MACE descriptors require a registered acquisition_view")
        if not ctx.registry.verify(artifact.artifact_id):
            raise _invalid("acquisition view failed its registered hash check")
        view_path = (ctx.run_dir / artifact.relative_path).resolve()
        try:
            view_path.relative_to(ctx.run_dir.resolve())
            view = AcquisitionView.load(view_path)
        except (OSError, ValueError) as exc:
            raise _invalid(f"invalid acquisition view: {exc}") from exc
        try:
            assert_no_hidden_labels(view.model_dump(mode="json"))
        except ValueError as exc:
            raise _invalid(f"acquisition view is not label-free: {exc}") from exc

        manifest_path = local_file(params.checkpoint_manifest_path, "checkpoint manifest")
        checkpoint_path = local_file(params.checkpoint_path, "MACE checkpoint")
        try:
            manifest = MACECheckpointManifest.load(manifest_path)
            resolved = resolve_checkpoint(manifest, checkpoint_path)
        except (OSError, ValueError) as exc:
            raise _invalid(f"MACE descriptor checkpoint validation failed: {exc}") from exc
        try:
            validate_view(view, manifest.supported_species)
        except ValueError as exc:
            raise _invalid(str(exc)) from exc
        if importlib.metadata.version("mace-torch") != manifest.mace_torch_version:
            raise _invalid("MACE descriptor environment does not match checkpoint manifest")
        if params.device == "cuda":
            torch = importlib.import_module("torch")
            if not bool(torch.cuda.is_available()):
                raise _invalid("CUDA descriptors requested without a visible CUDA device")

        try:
            calculator: Any = _make_calculator(
                checkpoint_path, params.device, params.default_dtype
            )
            num_interactions = int(calculator.models[0].num_interactions)
            validate_num_layers(params.num_layers, num_interactions)
        except Exception as exc:
            raise _invalid(f"MACE descriptor calculator initialization failed: {exc}") from exc
        raw: dict[str, np.ndarray] = {}
        for candidate in sorted(view.candidates, key=lambda item: item.record_id):
            atoms = Atoms(
                symbols=candidate.symbols,
                positions=np.asarray(candidate.positions),
                cell=np.asarray(candidate.cell),
                pbc=candidate.pbc,
            )
            try:
                atom_descriptors = np.asarray(
                    calculator.get_descriptors(
                        atoms,
                        invariants_only=True,
                        num_layers=params.num_layers,
                    ),
                    dtype=float,
                )
            except Exception as exc:
                raise _invalid(
                    f"candidate {candidate.record_id}: MACE descriptor extraction failed: {exc}"
                ) from exc
            if (
                atom_descriptors.ndim != 2
                or atom_descriptors.shape[0] != candidate.metadata.n_atoms
            ):
                raise _invalid(
                    f"candidate {candidate.record_id}: unexpected MACE descriptor shape"
                )
            raw[candidate.record_id] = atom_descriptors.mean(axis=0)
        try:
            vectors, raw_dimension, dimension, retained = standardize_descriptors(
                raw,
                min_feature_std=params.min_feature_std,
                l2_normalize=params.l2_normalize,
            )
        except ValueError as exc:
            raise _invalid(str(exc)) from exc
        descriptor_set = DescriptorSet(
            source="mace_descriptor_adapter",
            dimension=dimension,
            vectors=vectors,
        )
        payload = {
            "schema_version": "1.0.0",
            "dataset_id": view.dataset_id,
            "split_semantic_sha256": view.split_semantic_sha256,
            "acquisition_view_artifact": artifact.artifact_id,
            "acquisition_view_sha256": artifact.sha256,
            "checkpoint_sha256": resolved.sha256,
            "checkpoint_manifest_sha256": resolved.manifest_sha256,
            "pooling": params.pooling,
            "num_layers": params.num_layers,
            "invariants_only": True,
            "normalization": {
                "scope": "current_unlabeled_pool_only",
                "population_std_ddof": 0,
                "min_feature_std": params.min_feature_std,
                "retained_feature_indices": retained,
                "l2_normalize": params.l2_normalize,
            },
            "raw_dimension": raw_dimension,
            "descriptor_set": descriptor_set.model_dump(mode="json"),
        }
        assert_no_hidden_labels(payload)
        path = ctx.step_dir / DESCRIPTOR_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output_artifact = ctx.registry.register(
            path, kind="mace_descriptors", step_id=ctx.step_id
        )
        return MACEDescriptorOutput(
            descriptor_artifact=output_artifact.artifact_id,
            descriptor_path=output_artifact.relative_path,
            n_candidates=len(vectors),
            raw_dimension=raw_dimension,
            dimension=dimension,
            checkpoint_sha256=resolved.sha256,
        )
