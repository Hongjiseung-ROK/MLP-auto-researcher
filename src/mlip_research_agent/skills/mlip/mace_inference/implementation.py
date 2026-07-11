"""Pinned local-path MACE inference; convenience aliases/downloads are forbidden."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.atomistics.structures_io import (
    StructureSet,
    record_to_atoms,
)
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.mlip.mace_inference.checkpoint import (
    CheckpointResolutionError,
    MACECheckpointManifest,
    resolve_checkpoint,
)
from mlip_research_agent.skills.mlip.mace_inference.schema import (
    MACEInferenceInput,
    MACEInferenceOutput,
)
from mlip_research_agent.skills.mlip.mace_inference.validators import (
    local_file,
    require_finite,
    require_mace_installation,
    run_relative_file,
    validate_structures,
)

PREDICTIONS_FILE = "mace_predictions.json"
MODEL_MANIFEST_FILE = "model_manifest.json"


def _make_calculator(checkpoint_path: Path, device: str, default_dtype: str) -> Any:
    calculators = importlib.import_module("mace.calculators")
    factory: Any = calculators.mace_mp
    # Passing the verified local path prevents mace_mp from resolving an alias
    # or downloading a mutable default checkpoint.
    return factory(
        model=str(checkpoint_path),
        device=device,
        default_dtype=default_dtype,
        dispersion=False,
    )


@register_skill
class MACEInferenceSkill(Skill):
    name = "mace_inference"
    input_model = MACEInferenceInput
    output_model = MACEInferenceOutput
    cost_class = "cheap"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, MACEInferenceInput)
        require_mace_installation()
        structures_path = run_relative_file(
            ctx.run_dir, params.structures_path, "structure set"
        )
        manifest_path = local_file(params.checkpoint_manifest_path, "checkpoint manifest")
        checkpoint_path = local_file(params.checkpoint_path, "MACE checkpoint")
        try:
            manifest = MACECheckpointManifest.load(manifest_path)
            resolved = resolve_checkpoint(manifest, checkpoint_path)
        except (OSError, ValueError, CheckpointResolutionError) as exc:
            raise SkillError(
                f"MACE checkpoint validation failed: {exc}",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.CRITICAL,
                retryable=False,
            ) from exc
        installed_mace = importlib.metadata.version("mace-torch")
        if installed_mace != manifest.mace_torch_version:
            raise SkillError(
                f"mace-torch version mismatch: manifest pins {manifest.mace_torch_version}, "
                f"environment has {installed_mace}",
                failure_class=FailureClass.TOOL_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )
        if params.device == "cuda":
            torch = importlib.import_module("torch")
            if not bool(torch.cuda.is_available()):
                raise SkillError(
                    "CUDA inference requested but torch reports no CUDA device",
                    failure_class=FailureClass.RESOURCE_EXHAUSTED,
                    severity=Severity.HIGH,
                    retryable=False,
                )

        structures = StructureSet.load(structures_path)
        validate_structures(structures, set(manifest.supported_species))
        calculator = _make_calculator(checkpoint_path, params.device, params.default_dtype)
        predictions: list[dict[str, Any]] = []
        for record in structures.systems:
            atoms = record_to_atoms(record)
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
            forces = [[float(value) for value in row] for row in atoms.get_forces()]
            stress = [float(value) for value in atoms.get_stress()]
            require_finite([energy], "energy")
            require_finite([value for row in forces for value in row], "forces")
            require_finite(stress, "stress")
            predictions.append(
                {
                    "structure_index": record.index,
                    "energy_ev": energy,
                    "forces_ev_per_a": forces,
                    "stress_ev_per_a3_voigt6": stress,
                }
            )

        predictions_path = ctx.step_dir / PREDICTIONS_FILE
        predictions_payload = {
            "schema_version": "2.0.0",
            "model_id": manifest.model_id,
            "checkpoint_sha256": resolved.sha256,
            "device": params.device,
            "default_dtype": params.default_dtype,
            "structure_set_sha256": sha256_file(structures_path),
            "predictions": predictions,
        }
        predictions_path.write_text(
            json.dumps(predictions_payload, indent=2, sort_keys=True) + "\n"
        )
        predictions_artifact = ctx.registry.register(
            predictions_path, kind="mlip_predictions", step_id=ctx.step_id
        )

        model_manifest_path = ctx.step_dir / MODEL_MANIFEST_FILE
        model_manifest_payload = {
            "schema_version": "2.0.0",
            "model_id": manifest.model_id,
            "family": manifest.family,
            "scientific_status": manifest.scientific_status,
            "checkpoint": resolved.model_dump(mode="json"),
            "checkpoint_source_url": manifest.source_url,
            "checkpoint_license_spdx": manifest.license_spdx,
            "training_data_statement": manifest.training_data_statement,
            "mace_torch_version": installed_mace,
            "torch_version": importlib.metadata.version("torch"),
            "device": params.device,
            "default_dtype": params.default_dtype,
            "structures_path": params.structures_path,
            "structure_set_sha256": sha256_file(structures_path),
            "predictions_artifact": predictions_artifact.artifact_id,
        }
        model_manifest_path.write_text(
            json.dumps(model_manifest_payload, indent=2, sort_keys=True) + "\n"
        )
        model_manifest_artifact = ctx.registry.register(
            model_manifest_path, kind="model_manifest", step_id=ctx.step_id
        )
        return MACEInferenceOutput(
            predictions_artifact=predictions_artifact.artifact_id,
            predictions_path=predictions_artifact.relative_path,
            model_manifest_artifact=model_manifest_artifact.artifact_id,
            model_manifest_path=model_manifest_artifact.relative_path,
            model_id=manifest.model_id,
            checkpoint_sha256=resolved.sha256,
            n_structures=len(predictions),
            device=params.device,
            default_dtype=params.default_dtype,
        )
