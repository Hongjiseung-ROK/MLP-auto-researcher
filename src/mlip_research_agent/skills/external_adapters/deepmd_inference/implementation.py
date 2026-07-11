"""Pinned DeePMD model inference; fail-closed without deepmd-kit installed.

deepmd-kit is an MLIP backend extra with its own framework pins — never
co-install it with the `mace` extra in one environment (CLAUDE.md rule).
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill
from mlip_research_agent.skills.external_adapters.common import (
    adapter_error,
    load_structure_set,
    require_module,
    upstream_provenance,
    write_registered_json,
)
from mlip_research_agent.skills.external_adapters.deepmd_inference.schema import (
    DeepMDInferenceInput,
    DeepMDInferenceOutput,
)
from mlip_research_agent.skills.external_adapters.deepmd_inference.validators import (
    finite_structure_records,
    resolve_pinned_model,
)

UPSTREAM_PATH = "machine-learning-potentials/deepmd-python-inference"
PREDICTIONS_FILE = "deepmd_predictions.json"


@register_skill
class ExternalDeepMDInferenceSkill(Skill):
    """Predict energies/forces for a StructureSet with a pinned DeePMD model."""

    name = "external_deepmd_inference"
    input_model = DeepMDInferenceInput
    output_model = DeepMDInferenceOutput
    cost_class = "expensive"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, DeepMDInferenceInput)
        structures = load_structure_set(ctx, params.structures_path)
        finite_structure_records(structures.systems)
        # Hash gate first: nothing is deserialized on a mismatch, and the
        # dependency check happens only for a verified model file.
        model_path = resolve_pinned_model(params.model_path, params.model_sha256)
        infer_module = require_module("deepmd.infer", package="deepmd-kit", extra="deepmd")
        import numpy as np

        potential = infer_module.DeepPot(str(model_path))
        type_map: list[str] = [str(name) for name in potential.get_type_map()]
        predictions = []
        for record in structures.systems:
            unsupported = sorted(set(record.symbols) - set(type_map))
            if unsupported:
                raise adapter_error(
                    f"structure {record.index} has species outside the model type map: "
                    f"{unsupported}"
                )
            coords = np.array(record.positions, dtype=float).reshape(1, -1)
            cells = np.array(record.cell, dtype=float).reshape(1, 9)
            atom_types = [type_map.index(symbol) for symbol in record.symbols]
            energy, forces, _virial = potential.eval(coords, cells, atom_types)
            energy_ev = float(np.asarray(energy).reshape(-1)[0])
            forces_ev_per_a = np.asarray(forces, dtype=float).reshape(
                len(record.symbols), 3
            )
            if not math.isfinite(energy_ev) or not np.isfinite(forces_ev_per_a).all():
                raise adapter_error(f"non-finite DeePMD prediction for structure {record.index}")
            predictions.append(
                {
                    "index": record.index,
                    "energy_ev": energy_ev,
                    "forces_ev_per_a": forces_ev_per_a.tolist(),
                }
            )
        predictions_artifact = write_registered_json(
            ctx,
            PREDICTIONS_FILE,
            {
                **upstream_provenance(UPSTREAM_PATH),
                "tool": "deepmd-kit",
                "input": params.model_dump(mode="json"),
                "model_sha256": params.model_sha256,
                "energy_unit": "eV",
                "force_unit": "eV/angstrom",
                "predictions": predictions,
            },
            "external_deepmd_predictions",
        )
        return DeepMDInferenceOutput(
            predictions_artifact=predictions_artifact.artifact_id,
            predictions_path=predictions_artifact.relative_path,
            n_structures=len(predictions),
            model_sha256=params.model_sha256,
        )
