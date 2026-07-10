"""Structure generation: pristine bulk plus seeded random perturbations."""

from __future__ import annotations

import numpy as np
from ase.build import bulk
from pydantic import BaseModel

from mlip_research_agent.skills.atomistics.schema import (
    StructureGenerationInput,
    StructureGenerationOutput,
)
from mlip_research_agent.skills.atomistics.structures_io import StructureSet, atoms_to_record
from mlip_research_agent.skills.atomistics.validators import (
    validate_inputs,
    validate_structure_set,
)
from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill

STRUCTURES_FILENAME = "structures.json"


@register_skill
class StructureGenerationSkill(Skill):
    name = "structure_generation"
    input_model = StructureGenerationInput
    output_model = StructureGenerationOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, StructureGenerationInput)
        validate_inputs(params)

        base = bulk(
            params.formula, params.crystal_structure, a=params.lattice_constant, cubic=False
        ).repeat(params.supercell)
        rng = np.random.default_rng(ctx.seed)
        records = []
        for i in range(params.n_candidates):
            # Ramp perturbation from ~0 to max so candidates span easy to hard.
            scale = params.max_perturbation * i / max(params.n_candidates - 1, 1)
            atoms = base.copy()
            if scale > 0:
                atoms.positions = atoms.positions + rng.normal(0.0, scale, atoms.positions.shape)
            records.append(atoms_to_record(atoms, index=i, perturbation_scale=float(scale)))

        structures = StructureSet(systems=records)
        validate_structure_set(structures)
        path = ctx.step_dir / STRUCTURES_FILENAME
        structures.save(path)
        artifact = ctx.registry.register(path, kind="structure_set", step_id=ctx.step_id)
        return StructureGenerationOutput(
            structures_artifact=artifact.artifact_id,
            structures_path=artifact.relative_path,
            n_structures=len(records),
            n_atoms_per_structure=len(base),
            elements=sorted(set(base.get_chemical_symbols())),
        )
