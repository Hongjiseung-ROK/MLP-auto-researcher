"""Finite-displacement supercell generation via phonopy; fail-closed without it."""

from __future__ import annotations

from pydantic import BaseModel

from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill
from mlip_research_agent.skills.external_adapters.common import (
    adapter_error,
    load_structure_set,
    require_module,
    upstream_provenance,
    write_registered_json,
)
from mlip_research_agent.skills.external_adapters.phonopy.schema import (
    PhonopyDisplacementsInput,
    PhonopyDisplacementsOutput,
)
from mlip_research_agent.skills.external_adapters.phonopy.validators import (
    finite_structure_records,
    structure_index,
)

UPSTREAM_PATH = "analysis/phonopy"
STRUCTURES_FILE = "phonopy_displacements.json"
PROVENANCE_FILE = "phonopy_provenance.json"


@register_skill
class ExternalPhonopySkill(Skill):
    """Generate displaced supercells for finite-difference force constants."""

    name = "external_phonopy"
    input_model = PhonopyDisplacementsInput
    output_model = PhonopyDisplacementsOutput
    cost_class = "cheap"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, PhonopyDisplacementsInput)
        phonopy_module = require_module("phonopy", package="phonopy", extra="phonopy")
        atoms_module = require_module(
            "phonopy.structure.atoms", package="phonopy", extra="phonopy"
        )
        from mlip_research_agent.skills.atomistics.structures_io import (
            StructureRecord,
            StructureSet,
        )

        structures = load_structure_set(ctx, params.structures_path)
        finite_structure_records(structures.systems)
        record = structure_index(structures.systems, params.structure_index)
        unitcell = atoms_module.PhonopyAtoms(
            symbols=record.symbols,
            cell=record.cell,
            positions=record.positions,
        )
        supercell_matrix = [
            [params.supercell[0], 0, 0],
            [0, params.supercell[1], 0],
            [0, 0, params.supercell[2]],
        ]
        phonon = phonopy_module.Phonopy(unitcell, supercell_matrix=supercell_matrix)
        phonon.generate_displacements(distance=params.displacement_distance_a)
        supercells = phonon.supercells_with_displacements
        if not supercells:
            raise adapter_error("phonopy generated no displaced supercells")
        records = [
            StructureRecord(
                index=index,
                symbols=[str(symbol) for symbol in supercell.symbols],
                positions=[[float(x) for x in row] for row in supercell.positions],
                cell=[[float(x) for x in row] for row in supercell.cell],
                pbc=[True, True, True],
                perturbation_scale=params.displacement_distance_a,
            )
            for index, supercell in enumerate(supercells)
        ]
        displaced = StructureSet(systems=records)
        structures_path = ctx.step_dir / STRUCTURES_FILE
        displaced.save(structures_path)
        structures_artifact = ctx.registry.register(
            structures_path, "external_structure_set", ctx.step_id
        )
        provenance_artifact = write_registered_json(
            ctx,
            PROVENANCE_FILE,
            {
                **upstream_provenance(UPSTREAM_PATH),
                "tool": "phonopy",
                "input": params.model_dump(mode="json"),
                "structures_artifact": structures_artifact.artifact_id,
                "structures_sha256": structures_artifact.sha256,
            },
            "external_adapter_provenance",
        )
        return PhonopyDisplacementsOutput(
            structures_artifact=structures_artifact.artifact_id,
            structures_path=structures_artifact.relative_path,
            provenance_artifact=provenance_artifact.artifact_id,
            provenance_path=provenance_artifact.relative_path,
            n_displacements=len(records),
            n_atoms_per_supercell=len(records[0].symbols),
        )
