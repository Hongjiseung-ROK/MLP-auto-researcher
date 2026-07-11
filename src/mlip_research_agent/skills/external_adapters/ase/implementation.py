"""Deterministic ASE bulk-structure builder behind the SKILL contract."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill
from mlip_research_agent.skills.external_adapters.ase.schema import (
    ASEBuildInput,
    ASEBuildOutput,
)
from mlip_research_agent.skills.external_adapters.ase.validators import (
    finite_structure_records,
)
from mlip_research_agent.skills.external_adapters.common import (
    upstream_provenance,
    write_registered_json,
)

UPSTREAM_PATH = "atomistic-workflows/ase"
STRUCTURES_FILE = "ase_structures.json"
PROVENANCE_FILE = "ase_provenance.json"


@register_skill
class ExternalASEBuildSkill(Skill):
    """Build a bulk crystal plus seeded rattled copies as a StructureSet."""

    name = "external_ase"
    input_model = ASEBuildInput
    output_model = ASEBuildOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, ASEBuildInput)
        from ase.build import bulk

        from mlip_research_agent.skills.atomistics.structures_io import (
            StructureSet,
            atoms_to_record,
        )

        base = bulk(
            params.element,
            params.crystalstructure,
            a=params.lattice_a,
            cubic=params.cubic,
        )
        if params.repeat > 1:
            base = base.repeat(params.repeat)
        rng = np.random.default_rng(ctx.seed)
        records = [atoms_to_record(base, 0, 0.0)]
        for index in range(params.n_rattled):
            rattled = base.copy()
            rattled.positions = rattled.positions + rng.normal(
                0.0, params.rattle_stdev_a, rattled.positions.shape
            )
            records.append(atoms_to_record(rattled, index + 1, params.rattle_stdev_a))
        finite_structure_records(records)

        structures = StructureSet(systems=records)
        structures_path = ctx.step_dir / STRUCTURES_FILE
        structures.save(structures_path)
        structures_artifact = ctx.registry.register(
            structures_path, "external_structure_set", ctx.step_id
        )
        provenance_artifact = write_registered_json(
            ctx,
            PROVENANCE_FILE,
            {
                **upstream_provenance(UPSTREAM_PATH),
                "tool": "ase",
                "input": params.model_dump(mode="json"),
                "seed": ctx.seed,
                "structures_artifact": structures_artifact.artifact_id,
                "structures_sha256": structures_artifact.sha256,
            },
            "external_adapter_provenance",
        )
        return ASEBuildOutput(
            structures_artifact=structures_artifact.artifact_id,
            structures_path=structures_artifact.relative_path,
            provenance_artifact=provenance_artifact.artifact_id,
            provenance_path=provenance_artifact.relative_path,
            n_structures=len(records),
            n_atoms_per_structure=len(base),
        )
