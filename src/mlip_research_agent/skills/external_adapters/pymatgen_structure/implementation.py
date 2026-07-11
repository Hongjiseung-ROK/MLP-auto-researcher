"""Space-group analysis over a StructureSet via pymatgen; fail-closed without it."""

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
from mlip_research_agent.skills.external_adapters.pymatgen_structure.schema import (
    PymatgenSymmetryInput,
    PymatgenSymmetryOutput,
)
from mlip_research_agent.skills.external_adapters.pymatgen_structure.validators import (
    finite_structure_records,
)

UPSTREAM_PATH = "data-processing/pymatgen-structure"
REPORT_FILE = "pymatgen_symmetry_report.json"


@register_skill
class ExternalPymatgenSymmetrySkill(Skill):
    """Report the space group of every structure in a StructureSet."""

    name = "external_pymatgen_structure"
    input_model = PymatgenSymmetryInput
    output_model = PymatgenSymmetryOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, PymatgenSymmetryInput)
        analyzer_module = require_module(
            "pymatgen.symmetry.analyzer", package="pymatgen", extra="atomistics"
        )
        ase_io = require_module("pymatgen.io.ase", package="pymatgen", extra="atomistics")
        from mlip_research_agent.skills.atomistics.structures_io import record_to_atoms

        structures = load_structure_set(ctx, params.structures_path)
        finite_structure_records(structures.systems)
        entries = []
        for record in structures.systems:
            structure = ase_io.AseAtomsAdaptor.get_structure(record_to_atoms(record))
            try:
                analyzer = analyzer_module.SpacegroupAnalyzer(
                    structure, symprec=params.symprec
                )
                entries.append(
                    {
                        "index": record.index,
                        "space_group_symbol": analyzer.get_space_group_symbol(),
                        "space_group_number": analyzer.get_space_group_number(),
                        "n_atoms": len(record.symbols),
                    }
                )
            except Exception as exc:  # pymatgen raises broad types on degenerate cells
                raise adapter_error(
                    f"space-group analysis failed for structure {record.index}: {exc}"
                ) from exc
        report_artifact = write_registered_json(
            ctx,
            REPORT_FILE,
            {
                **upstream_provenance(UPSTREAM_PATH),
                "tool": "pymatgen",
                "input": params.model_dump(mode="json"),
                "structures": entries,
            },
            "external_symmetry_report",
        )
        return PymatgenSymmetryOutput(
            report_artifact=report_artifact.artifact_id,
            report_path=report_artifact.relative_path,
            n_structures=len(entries),
        )
