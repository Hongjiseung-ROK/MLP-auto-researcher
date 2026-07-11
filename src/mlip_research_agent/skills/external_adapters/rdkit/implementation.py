"""Seeded ETKDG conformer generation via RDKit; fail-closed without it."""

from __future__ import annotations

from pydantic import BaseModel

from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill
from mlip_research_agent.skills.external_adapters.common import (
    adapter_error,
    require_module,
    upstream_provenance,
    write_registered_json,
)
from mlip_research_agent.skills.external_adapters.rdkit.schema import (
    RDKitConformerInput,
    RDKitConformerOutput,
)
from mlip_research_agent.skills.external_adapters.rdkit.validators import finite_conformers

UPSTREAM_PATH = "molecular-representation/rdkit-repr"
MOLECULE_FILE = "rdkit_molecule.json"


@register_skill
class ExternalRDKitConformerSkill(Skill):
    """Generate seeded 3D conformers for a SMILES string."""

    name = "external_rdkit"
    input_model = RDKitConformerInput
    output_model = RDKitConformerOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, RDKitConformerInput)
        chem = require_module("rdkit.Chem", package="rdkit", extra="rdkit")
        all_chem = require_module("rdkit.Chem.AllChem", package="rdkit", extra="rdkit")

        molecule = chem.MolFromSmiles(params.smiles)
        if molecule is None:
            raise adapter_error(f"RDKit could not parse SMILES: {params.smiles!r}")
        molecule = chem.AddHs(molecule)
        embed_params = all_chem.ETKDGv3()
        embed_params.randomSeed = ctx.seed
        conformer_ids = list(
            all_chem.EmbedMultipleConfs(
                molecule, numConfs=params.n_conformers, params=embed_params
            )
        )
        if not conformer_ids:
            raise adapter_error("RDKit embedding produced no conformers")
        if params.optimize == "mmff94":
            all_chem.MMFFOptimizeMoleculeConfs(molecule)
        symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        conformers = [
            [
                [float(value) for value in molecule.GetConformer(conformer_id).GetAtomPosition(i)]
                for i in range(molecule.GetNumAtoms())
            ]
            for conformer_id in conformer_ids
        ]
        finite_conformers(conformers)
        molecule_artifact = write_registered_json(
            ctx,
            MOLECULE_FILE,
            {
                **upstream_provenance(UPSTREAM_PATH),
                "tool": "rdkit",
                "input": params.model_dump(mode="json"),
                "seed": ctx.seed,
                "canonical_smiles": chem.MolToSmiles(chem.RemoveHs(molecule)),
                "symbols": symbols,
                "coordinate_unit": "angstrom",
                "conformers": conformers,
            },
            "external_rdkit_molecule",
        )
        payload_smiles = chem.MolToSmiles(chem.RemoveHs(molecule))
        return RDKitConformerOutput(
            molecule_artifact=molecule_artifact.artifact_id,
            molecule_path=molecule_artifact.relative_path,
            canonical_smiles=payload_smiles,
            n_atoms=len(symbols),
            n_conformers=len(conformers),
        )
