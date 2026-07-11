"""Deterministic pw.x SCF input generation. Preparation only: QE execution is
not an approved backend (the Phase 2 DFT decision is ORCA), so this adapter
never runs a binary."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill
from mlip_research_agent.skills.external_adapters.common import (
    load_structure_set,
    upstream_provenance,
    write_registered_json,
    write_registered_text,
)
from mlip_research_agent.skills.external_adapters.dft_qe.schema import (
    QEPrepareInput,
    QEPrepareOutput,
)
from mlip_research_agent.skills.external_adapters.dft_qe.validators import (
    finite_structure_records,
    require_species_coverage,
    structure_index,
)

UPSTREAM_PATH = "quantum-chemistry/dft-qe"
INPUT_FILE = "pw_scf.in"
PROVENANCE_FILE = "qe_provenance.json"


def _atomic_mass(symbol: str) -> float:
    from ase.data import atomic_masses, atomic_numbers

    return float(atomic_masses[atomic_numbers[symbol]])


def render_pw_input(params: QEPrepareInput, record: Any) -> str:
    symbols: list[str] = record.symbols
    positions: list[list[float]] = record.positions
    cell: list[list[float]] = record.cell
    species = sorted(set(symbols))
    ecutrho = params.ecutrho_ry if params.ecutrho_ry is not None else 4 * params.ecutwfc_ry
    lines = [
        "&CONTROL",
        f"  calculation = '{params.calculation}'",
        "  prefix = 'external_qe'",
        "  pseudo_dir = './pseudo'",
        "  outdir = './out'",
        "/",
        "&SYSTEM",
        "  ibrav = 0",
        f"  nat = {len(symbols)}",
        f"  ntyp = {len(species)}",
        f"  ecutwfc = {params.ecutwfc_ry:.4f}",
        f"  ecutrho = {ecutrho:.4f}",
        "/",
        "&ELECTRONS",
        "  conv_thr = 1.0d-8",
        "/",
        "ATOMIC_SPECIES",
    ]
    for symbol in species:
        lines.append(
            f"  {symbol} {_atomic_mass(symbol):.6f} {params.pseudopotentials[symbol]}"
        )
    lines.append("CELL_PARAMETERS angstrom")
    for row in cell:
        lines.append("  " + " ".join(f"{value:.10f}" for value in row))
    lines.append("ATOMIC_POSITIONS angstrom")
    for symbol, position in zip(symbols, positions, strict=True):
        lines.append(f"  {symbol} " + " ".join(f"{value:.10f}" for value in position))
    lines.append("K_POINTS automatic")
    lines.append("  " + " ".join(str(k) for k in params.kpoints) + " 0 0 0")
    return "\n".join(lines) + "\n"


@register_skill
class ExternalQEPrepareSkill(Skill):
    """Render a deterministic pw.x SCF input deck for one structure."""

    name = "external_dft_qe"
    input_model = QEPrepareInput
    output_model = QEPrepareOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, QEPrepareInput)
        structures = load_structure_set(ctx, params.structures_path)
        finite_structure_records(structures.systems)
        record = structure_index(structures.systems, params.structure_index)
        require_species_coverage(
            set(record.symbols), set(params.pseudopotentials), "pseudopotentials"
        )
        deck_artifact = write_registered_text(
            ctx, INPUT_FILE, render_pw_input(params, record), "external_qe_input_deck"
        )
        provenance_artifact = write_registered_json(
            ctx,
            PROVENANCE_FILE,
            {
                **upstream_provenance(UPSTREAM_PATH),
                "tool": "quantum-espresso (input preparation only)",
                "execution": "forbidden: QE is not the approved Phase 2 DFT backend",
                "input": params.model_dump(mode="json"),
                "input_deck_artifact": deck_artifact.artifact_id,
                "input_deck_sha256": deck_artifact.sha256,
            },
            "external_adapter_provenance",
        )
        return QEPrepareOutput(
            input_deck_artifact=deck_artifact.artifact_id,
            input_deck_path=deck_artifact.relative_path,
            provenance_artifact=provenance_artifact.artifact_id,
            provenance_path=provenance_artifact.relative_path,
            n_atoms=len(record.symbols),
            n_species=len(set(record.symbols)),
        )
