"""Prepare-only LAMMPS/DeepMD input generation gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import bulk

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.atomistics.structures_io import StructureSet, atoms_to_record
from mlip_research_agent.skills.base import SkillContext, SkillError, get_skill
from mlip_research_agent.skills.external_adapters.lammps.implementation import (
    ExternalLAMMPSPrepareSkill,
)
from mlip_research_agent.skills.external_adapters.lammps.schema import (
    LAMMPSPrepareInput,
    LAMMPSPrepareOutput,
)


def prepare(tmp_path: Path, *, sheared: bool = False) -> SkillContext:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    if sheared:
        cell = atoms.cell[:]
        cell[0][1] = 0.7
        atoms.set_cell(cell)
    structures = StructureSet(systems=[atoms_to_record(atoms, 0, 0.0)])
    structures.save(inputs / "structures.json")
    return SkillContext(
        run_dir=tmp_path,
        step_id="lammps-prepare",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )


def lammps_input(**overrides: object) -> LAMMPSPrepareInput:
    payload: dict[str, object] = {
        "structures_path": "inputs/structures.json",
        "model_filename": "graph.pb",
        "n_steps": 10,
    }
    payload.update(overrides)
    return LAMMPSPrepareInput.model_validate(payload)


def test_skill_is_registered() -> None:
    assert get_skill("external_lammps") is ExternalLAMMPSPrepareSkill


def test_renders_deterministic_deck_and_data(tmp_path: Path) -> None:
    rendered = []
    for run in ("first", "second"):
        ctx = prepare(tmp_path / run)
        output = ExternalLAMMPSPrepareSkill().run(lammps_input(), ctx)
        assert isinstance(output, LAMMPSPrepareOutput)
        assert output.n_atoms == 4
        assert output.type_map == {"Cu": 1}
        deck = (tmp_path / run / output.input_deck_path).read_text()
        data = (tmp_path / run / output.data_file_path).read_text()
        assert "pair_style deepmd graph.pb" in deck
        assert "pair_coeff * * Cu" in deck
        assert "fix integrate all nve" in deck
        assert "run 10" in deck
        assert "4 atoms" in data
        assert "1 63.546000  # Cu" in data
        assert all(ctx.registry.verify(a.artifact_id) for a in ctx.registry.all())
        rendered.append(deck + data)
    assert rendered[0] == rendered[1]


def test_non_orthorhombic_cell_fails_closed(tmp_path: Path) -> None:
    ctx = prepare(tmp_path, sheared=True)
    with pytest.raises(SkillError, match="orthorhombic"):
        ExternalLAMMPSPrepareSkill().run(lammps_input(), ctx)


def test_schema_rejects_unsafe_model_filename() -> None:
    with pytest.raises(ValueError, match="model_filename"):
        lammps_input(model_filename="../graph.pb")
