"""Prepare-only QE input generation gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import bulk

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.atomistics.structures_io import StructureSet, atoms_to_record
from mlip_research_agent.skills.base import SkillContext, SkillError, get_skill
from mlip_research_agent.skills.external_adapters.dft_qe.implementation import (
    ExternalQEPrepareSkill,
)
from mlip_research_agent.skills.external_adapters.dft_qe.schema import (
    QEPrepareInput,
    QEPrepareOutput,
)


def prepare(tmp_path: Path) -> SkillContext:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    structures = StructureSet(
        systems=[atoms_to_record(bulk("Cu", "fcc", a=3.6, cubic=True), 0, 0.0)]
    )
    structures.save(inputs / "structures.json")
    return SkillContext(
        run_dir=tmp_path,
        step_id="qe-prepare",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )


def qe_input(**overrides: object) -> QEPrepareInput:
    payload: dict[str, object] = {
        "structures_path": "inputs/structures.json",
        "ecutwfc_ry": 60.0,
        "kpoints": [4, 4, 4],
        "pseudopotentials": {"Cu": "Cu.paw.upf"},
    }
    payload.update(overrides)
    return QEPrepareInput.model_validate(payload)


def test_skill_is_registered() -> None:
    assert get_skill("external_dft_qe") is ExternalQEPrepareSkill


def test_renders_deterministic_scf_deck(tmp_path: Path) -> None:
    decks = []
    for run in ("first", "second"):
        ctx = prepare(tmp_path / run)
        output = ExternalQEPrepareSkill().run(qe_input(), ctx)
        assert isinstance(output, QEPrepareOutput)
        assert output.n_atoms == 4
        assert output.n_species == 1
        deck = (tmp_path / run / output.input_deck_path).read_text()
        assert "calculation = 'scf'" in deck
        assert "ecutwfc = 60.0000" in deck
        assert "ecutrho = 240.0000" in deck
        assert "Cu 63.546000 Cu.paw.upf" in deck
        assert "K_POINTS automatic\n  4 4 4 0 0 0" in deck
        assert all(ctx.registry.verify(a.artifact_id) for a in ctx.registry.all())
        decks.append(deck)
    assert decks[0] == decks[1]


def test_missing_pseudopotential_fails_closed(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    with pytest.raises(SkillError, match="missing for species"):
        ExternalQEPrepareSkill().run(
            qe_input(pseudopotentials={"Al": "Al.paw.upf"}), ctx
        )


def test_schema_rejects_unsafe_and_inconsistent_inputs() -> None:
    with pytest.raises(ValueError, match="unsafe pseudopotential filename"):
        qe_input(pseudopotentials={"Cu": "../evil.upf"})
    with pytest.raises(ValueError, match="at least 4"):
        qe_input(ecutrho_ry=100.0)
    with pytest.raises(ValueError, match="within"):
        qe_input(kpoints=[0, 4, 4])


def test_escaping_structures_path_rejected(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    with pytest.raises(SkillError, match="escapes the run directory"):
        ExternalQEPrepareSkill().run(qe_input(structures_path="../outside.json"), ctx)
