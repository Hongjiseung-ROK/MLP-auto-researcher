"""Fail-closed dependency gate plus real displacement generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from ase.build import bulk

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.atomistics.structures_io import (
    StructureSet,
    atoms_to_record,
)
from mlip_research_agent.skills.base import SkillContext, SkillError, get_skill
from mlip_research_agent.skills.external_adapters.common import missing_dependency
from mlip_research_agent.skills.external_adapters.phonopy.implementation import (
    ExternalPhonopySkill,
)
from mlip_research_agent.skills.external_adapters.phonopy.schema import (
    PhonopyDisplacementsInput,
    PhonopyDisplacementsOutput,
)

IMPLEMENTATION = "mlip_research_agent.skills.external_adapters.phonopy.implementation"


def prepare(tmp_path: Path) -> SkillContext:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    structures = StructureSet(
        systems=[atoms_to_record(bulk("Cu", "fcc", a=3.6, cubic=True), 0, 0.0)]
    )
    structures.save(inputs / "structures.json")
    return SkillContext(
        run_dir=tmp_path,
        step_id="phonopy-displacements",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )


def displacement_input(**overrides: object) -> PhonopyDisplacementsInput:
    payload: dict[str, object] = {
        "structures_path": "inputs/structures.json",
        "supercell": [2, 2, 2],
    }
    payload.update(overrides)
    return PhonopyDisplacementsInput.model_validate(payload)


def test_skill_is_registered() -> None:
    assert get_skill("external_phonopy") is ExternalPhonopySkill


def test_schema_rejects_oversized_supercell() -> None:
    with pytest.raises(ValueError, match="within"):
        displacement_input(supercell=[5, 1, 1])


def test_missing_dependency_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(module: str, *, package: str, extra: str | None) -> Any:
        raise missing_dependency(package, extra)

    monkeypatch.setattr(f"{IMPLEMENTATION}.require_module", refuse)
    ctx = prepare(tmp_path)
    with pytest.raises(SkillError, match="not installed"):
        ExternalPhonopySkill().run(displacement_input(), ctx)
    assert ctx.registry.all() == []


def test_real_displacement_generation(tmp_path: Path) -> None:
    pytest.importorskip("phonopy")
    ctx = prepare(tmp_path)
    output = ExternalPhonopySkill().run(displacement_input(), ctx)
    assert isinstance(output, PhonopyDisplacementsOutput)
    assert output.n_displacements >= 1
    assert output.n_atoms_per_supercell == 32
    displaced = StructureSet.load(tmp_path / output.structures_path)
    assert len(displaced.systems) == output.n_displacements
    assert all(ctx.registry.verify(a.artifact_id) for a in ctx.registry.all())
