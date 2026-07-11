"""Fail-closed dependency gate plus real space-group analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ase.build import bulk

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.atomistics.structures_io import StructureSet, atoms_to_record
from mlip_research_agent.skills.base import SkillContext, SkillError, get_skill
from mlip_research_agent.skills.external_adapters.common import missing_dependency
from mlip_research_agent.skills.external_adapters.pymatgen_structure.implementation import (
    ExternalPymatgenSymmetrySkill,
)
from mlip_research_agent.skills.external_adapters.pymatgen_structure.schema import (
    PymatgenSymmetryInput,
    PymatgenSymmetryOutput,
)

IMPLEMENTATION = (
    "mlip_research_agent.skills.external_adapters.pymatgen_structure.implementation"
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
        step_id="pymatgen-symmetry",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )


def test_skill_is_registered() -> None:
    assert get_skill("external_pymatgen_structure") is ExternalPymatgenSymmetrySkill


def test_missing_dependency_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(module: str, *, package: str, extra: str | None) -> Any:
        raise missing_dependency(package, extra)

    monkeypatch.setattr(f"{IMPLEMENTATION}.require_module", refuse)
    ctx = prepare(tmp_path)
    with pytest.raises(SkillError, match="not installed"):
        ExternalPymatgenSymmetrySkill().run(
            PymatgenSymmetryInput(structures_path="inputs/structures.json"), ctx
        )
    assert ctx.registry.all() == []


def test_real_fcc_cu_space_group(tmp_path: Path) -> None:
    pytest.importorskip("pymatgen")
    ctx = prepare(tmp_path)
    output = ExternalPymatgenSymmetrySkill().run(
        PymatgenSymmetryInput(structures_path="inputs/structures.json"), ctx
    )
    assert isinstance(output, PymatgenSymmetryOutput)
    report = json.loads((tmp_path / output.report_path).read_text())
    assert report["structures"][0]["space_group_symbol"] == "Fm-3m"
    assert report["structures"][0]["space_group_number"] == 225
    assert ctx.registry.verify(output.report_artifact)
