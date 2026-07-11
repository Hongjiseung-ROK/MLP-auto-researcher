"""Determinism and schema gates for the ASE build adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.base import SkillContext, get_skill
from mlip_research_agent.skills.external_adapters.ase.implementation import (
    ExternalASEBuildSkill,
)
from mlip_research_agent.skills.external_adapters.ase.schema import (
    ASEBuildInput,
    ASEBuildOutput,
)


def make_context(tmp_path: Path, seed: int = 11) -> SkillContext:
    return SkillContext(
        run_dir=tmp_path,
        step_id="ase-build",
        seed=seed,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )


def build_input(**overrides: object) -> ASEBuildInput:
    payload: dict[str, object] = {
        "element": "Cu",
        "crystalstructure": "fcc",
        "lattice_a": 3.6,
        "cubic": True,
        "n_rattled": 2,
        "rattle_stdev_a": 0.02,
    }
    payload.update(overrides)
    return ASEBuildInput.model_validate(payload)


def test_skill_is_registered() -> None:
    assert get_skill("external_ase") is ExternalASEBuildSkill


def test_build_is_seed_deterministic(tmp_path: Path) -> None:
    outputs = []
    for run in ("first", "second"):
        ctx = make_context(tmp_path / run)
        output = ExternalASEBuildSkill().run(build_input(), ctx)
        assert isinstance(output, ASEBuildOutput)
        assert output.n_structures == 3
        assert output.n_atoms_per_structure == 4
        assert all(ctx.registry.verify(a.artifact_id) for a in ctx.registry.all())
        outputs.append((tmp_path / run / output.structures_path).read_bytes())
    assert outputs[0] == outputs[1]


def test_different_seed_changes_rattled_structures(tmp_path: Path) -> None:
    first = ExternalASEBuildSkill().run(build_input(), make_context(tmp_path / "a", seed=1))
    second = ExternalASEBuildSkill().run(build_input(), make_context(tmp_path / "b", seed=2))
    assert isinstance(first, ASEBuildOutput) and isinstance(second, ASEBuildOutput)
    first_bytes = (tmp_path / "a" / first.structures_path).read_bytes()
    second_bytes = (tmp_path / "b" / second.structures_path).read_bytes()
    assert first_bytes != second_bytes


def test_schema_rejects_inconsistent_inputs() -> None:
    with pytest.raises(ValueError, match="no cubic conventional cell"):
        build_input(crystalstructure="hcp")
    with pytest.raises(ValueError, match="positive rattle_stdev_a"):
        build_input(n_rattled=3, rattle_stdev_a=0.0)
    with pytest.raises(ValueError, match="no effect"):
        build_input(n_rattled=0, rattle_stdev_a=0.05)
