"""Correctness tests for the structure_generation skill."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.schemas.failure import FailureClass
from mlip_research_agent.skills.atomistics.implementation import StructureGenerationSkill
from mlip_research_agent.skills.atomistics.schema import StructureGenerationInput
from mlip_research_agent.skills.base import SkillContext, SkillError

EXAMPLE = json.loads(
    (Path(__file__).parent.parent / "examples" / "example_input.json").read_text()
)


def make_ctx(tmp_path: Path, seed: int = 7) -> SkillContext:
    return SkillContext(
        run_dir=tmp_path,
        step_id="structures",
        seed=seed,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )


def run_skill(tmp_path: Path, seed: int = 7, **overrides: object) -> tuple[str, str]:
    inputs = StructureGenerationInput.model_validate({**EXAMPLE, **overrides})
    ctx = make_ctx(tmp_path, seed)
    out = StructureGenerationSkill().run(inputs, ctx)
    artifact = ctx.registry.get(out.structures_artifact)  # type: ignore[attr-defined]
    assert artifact is not None
    return (tmp_path / artifact.relative_path).read_text(), artifact.sha256


def test_deterministic_for_equal_seeds(tmp_path: Path) -> None:
    content_a, sha_a = run_skill(tmp_path / "a", seed=7)
    content_b, sha_b = run_skill(tmp_path / "b", seed=7)
    assert content_a == content_b
    assert sha_a == sha_b


def test_different_seed_changes_output(tmp_path: Path) -> None:
    _, sha_a = run_skill(tmp_path / "a", seed=7)
    _, sha_b = run_skill(tmp_path / "b", seed=8)
    assert sha_a != sha_b


def test_unsupported_lattice_rejected(tmp_path: Path) -> None:
    inputs = StructureGenerationInput.model_validate(
        {**EXAMPLE, "crystal_structure": "quasicrystal"}
    )
    with pytest.raises(SkillError) as excinfo:
        StructureGenerationSkill().run(inputs, make_ctx(tmp_path))
    assert excinfo.value.failure_class is FailureClass.VALIDATION_ERROR
    assert not excinfo.value.retryable


def test_extreme_perturbation_fails_with_repair(tmp_path: Path) -> None:
    inputs = StructureGenerationInput.model_validate({**EXAMPLE, "max_perturbation": 1.0})
    with pytest.raises(SkillError) as excinfo:
        StructureGenerationSkill().run(inputs, make_ctx(tmp_path, seed=3))
    assert excinfo.value.failure_class is FailureClass.SIMULATION_INSTABILITY
    assert excinfo.value.retryable
    assert excinfo.value.repair_params == {"max_perturbation": 0.05}


def test_input_schema_rejects_bad_values() -> None:
    with pytest.raises(ValidationError):
        StructureGenerationInput.model_validate({**EXAMPLE, "n_candidates": 1})
    with pytest.raises(ValidationError):
        StructureGenerationInput.model_validate({**EXAMPLE, "lattice_constant": -1.0})
