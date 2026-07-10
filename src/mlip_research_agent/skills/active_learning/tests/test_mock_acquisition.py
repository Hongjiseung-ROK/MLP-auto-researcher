"""Correctness tests for the mock_acquisition skill."""

import json
from pathlib import Path
from typing import Any

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.schemas.failure import FailureClass
from mlip_research_agent.skills.active_learning.implementation import MockAcquisitionSkill
from mlip_research_agent.skills.active_learning.schema import (
    AcquisitionInput,
    AcquisitionOutput,
)
from mlip_research_agent.skills.atomistics.implementation import StructureGenerationSkill
from mlip_research_agent.skills.atomistics.schema import StructureGenerationInput
from mlip_research_agent.skills.base import SkillContext, SkillError


def make_run(tmp_path: Path, seed: int = 7) -> tuple[ArtifactRegistry, str]:
    registry = ArtifactRegistry(tmp_path)
    gen_ctx = SkillContext(
        run_dir=tmp_path, step_id="structures", seed=seed, attempt=0, registry=registry
    )
    gen_out = StructureGenerationSkill().run(
        StructureGenerationInput(
            formula="Cu",
            crystal_structure="fcc",
            lattice_constant=3.6,
            n_candidates=10,
            max_perturbation=0.1,
        ),
        gen_ctx,
    )
    return registry, gen_out.structures_path  # type: ignore[attr-defined]


def select(
    tmp_path: Path, registry: ArtifactRegistry, structures_path: str, **kw: object
) -> dict[str, Any]:
    ctx = SkillContext(run_dir=tmp_path, step_id="selection", seed=7, attempt=0, registry=registry)
    inputs = AcquisitionInput.model_validate(
        {"structures_path": structures_path, "n_select": 4, **kw}
    )
    out = MockAcquisitionSkill().run(inputs, ctx)
    assert isinstance(out, AcquisitionOutput)
    payload: dict[str, Any] = json.loads((tmp_path / out.selection_path).read_text())
    return payload


def test_deterministic(tmp_path: Path) -> None:
    registry_a, path_a = make_run(tmp_path / "a")
    registry_b, path_b = make_run(tmp_path / "b")
    sel_a = select(tmp_path / "a", registry_a, path_a)
    sel_b = select(tmp_path / "b", registry_b, path_b)
    assert sel_a == sel_b


def test_uncertainty_proxy_prefers_perturbed(tmp_path: Path) -> None:
    registry, structures_path = make_run(tmp_path)
    sel = select(tmp_path, registry, structures_path)
    # Highest-index structures carry the largest perturbations.
    assert all(i >= 5 for i in sel["selected_indices"])


def test_oversized_selection_rejected(tmp_path: Path) -> None:
    registry, structures_path = make_run(tmp_path)
    ctx = SkillContext(run_dir=tmp_path, step_id="selection", seed=7, attempt=0, registry=registry)
    with pytest.raises(SkillError) as excinfo:
        MockAcquisitionSkill().run(
            AcquisitionInput(structures_path=structures_path, n_select=999), ctx
        )
    assert excinfo.value.failure_class is FailureClass.VALIDATION_ERROR
