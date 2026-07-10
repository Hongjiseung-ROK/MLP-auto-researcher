"""Correctness tests for the mock_dft_labeling skill."""

import json
from pathlib import Path
from typing import Any

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.schemas.failure import FailureClass
from mlip_research_agent.skills.atomistics.implementation import StructureGenerationSkill
from mlip_research_agent.skills.atomistics.schema import StructureGenerationInput
from mlip_research_agent.skills.base import SkillContext, SkillError
from mlip_research_agent.skills.dft.implementation import MockDFTLabelingSkill
from mlip_research_agent.skills.dft.schema import LabelingInput, LabelingOutput


def make_structures(tmp_path: Path, seed: int = 7) -> tuple[ArtifactRegistry, str]:
    registry = ArtifactRegistry(tmp_path)
    ctx = SkillContext(
        run_dir=tmp_path, step_id="structures", seed=seed, attempt=0, registry=registry
    )
    out = StructureGenerationSkill().run(
        StructureGenerationInput(
            formula="Cu",
            crystal_structure="fcc",
            lattice_constant=3.6,
            n_candidates=6,
            max_perturbation=0.08,
        ),
        ctx,
    )
    return registry, out.structures_path  # type: ignore[attr-defined]


def label(
    tmp_path: Path,
    registry: ArtifactRegistry,
    structures_path: str,
    attempt: int = 0,
    **kw: object,
) -> dict[str, Any]:
    ctx = SkillContext(
        run_dir=tmp_path, step_id="labeling", seed=7, attempt=attempt, registry=registry
    )
    inputs = LabelingInput.model_validate({"structures_path": structures_path, **kw})
    out = MockDFTLabelingSkill().run(inputs, ctx)
    assert isinstance(out, LabelingOutput)
    payload: dict[str, Any] = json.loads((tmp_path / out.labels_path).read_text())
    return payload


def test_deterministic(tmp_path: Path) -> None:
    registry_a, path_a = make_structures(tmp_path / "a")
    registry_b, path_b = make_structures(tmp_path / "b")
    assert label(tmp_path / "a", registry_a, path_a) == label(tmp_path / "b", registry_b, path_b)


def test_selection_subsets_labels(tmp_path: Path) -> None:
    registry, structures_path = make_structures(tmp_path)
    selection_dir = tmp_path / "steps" / "selection"
    selection_dir.mkdir(parents=True)
    (selection_dir / "selection.json").write_text(json.dumps({"selected_indices": [1, 3]}))
    payload = label(
        tmp_path, registry, structures_path, selection_path="steps/selection/selection.json"
    )
    assert [rec["index"] for rec in payload["labels"]] == [1, 3]


def test_failure_injection_honors_attempt(tmp_path: Path) -> None:
    registry, structures_path = make_structures(tmp_path)
    with pytest.raises(SkillError) as excinfo:
        label(tmp_path, registry, structures_path, attempt=0, inject_failure_times=1)
    assert excinfo.value.failure_class is FailureClass.CONVERGENCE_FAILURE
    assert excinfo.value.retryable
    assert excinfo.value.repair_params == {"scf_damping": 0.7}
    # Attempt 1 succeeds and records the repair knob in settings provenance.
    payload = label(
        tmp_path, registry, structures_path, attempt=1, inject_failure_times=1, scf_damping=0.7
    )
    assert payload["settings"]["scf_damping"] == 0.7


def test_unknown_method_rejected(tmp_path: Path) -> None:
    registry, structures_path = make_structures(tmp_path)
    with pytest.raises(SkillError) as excinfo:
        label(tmp_path, registry, structures_path, method="b3lyp/def2-tzvp")
    assert excinfo.value.failure_class is FailureClass.VALIDATION_ERROR
    assert not excinfo.value.retryable
