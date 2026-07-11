"""Hash gate before dependency gate; both fail closed."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from ase.build import bulk

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.atomistics.structures_io import StructureSet, atoms_to_record
from mlip_research_agent.skills.base import SkillContext, SkillError, get_skill
from mlip_research_agent.skills.external_adapters.common import missing_dependency
from mlip_research_agent.skills.external_adapters.deepmd_inference.implementation import (
    ExternalDeepMDInferenceSkill,
)
from mlip_research_agent.skills.external_adapters.deepmd_inference.schema import (
    DeepMDInferenceInput,
)

IMPLEMENTATION = (
    "mlip_research_agent.skills.external_adapters.deepmd_inference.implementation"
)


def prepare(tmp_path: Path) -> tuple[SkillContext, Path, str]:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    structures = StructureSet(
        systems=[atoms_to_record(bulk("Cu", "fcc", a=3.6, cubic=True), 0, 0.0)]
    )
    structures.save(inputs / "structures.json")
    model_path = inputs / "fake-model.pb"
    payload = b"not a real deepmd model"
    model_path.write_bytes(payload)
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="deepmd-inference",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )
    return ctx, model_path, hashlib.sha256(payload).hexdigest()


def test_skill_is_registered() -> None:
    assert get_skill("external_deepmd_inference") is ExternalDeepMDInferenceSkill


def test_model_hash_mismatch_aborts_before_dependency_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(module: str, *, package: str, extra: str | None) -> Any:
        raise AssertionError("dependency must not be consulted on a hash mismatch")

    monkeypatch.setattr(f"{IMPLEMENTATION}.require_module", explode)
    ctx, model_path, _ = prepare(tmp_path)
    with pytest.raises(SkillError, match="hash mismatch"):
        ExternalDeepMDInferenceSkill().run(
            DeepMDInferenceInput(
                structures_path="inputs/structures.json",
                model_path=str(model_path),
                model_sha256="0" * 64,
            ),
            ctx,
        )
    assert ctx.registry.all() == []


def test_missing_dependency_fails_closed_after_hash_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(module: str, *, package: str, extra: str | None) -> Any:
        raise missing_dependency(package, extra)

    monkeypatch.setattr(f"{IMPLEMENTATION}.require_module", refuse)
    ctx, model_path, model_sha = prepare(tmp_path)
    with pytest.raises(SkillError, match="not installed"):
        ExternalDeepMDInferenceSkill().run(
            DeepMDInferenceInput(
                structures_path="inputs/structures.json",
                model_path=str(model_path),
                model_sha256=model_sha,
            ),
            ctx,
        )
    assert ctx.registry.all() == []
