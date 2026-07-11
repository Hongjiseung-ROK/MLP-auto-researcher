"""Fail-closed dependency gate plus real seeded conformer generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.base import SkillContext, SkillError, get_skill
from mlip_research_agent.skills.external_adapters.common import missing_dependency
from mlip_research_agent.skills.external_adapters.rdkit.implementation import (
    ExternalRDKitConformerSkill,
)
from mlip_research_agent.skills.external_adapters.rdkit.schema import (
    RDKitConformerInput,
    RDKitConformerOutput,
)

IMPLEMENTATION = "mlip_research_agent.skills.external_adapters.rdkit.implementation"


def make_context(tmp_path: Path, seed: int = 7) -> SkillContext:
    return SkillContext(
        run_dir=tmp_path,
        step_id="rdkit-conformers",
        seed=seed,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )


def test_skill_is_registered() -> None:
    assert get_skill("external_rdkit") is ExternalRDKitConformerSkill


def test_missing_dependency_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(module: str, *, package: str, extra: str | None) -> Any:
        raise missing_dependency(package, extra)

    monkeypatch.setattr(f"{IMPLEMENTATION}.require_module", refuse)
    with pytest.raises(SkillError, match="not installed"):
        ExternalRDKitConformerSkill().run(
            RDKitConformerInput(smiles="CCO"), make_context(tmp_path)
        )


def test_real_seeded_conformers_are_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    payloads = []
    for run in ("first", "second"):
        ctx = make_context(tmp_path / run)
        output = ExternalRDKitConformerSkill().run(
            RDKitConformerInput(smiles="CCO", n_conformers=2), ctx
        )
        assert isinstance(output, RDKitConformerOutput)
        assert output.n_atoms == 9  # CCO with explicit hydrogens
        assert output.n_conformers == 2
        assert ctx.registry.verify(output.molecule_artifact)
        payloads.append((tmp_path / run / output.molecule_path).read_bytes())
    assert payloads[0] == payloads[1]


def test_real_invalid_smiles_fails_closed(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    with pytest.raises(SkillError, match="could not parse SMILES"):
        ExternalRDKitConformerSkill().run(
            RDKitConformerInput(smiles="not-a-smiles(("), make_context(tmp_path)
        )


def test_upstream_pin_recorded_in_output(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    ctx = make_context(tmp_path)
    output = ExternalRDKitConformerSkill().run(
        RDKitConformerInput(smiles="C"), ctx
    )
    assert isinstance(output, RDKitConformerOutput)
    payload = json.loads((tmp_path / output.molecule_path).read_text())
    assert payload["upstream_commit"] == "93ea0c4c716ad116869fba2ade26cccfd5cd05fc"
