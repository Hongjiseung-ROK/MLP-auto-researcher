"""Fail-closed dependency gate plus real deepmd/npy conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.data.manifests import LabeledConfiguration, NormalizedDataset
from mlip_research_agent.skills.base import SkillContext, SkillError, get_skill
from mlip_research_agent.skills.external_adapters.common import missing_dependency
from mlip_research_agent.skills.external_adapters.dpdata_cli.implementation import (
    ExternalDpdataConvertSkill,
)
from mlip_research_agent.skills.external_adapters.dpdata_cli.schema import (
    DpdataConvertInput,
    DpdataConvertOutput,
)

IMPLEMENTATION = "mlip_research_agent.skills.external_adapters.dpdata_cli.implementation"


def make_dataset() -> NormalizedDataset:
    configurations = [
        LabeledConfiguration(
            config_id=f"dpdata-fixture-{index:02d}",
            source_id=f"fixture.json[{index}]",
            top_group="fixture",
            group_id="fixture",
            split_unit_id=f"unit-{index:02d}",
            symbols=["Cu", "Cu"],
            positions=[[0.0, 0.0, 0.0], [1.8 + index / 100, 1.8, 1.8]],
            cell=[[3.6, 0.0, 0.0], [0.0, 3.6, 0.0], [0.0, 0.0, 3.6]],
            pbc=[True, True, True],
            energy_ev=-7.0 - index / 10,
            forces_ev_per_a=[[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]],
            virial_stress_kbar=None,
            level_of_theory="synthetic-test",
            source_record_sha256=f"{index + 400:064x}",
        )
        for index in range(3)
    ]
    return NormalizedDataset(
        dataset_id="dpdata_fixture",
        level_of_theory="synthetic-test",
        configurations=configurations,
    )


def prepare(tmp_path: Path) -> SkillContext:
    inputs = tmp_path / "inputs"
    make_dataset().save(inputs / "normalized_dataset.json")
    return SkillContext(
        run_dir=tmp_path,
        step_id="dpdata-convert",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )


def test_skill_is_registered() -> None:
    assert get_skill("external_dpdata_cli") is ExternalDpdataConvertSkill


def test_missing_dependency_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(module: str, *, package: str, extra: str | None) -> Any:
        raise missing_dependency(package, extra)

    monkeypatch.setattr(f"{IMPLEMENTATION}.require_module", refuse)
    ctx = prepare(tmp_path)
    with pytest.raises(SkillError, match="not installed"):
        ExternalDpdataConvertSkill().run(
            DpdataConvertInput(dataset_path="inputs/normalized_dataset.json"), ctx
        )
    assert ctx.registry.all() == []


def test_real_conversion_to_deepmd_npy(tmp_path: Path) -> None:
    pytest.importorskip("dpdata")
    ctx = prepare(tmp_path)
    output = ExternalDpdataConvertSkill().run(
        DpdataConvertInput(dataset_path="inputs/normalized_dataset.json"), ctx
    )
    assert isinstance(output, DpdataConvertOutput)
    assert output.n_configurations == 3
    assert output.n_systems == 1
    assert output.n_files > 0
    manifest = json.loads((tmp_path / output.output_manifest_path).read_text())
    assert manifest["dataset_content_sha256"] == make_dataset().content_hash()
    assert len(manifest["files"]) == output.n_files
    assert all(ctx.registry.verify(a.artifact_id) for a in ctx.registry.all())
