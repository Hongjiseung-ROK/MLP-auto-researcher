import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlip_research_agent.data.manifests import NormalizedDataset, LabeledConfiguration
from mlip_research_agent.skills.active_learning.ensemble_uq.implementation import EnsembleUqSkill
from mlip_research_agent.skills.active_learning.ensemble_uq.schema import EnsembleUqInput
from mlip_research_agent.skills.base import SkillContext, SkillError

def generate_mock_dataset(path: Path, offset: float) -> Path:
    dataset = NormalizedDataset(
        dataset_id="mock-dataset",
        level_of_theory="mock",
        configurations=[
            LabeledConfiguration(
                config_id=f"c_{i}",
                source_id="mock",
                top_group="mock",
                group_id="mock",
                split_unit_id="mock",
                symbols=["Cu"],
                positions=[[0, 0, 0]],
                cell=[[1,0,0],[0,1,0],[0,0,1]],
                pbc=[True, True, True],
                energy_ev=offset * i,
                forces_ev_per_a=[[offset * i, 0.0, 0.0]],
                level_of_theory="mock",
                source_record_sha256="0"*64,
            )
            for i in range(10)
        ]
    )
    path.write_text(dataset.model_dump_json())
    return path

@pytest.fixture
def mock_datasets(tmp_path: Path) -> list[Path]:
    paths = []
    for m in range(3):
        p = tmp_path / f"member_{m}.json"
        generate_mock_dataset(p, offset=float(m))
        paths.append(p)
    return paths

def test_ensemble_uq_success(mock_datasets, tmp_path):
    skill = EnsembleUqSkill()
    registry = MagicMock()
    artifact = MagicMock()
    artifact.artifact_id = "art-1"
    artifact.relative_path = "steps/step-1/ensemble_uq.json"
    registry.register.return_value = artifact
    
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="step-1",
        seed=42,
        attempt=1,
        registry=registry,
    )
    
    inputs = EnsembleUqInput(
        dataset_path="pool.json", # not strictly used but needed for schema
        pool_ids=[f"c_{i}" for i in range(10)],
        member_identities=["m1", "m2", "m3"],
        member_prediction_paths=[p.name for p in mock_datasets],
        aggregation_formula="mean_std_dev",
        include_energy_disagreement=False,
    )
    
    out = skill.run(inputs, ctx)
    
    assert len(out.ranked_ids) == 10
    # higher i has higher offset spread -> higher disagreement
    assert out.ranked_ids[0] == "c_9"
    assert out.ranked_ids[-1] == "c_0"

def test_ensemble_uq_mismatch_members(tmp_path):
    skill = EnsembleUqSkill()
    ctx = SkillContext(run_dir=tmp_path, step_id="step-1", seed=42, attempt=1, registry=MagicMock())
    
    inputs = EnsembleUqInput(
        dataset_path="pool.json",
        pool_ids=["c_1"],
        member_identities=["m1", "m2", "m3"],
        member_prediction_paths=["p1", "p2", "p3", "p4"], # Mismatch
        aggregation_formula="mean_std_dev",
    )
    
    with pytest.raises(SkillError, match="must have the same length"):
        skill.run(inputs, ctx)
