import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from mlip_research_agent.data.manifests import NormalizedDataset, LabeledConfiguration
from mlip_research_agent.skills.active_learning.random_select.implementation import RandomSelectSkill
from mlip_research_agent.skills.active_learning.random_select.schema import RandomSelectInput
from mlip_research_agent.skills.base import SkillContext, SkillError

@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "pool.json"
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
                energy_ev=0.0,
                forces_ev_per_a=[[0.0, 0.0, 0.0]],
                level_of_theory="mock",
                source_record_sha256="0"*64,
            )
            for i in range(10)
        ]
    )
    dataset_path.write_text(dataset.model_dump_json())
    return dataset_path

def test_random_select_success(mock_dataset, tmp_path):
    skill = RandomSelectSkill()
    registry = MagicMock()
    artifact = MagicMock()
    artifact.artifact_id = "art-1"
    artifact.relative_path = "steps/step-1/random_selection.json"
    registry.register.return_value = artifact
    
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="step-1",
        seed=42,
        attempt=1,
        registry=registry,
    )
    
    inputs = RandomSelectInput(
        dataset_path=mock_dataset.name,
        pool_ids=[f"c_{i}" for i in range(10)],
        budget=3,
        seed=123,
    )
    
    out = skill.run(inputs, ctx)
    
    assert len(out.selected_ids) == 3
    assert out.selection_artifact == "art-1"
    assert "c_" in out.selected_ids[0]
    
    # check idempotency / deterministic
    out2 = skill.run(inputs, ctx)
    assert out.selected_ids == out2.selected_ids

def test_random_select_budget_exceeds(mock_dataset, tmp_path):
    skill = RandomSelectSkill()
    ctx = SkillContext(
        run_dir=tmp_path, step_id="step-1", seed=42, attempt=1, registry=MagicMock()
    )
    
    inputs = RandomSelectInput(
        dataset_path=mock_dataset.name,
        pool_ids=[f"c_{i}" for i in range(2)],
        budget=3,
        seed=123,
    )
    with pytest.raises(SkillError, match="budget \\(3\\) exceeds pool size \\(2\\)"):
        skill.run(inputs, ctx)

def test_random_select_pool_mismatch(mock_dataset, tmp_path):
    skill = RandomSelectSkill()
    ctx = SkillContext(
        run_dir=tmp_path, step_id="step-1", seed=42, attempt=1, registry=MagicMock()
    )
    
    inputs = RandomSelectInput(
        dataset_path=mock_dataset.name,
        pool_ids=["invalid_id"],
        budget=1,
        seed=123,
    )
    with pytest.raises(SkillError, match="pool_ids contains IDs not in dataset"):
        skill.run(inputs, ctx)

def test_random_select_duplicate_pool(mock_dataset, tmp_path):
    skill = RandomSelectSkill()
    ctx = SkillContext(
        run_dir=tmp_path, step_id="step-1", seed=42, attempt=1, registry=MagicMock()
    )
    
    inputs = RandomSelectInput(
        dataset_path=mock_dataset.name,
        pool_ids=["c_1", "c_1"],
        budget=1,
        seed=123,
    )
    with pytest.raises(SkillError, match="pool_ids contains duplicates"):
        skill.run(inputs, ctx)
