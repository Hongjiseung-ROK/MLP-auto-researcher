import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlip_research_agent.skills.active_learning.diversity_select.implementation import DiversitySelectSkill
from mlip_research_agent.skills.active_learning.diversity_select.schema import DiversitySelectInput
from mlip_research_agent.skills.base import SkillContext, SkillError

@pytest.fixture
def mock_descriptors(tmp_path: Path) -> Path:
    # 2 clusters:
    # A at [0, 0] (c_0, c_1)
    # B at [10, 10] (c_2, c_3)
    descriptors = {
        "c_0": [0.0, 0.0],
        "c_1": [0.1, 0.1],
        "c_2": [10.0, 10.0],
        "c_3": [10.1, 10.1],
    }
    desc_path = tmp_path / "descriptors.json"
    desc_path.write_text(json.dumps(descriptors))
    return desc_path


def test_diversity_select_success(mock_descriptors, tmp_path):
    skill = DiversitySelectSkill()
    registry = MagicMock()
    artifact = MagicMock()
    artifact.artifact_id = "art-1"
    artifact.relative_path = "steps/step-1/diversity_selection.json"
    registry.register.return_value = artifact
    
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="step-1",
        seed=42,
        attempt=1,
        registry=registry,
    )
    
    inputs = DiversitySelectInput(
        dataset_path="pool.json", # not strictly used
        pool_ids=["c_0", "c_1", "c_2", "c_3"],
        budget=2,
        seed=123,
        descriptor_path=mock_descriptors.name,
    )
    
    out = skill.run(inputs, ctx)
    
    assert len(out.selected_ids) == 2
    
    # Selecting 2 items from 2 far apart clusters should yield one from each cluster
    selected = set(out.selected_ids)
    has_cluster_a = "c_0" in selected or "c_1" in selected
    has_cluster_b = "c_2" in selected or "c_3" in selected
    assert has_cluster_a
    assert has_cluster_b
    
    # Tie-breaking determinism check
    out2 = skill.run(inputs, ctx)
    assert out.selected_ids == out2.selected_ids


def test_diversity_select_duplicates(mock_descriptors, tmp_path):
    skill = DiversitySelectSkill()
    ctx = SkillContext(run_dir=tmp_path, step_id="step-1", seed=42, attempt=1, registry=MagicMock())
    
    inputs = DiversitySelectInput(
        dataset_path="pool.json",
        pool_ids=["c_0", "c_0"],
        budget=1,
        seed=123,
        descriptor_path=mock_descriptors.name,
    )
    with pytest.raises(SkillError, match="duplicates"):
        skill.run(inputs, ctx)


def test_diversity_select_missing_descriptor(mock_descriptors, tmp_path):
    skill = DiversitySelectSkill()
    ctx = SkillContext(run_dir=tmp_path, step_id="step-1", seed=42, attempt=1, registry=MagicMock())
    
    inputs = DiversitySelectInput(
        dataset_path="pool.json",
        pool_ids=["c_0", "c_999"], # c_999 is missing
        budget=2,
        seed=123,
        descriptor_path=mock_descriptors.name,
    )
    with pytest.raises(SkillError, match="missing from descriptors"):
        skill.run(inputs, ctx)
