import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlip_research_agent.data.manifests import NormalizedDataset, LabeledConfiguration
from mlip_research_agent.skills.active_learning.decision_gate.implementation import DecisionGateSkill
from mlip_research_agent.skills.active_learning.decision_gate.schema import DecisionGateInput
from mlip_research_agent.skills.base import SkillContext, SkillError

@pytest.fixture
def setup_files(tmp_path: Path):
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
                source_record_sha256=f"{i:064d}",
            )
            for i in range(20)
        ]
    )
    dataset_path.write_text(dataset.model_dump_json())
    
    uq_path = tmp_path / "uq.json"
    # c_0 to c_19, c_19 is most uncertain
    uq_scores = {f"c_{i}": float(i) for i in range(20)}
    uq_path.write_text(json.dumps(uq_scores))
    
    desc_path = tmp_path / "desc.json"
    descriptors = {f"c_{i}": [float(i), float(i)] for i in range(20)}
    desc_path.write_text(json.dumps(descriptors))
    
    return dataset_path, uq_path, desc_path

def test_decision_gate_success(setup_files, tmp_path):
    dataset_path, uq_path, desc_path = setup_files
    
    skill = DecisionGateSkill()
    registry = MagicMock()
    artifact = MagicMock()
    artifact.artifact_id = "art-1"
    artifact.relative_path = "steps/step-1/decision_gate.json"
    registry.register.return_value = artifact
    
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="step-1",
        seed=42,
        attempt=1,
        registry=registry,
    )
    
    inputs = DecisionGateInput(
        dataset_path=dataset_path.name,
        pool_ids=[f"c_{i}" for i in range(20)],
        budget=2,
        seed=123,
        policy_identity="test_policy",
        uncertainty_scores_path=uq_path.name,
        descriptors_path=desc_path.name,
    )
    
    out = skill.run(inputs, ctx)
    
    # top 2 * 5 = 10 filtered in. (c_19 down to c_10)
    assert len(out.selected_ids) == 2
    assert len(out.rejected_ids) == 18
    # Source hashes correctly mapped
    for c_id in out.selected_ids:
        assert c_id in out.source_hashes
        assert c_id in out.reasons
        
    for c_id in out.rejected_ids:
        assert "rejected" in out.reasons[c_id]
        
    assert len(out.reasons) == 20
