from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.base import SkillContext, get_skill
from mlip_research_agent.skills.competition.ralphthon_mlip_track1.implementation import (
    RalphthonMLIPTrack1Skill,
)
from mlip_research_agent.skills.competition.ralphthon_mlip_track1.schema import (
    RalphthonMLIPTrack1Input,
    RalphthonMLIPTrack1Output,
)
from mlip_research_agent.skills.competition.ralphthon_mlip_track1.validators import (
    validate_karpathy_training_metrics,
)


def test_bridge_is_registered() -> None:
    assert get_skill("ralphthon_mlip_track1") is RalphthonMLIPTrack1Skill


def test_general_track1_contract_is_non_claim_bearing(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    ctx = SkillContext(tmp_path, "track1", 7, 0, registry)
    output = RalphthonMLIPTrack1Skill().run(
        RalphthonMLIPTrack1Input(
            metrics=["validation_force_component_mae_ev_per_a"],
            use_vessl_compute=True,
        ),
        ctx,
    )
    assert isinstance(output, RalphthonMLIPTrack1Output)
    assert output.required_official_skills == [
        "auto-research",
        "vessl-cloud-onboarding",
    ]
    assert output.claim_eligible is False
    assert output.karpathy_training_used is False
    artifact = registry.get(output.track1_contract_artifact)
    assert artifact is not None
    payload = json.loads((tmp_path / artifact.relative_path).read_text())
    assert payload["scientific_status"] == "infrastructure_only"
    assert payload["h2_open"] and payload["h3_open"] and payload["h5_open"]


def test_karpathy_metric_cannot_enter_mlip_route() -> None:
    with pytest.raises(ValueError, match="val_bpb"):
        RalphthonMLIPTrack1Input(metrics=["val_bpb"])


def test_mlip_metric_cannot_enter_karpathy_training_route() -> None:
    validate_karpathy_training_metrics(["val_bpb"])
    with pytest.raises(ValueError, match="only val_bpb"):
        validate_karpathy_training_metrics(["val_bpb", "force_mae_ev_per_a"])
