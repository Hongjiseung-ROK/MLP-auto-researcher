from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_track1_config_pins_official_sources_and_general_route() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / "configs/competition/ralphthon_icml_track1.yaml").read_text()
    )
    assert config["plugin"] == {
        "repository": "https://github.com/team-attention/ralphthon-icml.git",
        "commit": "a9f4f2583648ef4ca54f980f951ae393d153473f",
        "version": "0.5.0",
    }
    assert config["route"] == "general_track_1"
    assert config["karpathy_training_path"]["enabled"] is False
    assert config["karpathy_training_path"]["metric"] == "val_bpb"
    assert config["karpathy_training_path"]["may_mix_with_mlip_metrics"] is False
    assert config["status"]["claim_eligible"] is False
    assert config["status"]["paper_drafting_authorized"] is False
    assert config["wandb"]["online_sync_authorized"] is False


def test_track1_template_is_byte_identical_to_official_asset() -> None:
    official = (
        REPO_ROOT
        / "third_party/ralphthon-icml/skills/auto-research/assets/track-1-submission-template.md"
    )
    project = REPO_ROOT / "paper/track1/track1-submission-template.md"
    assert project.read_bytes() == official.read_bytes()
    digest = hashlib.sha256(project.read_bytes()).hexdigest()
    assert digest == "302bb1fc5aa91ffb7d28ce0d2d08783540c427d33baa503e7a3ca172796ded01"


def test_evidence_map_is_template_only_and_non_claim_bearing() -> None:
    payload = json.loads(
        (REPO_ROOT / "paper/track1/evidence-map.template.json").read_text()
    )
    assert payload["template_only"] is True
    assert payload["evidence_frozen"] is False
    assert payload["claim_eligible"] is False
    assert payload["scientific_status"] == "infrastructure_only"
    assert payload["entries"] == []
