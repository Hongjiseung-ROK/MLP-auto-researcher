from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_claude_project_mcp_uses_current_vessl_docs_endpoint() -> None:
    payload = json.loads((REPO_ROOT / ".mcp.json").read_text())
    assert payload == {
        "mcpServers": {
            "vessl-docs": {
                "type": "http",
                "url": "https://docs.cloud.vessl.ai/mcp",
            }
        }
    }


def test_missing_vesslctl_skill_is_recorded_without_credentials() -> None:
    report = json.loads(
        (REPO_ROOT / "artifacts/bootstrap/vesslctl-skill-report.json").read_text()
    )
    assert report["cli_installed"] is False
    assert report["official_skill_installed"] is False
    assert report["official_skill_sha256"] is None
    assert report["token_material_recorded"] is False
    assert report["install_command_pending_confirmation"] == (
        "curl -fsSL https://api.cloud.vessl.ai/cli/install.sh | bash"
    )


def test_vessl_profile_is_dry_run_only_and_non_claim_bearing() -> None:
    profile = yaml.safe_load(
        (REPO_ROOT / "configs/vessl/mlip_infrastructure_replay.yaml").read_text()
    )
    assert profile["execution_status"] == "local_dry_run_only"
    assert profile["scientific_status"] == "infrastructure_only"
    assert profile["claim_eligible"] is False
    assert profile["resource_gate"]["cost_approval_required"] is True
    assert profile["resource_gate"]["vesslctl_job_file_schema_verified"] is False
    assert profile["topology"]["jobs"] == 2
    assert profile["topology"]["sequential"] is True
