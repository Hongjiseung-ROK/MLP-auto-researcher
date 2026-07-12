from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "third_party" / "ralphthon-icml"
EXPECTED_COMMIT = "a9f4f2583648ef4ca54f980f951ae393d153473f"
EXPECTED_PLUGIN_VERSION = "0.5.0"
EXPOSED_SKILLS = {
    "auto-research": "skills/auto-research",
    "vessl-cloud-onboarding": "skills/vessl-cloud-onboarding",
    "wandb-onboarding": "skills/wandb-onboarding",
    "world-model-ideation": "skills/world-model-ideation",
    "wandb-track-experiment": "skills/wandb/wandb-track-experiment",
    "weave-add-tracing": "skills/wandb/weave-add-tracing",
    "wandb-project-analyst": "skills/wandb/wandb-project-analyst",
}


def _manifest() -> dict[str, object]:
    value = json.loads((VENDOR_ROOT / "SOURCE_MANIFEST.json").read_text())
    assert isinstance(value, dict)
    return value


def _frontmatter_name(path: Path) -> str:
    text = path.read_text()
    assert text.startswith("---\n")
    _, frontmatter, _ = text.split("---", 2)
    payload = yaml.safe_load(frontmatter)
    assert isinstance(payload, dict)
    name = payload.get("name")
    assert isinstance(name, str)
    return name


def test_exact_upstream_identity_and_plugin_version() -> None:
    manifest = _manifest()
    assert manifest["commit"] == EXPECTED_COMMIT
    assert manifest["plugin_version"] == EXPECTED_PLUGIN_VERSION
    assert manifest["repository"] == "https://github.com/team-attention/ralphthon-icml.git"
    plugin = json.loads((VENDOR_ROOT / ".codex-plugin" / "plugin.json").read_text())
    assert plugin["name"] == "ralphthon-icml"
    assert plugin["version"] == EXPECTED_PLUGIN_VERSION


def test_vendor_snapshot_hashes_are_byte_identical() -> None:
    manifest = _manifest()
    entries = manifest["files"]
    assert isinstance(entries, list) and entries
    registered: set[str] = set()
    for untyped in entries:
        assert isinstance(untyped, dict)
        relative = untyped["path"]
        assert isinstance(relative, str)
        registered.add(relative)
        path = VENDOR_ROOT / relative
        assert path.is_file()
        assert untyped["source_path"] == relative
        assert untyped["provenance"] == "copied_verbatim"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == untyped["sha256"]
    actual = {
        path.relative_to(VENDOR_ROOT).as_posix()
        for path in VENDOR_ROOT.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_MANIFEST.json"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    assert actual == registered


@pytest.mark.parametrize("client_root", [".agents", ".claude"])
def test_project_skill_links_resolve_inside_vendor(client_root: str) -> None:
    vendor_resolved = VENDOR_ROOT.resolve()
    for name, target_relative in EXPOSED_SKILLS.items():
        link = REPO_ROOT / client_root / "skills" / name
        assert link.is_symlink()
        target = link.resolve(strict=True)
        assert target.is_relative_to(vendor_resolved)
        assert target == (VENDOR_ROOT / target_relative).resolve()
        skill_path = target / "SKILL.md"
        assert skill_path.is_file()
        assert _frontmatter_name(skill_path) == name


def test_vendored_plugin_validator_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(VENDOR_ROOT / "scripts" / "validate_plugin.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
