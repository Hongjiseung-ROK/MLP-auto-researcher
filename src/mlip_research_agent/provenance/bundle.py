"""Minimal local provenance bundle for a run.

Reproducibility tolerance: everything except `timestamps`, `command`, and
`git` dirty-state is expected to be identical across reruns with the same
inputs and seed; registered artifact hashes must match exactly.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from mlip_research_agent.artifacts.registry import ArtifactRegistry, sha256_file
from mlip_research_agent.runtime.events import EventLog
from mlip_research_agent.schemas.campaign import CampaignSpec
from mlip_research_agent.schemas.events import EventType
from mlip_research_agent.schemas.workflow import WorkflowSpec

PROVENANCE_NAME = "provenance.json"
DIRECT_DEPENDENCIES = ("numpy", "ase", "pydantic", "PyYAML")


def _git_info(repo_dir: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "available": commit is not None,
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in DIRECT_DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _dependency_lock_hash() -> str:
    entries = sorted(
        f"{dist.metadata['Name']}=={dist.version}".lower() for dist in metadata.distributions()
    )
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def build_provenance(
    run_dir: Path,
    campaign: CampaignSpec,
    workflow: WorkflowSpec,
    command: list[str],
    repo_dir: Path | None = None,
) -> Path:
    repo_dir = repo_dir or Path.cwd()
    registry = ArtifactRegistry.load(run_dir)
    events = EventLog(run_dir, run_id="provenance-reader").replay()

    recovery_actions = [
        {"step_id": e.step_id, **e.payload}
        for e in events
        if e.event_type is EventType.RECOVERY_DECISION
    ]
    approval_events = [
        {"step_id": e.step_id, **e.payload}
        for e in events
        if e.event_type is EventType.APPROVAL_EVENT
    ]
    claims_path = run_dir / "claims.json"
    claims = json.loads(claims_path.read_text()) if claims_path.is_file() else []

    env_file = repo_dir / "environment.yml"
    timestamps = [e.timestamp for e in events]
    bundle = {
        "schema_version": "0.1.0",
        "run_id": events[0].run_id if events else None,
        "git": _git_info(repo_dir),
        "campaign_hash": campaign.content_hash(),
        "workflow_hash": workflow.content_hash(),
        "seed": workflow.seed,
        "model_identifiers": ["mock_mean_baseline"],
        "dataset_identifiers": ["mock-perturbed-bulk (generated in-run)"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "tool_versions": _package_versions(),
        "dependency_lock_hash": _dependency_lock_hash(),
        "environment_file_hash": sha256_file(env_file) if env_file.is_file() else None,
        "command": command,
        "timestamps": {
            "first_event": timestamps[0] if timestamps else None,
            "last_event": timestamps[-1] if timestamps else None,
        },
        "artifacts": {
            "count": len(registry.all()),
            "checksums": {a.artifact_id: a.sha256 for a in registry.all()},
        },
        "recovery_actions": recovery_actions,
        "approval_events": approval_events,
        "claims": claims,
    }
    path = run_dir / PROVENANCE_NAME
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return path
