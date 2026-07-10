"""Machine-readable bootstrap environment report (plan bootstrap §2)."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.provenance.bundle import DIRECT_DEPENDENCIES

DEV_DEPENDENCIES = ("pytest", "ruff", "mypy")
EXTERNAL_TOOLS = ("git", "conda", "docker", "orca", "lmp", "pw.x")


def _detect_cuda() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {"available": False, "reason": "nvidia-smi not found on PATH"}
    try:
        proc = subprocess.run([nvidia_smi, "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"nvidia-smi failed: {exc}"}
    if proc.returncode != 0:
        return {"available": False, "reason": proc.stderr.strip() or "nvidia-smi error"}
    return {"available": True, "devices": proc.stdout.strip().splitlines()}


def generate_environment_report(
    output_path: Path,
    environment_name: str = "mlip-research-agent",
    environment_file: Path | None = None,
    commands_executed: list[str] | None = None,
) -> Path:
    installed: dict[str, str] = {}
    for name in (*DIRECT_DEPENDENCIES, *DEV_DEPENDENCIES):
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = "not-installed"

    tools: dict[str, Any] = {}
    for tool in EXTERNAL_TOOLS:
        path = shutil.which(tool)
        tools[tool] = {"available": path is not None, "path": path}
    failed_tools = sorted(t for t, info in tools.items() if not info["available"])

    report = {
        "schema_version": "0.1.0",
        "environment_name": environment_name,
        "python_version": sys.version.split()[0],
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "cuda": _detect_cuda(),
        "installed_direct_dependencies": installed,
        "external_tools": tools,
        "failed_or_unavailable_tools": failed_tools,
        "commands_executed": commands_executed or [],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment_file_hash": (
            sha256_file(environment_file)
            if environment_file is not None and environment_file.is_file()
            else None
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output_path
