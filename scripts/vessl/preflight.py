#!/usr/bin/env python3
"""Run only sanitized, read-only current-Cloud vesslctl diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from mlip_research_agent.compute.vessl_cli_transport import (
    VesslCliError,
    VesslCliNotInstalled,
    VesslCliTransport,
)
from mlip_research_agent.compute.vessl_cloud import VesslCloudProvider


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_preflight(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    transport = VesslCliTransport()
    provider = VesslCloudProvider(transport)
    try:
        version = provider.version()
    except VesslCliNotInstalled:
        _write(
            output / "cli_version.json",
            {
                "installed": False,
                "install_requires_confirmation": True,
                "install_command": "curl -fsSL https://api.cloud.vessl.ai/cli/install.sh | bash",
            },
        )
        return 2
    try:
        payloads = {
            "cli_version.json": {"installed": True, "version": version},
            "account_context.json": {
                "auth": provider.auth_status(),
                "config": provider.config(),
                "organizations": provider.organizations(),
                "teams": provider.teams(),
            },
            "billing_snapshot.json": provider.billing(),
            "resource_inventory.json": {
                "clusters": provider.clusters(),
                "resource_specs": provider.resource_specs(),
            },
            "storage_inventory.json": {
                "storage": provider.storage(),
                "volumes": provider.volumes(),
            },
            "cost_card_candidates.json": {
                "status": "requires_versioned_live_decoder",
                "candidates": [],
                "reason": "No live JSON field map has been verified for this CLI version.",
            },
        }
    except VesslCliError as exc:
        _write(output / "preflight_error.json", {"error": str(exc)})
        return 3
    for name, payload in payloads.items():
        _write(output / name, payload)
    files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "relative_path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
    _write(
        output / "manifest.json",
        {
            "schema_version": "1.0.0",
            "observed_at": datetime.now(UTC).isoformat(),
            "files": files,
            "scientific_status": "infrastructure_only",
            "claim_eligible": False,
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return run_preflight(parser.parse_args().output)


if __name__ == "__main__":
    raise SystemExit(main())
