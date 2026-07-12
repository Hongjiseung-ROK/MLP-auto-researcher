#!/usr/bin/env python3
"""Hash-verify a fresh VESSL pull-back staging tree and publish atomically."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlip_research_agent.compute.vessl_replay import atomic_publish


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", default="artifact_manifest.json")
    args = parser.parse_args()
    atomic_publish(args.staging, args.destination, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
