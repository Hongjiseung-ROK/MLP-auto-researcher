#!/usr/bin/env python
"""Verify a pulled-back Colab artifact directory against its own manifests.

Every run directory produced on the VM contains an ``artifact_manifest.json``
(staging) or ``artifact-manifest.json`` (preflight) listing the SHA-256 of
each produced file. This script re-hashes everything locally after
``colab download`` so nothing corrupted or truncated in transit can be cited
as evidence. Exits nonzero on any mismatch, missing file, or a directory
containing no manifest at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MANIFEST_NAMES = ("artifact_manifest.json", "artifact-manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text())
    entries = payload.get("files", payload if isinstance(payload, list) else [])
    problems: list[str] = []
    base = manifest_path.parent
    for entry in entries:
        relative = entry.get("relative_path") or entry.get("path")
        expected = entry.get("sha256")
        if not relative or not expected:
            problems.append(f"{manifest_path}: malformed entry {entry!r}")
            continue
        target = base / relative
        if not target.is_file():
            problems.append(f"missing file: {target}")
        elif sha256_file(target) != expected:
            problems.append(f"hash mismatch: {target}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pullback_dir", type=Path)
    args = parser.parse_args()

    manifests = [
        path
        for name in MANIFEST_NAMES
        for path in sorted(args.pullback_dir.rglob(name))
    ]
    if not manifests:
        print(f"error: no artifact manifest found under {args.pullback_dir}", file=sys.stderr)
        return 1
    problems: list[str] = []
    verified = 0
    for manifest in manifests:
        found = verify_manifest(manifest)
        problems.extend(found)
        if not found:
            verified += 1
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"verified {verified} manifest(s) under {args.pullback_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
