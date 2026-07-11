"""Download the pinned mlearn Cu benchmark files with checksum verification.

Usage:
    python scripts/data/fetch_cu_benchmark.py [--force]

Sources, pinned commit, and expected SHA-256 hashes come from
docs/research/dataset_due_diligence_cu.json (the due-diligence record).
Any checksum mismatch aborts and deletes the offending download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DUE_DILIGENCE = REPO_ROOT / "docs" / "research" / "dataset_due_diligence_cu.json"
MAX_ATTEMPTS = 3
RETRY_DELAY_S = 2.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_sha256: str, *, force: bool) -> str:
    if dest.is_file() and not force:
        actual = sha256_file(dest)
        if actual == expected_sha256:
            return "already present, checksum OK"
        dest.unlink()  # stale/corrupt copy: refuse to keep it
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                dest.write_bytes(response.read())
            break
        except OSError as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_S * attempt)
    else:
        raise SystemExit(f"ABORT: download failed after {MAX_ATTEMPTS} attempts: {last_error}")
    actual = sha256_file(dest)
    if actual != expected_sha256:
        dest.unlink()
        raise SystemExit(
            f"ABORT: checksum mismatch for {url}\n  expected {expected_sha256}\n  got      {actual}"
        )
    return "downloaded, checksum OK"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    facts = json.loads(DUE_DILIGENCE.read_text())
    for record in facts["files"]:
        dest = REPO_ROOT / record["path"]
        status = fetch(record["url"], dest, record["sha256"], force=args.force)
        print(f"{record['path']}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
