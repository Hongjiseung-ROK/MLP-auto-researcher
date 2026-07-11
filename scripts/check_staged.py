#!/usr/bin/env python
"""Pre-publication gate: scan staged files for secrets and oversized blobs.

Usage: python scripts/check_staged.py [--max-bytes 1048576] [--tracked]
`--tracked` scans every tracked file at HEAD instead of the staged index —
that is the CI mode, where nothing is staged.
Exit codes: 0 clean, 1 findings, 2 git error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("github token", re.compile(r"\bgh[opsu]_[A-Za-z0-9]{16,}\b")),
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("google oauth token", re.compile(r"\bya29\.[A-Za-z0-9_-]{16,}\b")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "assigned credential",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|password|access[_-]?token|auth[_-]?token)\b"
            r"\s*[=:]\s*['\"][^'\"\s]{8,}['\"]"
        ),
    ),
]
# Documentation/tests legitimately mention placeholder patterns; skip them.
ALLOWLIST_SUBSTRINGS = ("[REDACTED]", "supersecret123", "example", "PLACEHOLDER")


def _git_listing(command: list[str]) -> list[str]:
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(2)
    return [line for line in proc.stdout.splitlines() if line.strip()]


def staged_files() -> list[str]:
    return _git_listing(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    )


def tracked_files() -> list[str]:
    return _git_listing(["git", "ls-files"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-bytes", type=int, default=1_048_576)
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="scan all tracked files at HEAD (CI mode) instead of the staged index",
    )
    args = parser.parse_args()

    revision_prefix = "HEAD:" if args.tracked else ":"
    findings: list[str] = []
    for path in tracked_files() if args.tracked else staged_files():
        show = subprocess.run(
            ["git", "show", f"{revision_prefix}{path}"], capture_output=True
        )
        if show.returncode != 0:
            continue
        blob = show.stdout
        if len(blob) > args.max_bytes:
            findings.append(f"LARGE FILE: {path} ({len(blob)} bytes > {args.max_bytes})")
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            findings.append(f"BINARY FILE staged: {path} ({len(blob)} bytes) — verify intent")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(marker in line for marker in ALLOWLIST_SUBSTRINGS):
                continue
            for label, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    findings.append(f"POSSIBLE SECRET ({label}): {path}:{lineno}")

    if findings:
        print("check_staged: FINDINGS")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print("check_staged: clean (no secrets, no oversized files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
