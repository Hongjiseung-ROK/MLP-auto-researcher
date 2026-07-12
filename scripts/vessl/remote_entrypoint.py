#!/usr/bin/env python3
"""Remote exact-commit VESSL Job entrypoint; never invoked by local tests."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from mlip_research_agent.compute.vessl_replay import (
    build_artifact_manifest,
    verify_artifact_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        shell=False,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"remote command failed: {command!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_environment() -> None:
    expected = os.environ["MLIP_REPLAY_COMMIT"]
    if len(expected) != 40 or _run(["git", "rev-parse", "HEAD"]) != expected:
        raise RuntimeError("remote checkout is not the approved exact commit")
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=REPO_ROOT,
        shell=False,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if symbolic.returncode == 0 or _run(["git", "status", "--porcelain"]):
        raise RuntimeError("remote checkout must be clean and detached")


def attest_gpu() -> None:
    names = [
        line.strip()
        for line in _run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).splitlines()
        if line.strip()
    ]
    if len(names) != 1 or "a100" not in names[0].casefold():
        raise RuntimeError("VESSL MLIP Job requires exactly one observed A100")


def verify_input_hashes() -> None:
    root = Path(os.environ["MLIP_REPLAY_INPUT_ROOT"])
    verify_artifact_manifest(root, root / "input_manifest.json")


def run_phase(phase: str) -> None:
    command = [
        sys.executable,
        "scripts/research/run_mace_auto_research.py",
        "iteration1" if phase == "iteration_1" else "iteration2",
        "--commit",
        os.environ["MLIP_REPLAY_COMMIT"],
        "--run-id",
        os.environ["MLIP_REPLAY_RUN_ID"],
        "--label-view",
        os.environ["MLIP_REPLAY_LABEL_VIEW"],
        "--label-view-sha256",
        os.environ["MLIP_REPLAY_LABEL_VIEW_SHA256"],
    ]
    if phase == "iteration_2":
        command.extend(["--review-bundle", os.environ["MLIP_REPLAY_REVIEW_BUNDLE"]])
    _run(command)


def write_manifest() -> None:
    output = Path(os.environ["MLIP_REPLAY_OUTPUT_ROOT"])
    build_artifact_manifest(output, output / "artifact_manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "verify-environment",
            "attest-gpu",
            "verify-input-hashes",
            "run-phase",
            "write-artifact-manifest",
        ],
    )
    parser.add_argument("phase", nargs="?", choices=["iteration_1", "iteration_2"])
    args = parser.parse_args()
    if args.action == "verify-environment":
        verify_environment()
    elif args.action == "attest-gpu":
        attest_gpu()
    elif args.action == "verify-input-hashes":
        verify_input_hashes()
    elif args.action == "run-phase":
        if args.phase is None:
            parser.error("run-phase requires an iteration")
        run_phase(args.phase)
    else:
        write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
