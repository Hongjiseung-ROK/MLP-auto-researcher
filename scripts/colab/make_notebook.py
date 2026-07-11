#!/usr/bin/env python
"""Generate notebooks/colab_preflight.ipynb deterministically.

The notebook is checked-in generated output, not hand-edited: run this
script to (re)write it, or with `--check` to verify the on-disk file still
matches what this script would generate (byte-for-byte), which is what CI
should call to catch drift. No `nbformat` package dependency is needed —
the JSON structure is written directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "colab_preflight.ipynb"
REPO_URL = "https://github.com/Hongjiseung-ROK/MLP-auto-researcher.git"


def _markdown_cell(cell_id: str, source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code_cell(cell_id: str, source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict[str, Any]:
    title_cell = _markdown_cell(
        "title",
        "# MLIP Research Agent — Colab GPU Preflight\n"
        "\n"
        "Runs this repo's Colab bootstrap and preflight battery against an exact "
        "commit on a fresh Colab GPU runtime. See `scripts/colab/bootstrap.sh` and "
        "`scripts/colab/preflight.py` for what each step does.\n"
        "\n"
        "This notebook reads or stores no credentials, and never invokes the "
        "`colab` CLI.",
    )

    parameters_cell = _code_cell(
        "parameters",
        '# Set this to the exact commit SHA you want to test (never a branch\n'
        '# name or "HEAD" — preflight exists to validate one reproducible commit).\n'
        '# Example: COMMIT_SHA = "ede6d8ea37d0e6e1b849b96ffb464dbdc640bc2d"\n'
        'COMMIT_SHA = ""\n'
        "\n"
        'assert COMMIT_SHA, (\n'
        '    "Set COMMIT_SHA to a full git commit SHA before running the next cell."\n'
        ")\n"
        'assert COMMIT_SHA not in {"main", "HEAD", "master"}, (\n'
        '    "COMMIT_SHA must be an exact commit SHA, not a branch name."\n'
        ")\n",
    )

    bootstrap_cell = _code_cell(
        "bootstrap",
        f"!git clone {REPO_URL} /content/MLP-auto-researcher\n"
        "!bash /content/MLP-auto-researcher/scripts/colab/bootstrap.sh $COMMIT_SHA\n",
    )

    report_cell = _code_cell(
        "report",
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        'artifacts_root = Path("/content/MLP-auto-researcher/artifacts/colab")\n'
        'report_paths = sorted(artifacts_root.glob("*/test-report.json"))\n'
        "assert report_paths, (\n"
        '    "No test-report.json found; bootstrap must have failed before writing artifacts."\n'
        ")\n"
        "latest = report_paths[-1]\n"
        'print(f"Reading: {latest}")\n'
        "print(json.dumps(json.loads(latest.read_text()), indent=2, sort_keys=True))\n",
    )

    cleanup_cell = _markdown_cell(
        "cleanup",
        "## Download your artifacts before the VM dies\n"
        "\n"
        "Colab VMs are ephemeral: once this runtime disconnects or recycles, "
        "everything under `/content` is gone. Before you close this notebook, "
        "download `/content/MLP-auto-researcher/artifacts/colab/` (zip it and use "
        "the Colab file browser, or `google.colab.files.download`). The "
        "`attestation.json`, `test-report.json`, `provenance.json`, and "
        "`artifact-manifest.json` files under the run directory are the pieces "
        "worth keeping.\n",
    )

    return {
        "cells": [title_cell, parameters_cell, bootstrap_cell, report_cell, cleanup_cell],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_text() -> str:
    return json.dumps(build_notebook(), indent=1, sort_keys=False) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the on-disk notebook matches generated output; exit nonzero on drift.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generated = notebook_text()

    if args.check:
        if not NOTEBOOK_PATH.is_file():
            print(f"drift: {NOTEBOOK_PATH} does not exist", file=sys.stderr)
            return 1
        existing = NOTEBOOK_PATH.read_text()
        if existing != generated:
            print(f"drift: {NOTEBOOK_PATH} does not match generated output", file=sys.stderr)
            return 1
        print(f"{NOTEBOOK_PATH} matches generated output")
        return 0

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(generated)
    print(f"wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
