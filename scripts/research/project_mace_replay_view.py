#!/usr/bin/env python
"""Project the exact D0+validation replay label view on the trusted host."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlip_research_agent.data.bounded_view import project_bounded_label_view

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--data-steward-authorized",
        action="store_true",
        help="required because the source container also holds protected labels",
    )
    args = parser.parse_args()
    root = REPO_ROOT / "data_registry/datasets/cu_phase2"
    view = project_bounded_label_view(
        dataset_path=root / "normalized_dataset.json",
        normalized_manifest_path=root / "normalized_manifest.json",
        split_manifest_path=root / "split_manifest.json",
        data_steward_authorized=args.data_steward_authorized,
    )
    digest = view.save(args.output)
    print(f"wrote bounded replay view sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
