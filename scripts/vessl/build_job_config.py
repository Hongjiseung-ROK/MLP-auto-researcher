#!/usr/bin/env python3
"""Build a sealed normalized VESSL Job dry-run control record."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlip_research_agent.compute.vessl_job_config import build_dry_run_job_config
from mlip_research_agent.compute.vessl_schemas import VesslVolumeMount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["iteration_1", "iteration_2"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--resource-spec", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--volume-slug", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bounded-label-view-sha256", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--environment-lock-sha256", required=True)
    args = parser.parse_args()
    config = build_dry_run_job_config(
        phase=args.phase,
        name=args.name,
        git_commit=args.commit,
        resource_spec_slug=args.resource_spec,
        image=args.image,
        volume_mounts=[
            VesslVolumeMount(
                volume_type="object",
                volume_slug=args.volume_slug,
                mount_path="/runs",
            ).sealed()
        ],
        bounded_label_view_sha256=args.bounded_label_view_sha256,
        dataset_sha256=args.dataset_sha256,
        split_sha256=args.split_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
        environment_lock_sha256=args.environment_lock_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(config.canonical_text())
    print(config.content_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
