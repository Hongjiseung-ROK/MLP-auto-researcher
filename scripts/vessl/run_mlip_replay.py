#!/usr/bin/env python3
"""Host-side VESSL MLIP replay planner; billable execution is disabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlip_research_agent.compute.vessl_job_config import build_dry_run_job_config
from mlip_research_agent.compute.vessl_schemas import VesslVolumeMount

DATASET_SHA256 = "bc9b78ca5e3b95f91bf34bbc3641a3d6e3f92338b4e3d97065165157848cfc48"
SPLIT_SHA256 = "80d9b95083ebba9d8f988a461525b326ba807c79da834b066d1917afc9d8a15e"
VIEW_SHA256 = "90ef76ca4c927401ac59ee645236af23b6d49d229ec9da53cae0f16dd27e2be7"
CHECKPOINT_SHA256 = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"


def build_plan(args: argparse.Namespace) -> dict[str, object]:
    mount = VesslVolumeMount(
        volume_type="object",
        volume_slug=args.volume_slug,
        mount_path="/runs",
    ).sealed()
    jobs = [
        build_dry_run_job_config(
            phase=phase,
            name=f"{args.run_id}-{index}",
            git_commit=args.commit,
            resource_spec_slug=args.resource_spec,
            image=args.image,
            volume_mounts=[mount],
            bounded_label_view_sha256=VIEW_SHA256,
            dataset_sha256=DATASET_SHA256,
            split_sha256=SPLIT_SHA256,
            checkpoint_sha256=CHECKPOINT_SHA256,
            environment_lock_sha256=args.environment_lock_sha256,
        )
        for index, phase in enumerate(("iteration_1", "iteration_2"), start=1)
    ]
    return {
        "schema_version": "1.0.0",
        "run_id": args.run_id,
        "execution": "blocked_local_dry_run",
        "reason": "vesslctl Job file schema and sealed live cost approval are absent",
        "jobs": [job.model_dump(mode="json") for job in jobs],
        "host_pause": "reviews_and_sealed_iteration_2_proposal",
        "scientific_status": "infrastructure_only",
        "claim_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--resource-spec", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--volume-slug", required=True)
    parser.add_argument("--environment-lock-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
