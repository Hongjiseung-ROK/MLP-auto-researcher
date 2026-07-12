#!/usr/bin/env python3
"""Capture a non-secret live VESSL snapshot and seal the campaign cost card."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.compute.vessl_schemas import (
    VesslCostApproval,
    VesslCostCard,
    VesslVolumeMount,
)


def read_json(executable: Path, arguments: list[str]) -> Any:
    completed = subprocess.run(
        [str(executable), *arguments, "-o", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vesslctl", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/vessl"))
    parser.add_argument("--volume", default="objvol-okbbesmemio8")
    args = parser.parse_args()

    now = datetime.now(UTC).replace(microsecond=0)
    billing = read_json(args.vesslctl, ["billing", "show"])
    resource_specs = read_json(
        args.vesslctl, ["resource-spec", "list", "--usable-only"]
    )
    volume = read_json(args.vesslctl, ["volume", "show", args.volume])
    jobs = read_json(args.vesslctl, ["job", "list"])
    candidates = [
        spec
        for spec in resource_specs
        if spec.get("slug") == "resourcespec-a100x1"
        and spec.get("acceleratorLimit") == 1
        and spec.get("isUsable") is True
    ]
    if len(candidates) != 1:
        raise RuntimeError("live one-A100 resource spec is unavailable or ambiguous")
    spec = candidates[0]
    active_jobs = [
        job
        for job in (jobs or [])
        if str(job.get("status", job.get("state", ""))).lower()
        not in {"succeeded", "failed", "cancelled", "terminated"}
    ]
    if active_jobs:
        raise RuntimeError("a VESSL Job is already active; refuse overlapping spend")
    hourly_price = float(spec["hourlyCostInMicrocredit"]) / 1_000_000.0
    credit = float(billing["balance"])
    planned_jobs = 4
    minutes_per_job = 90
    planned_ceiling_cost = planned_jobs * hourly_price * minutes_per_job / 60.0
    if planned_ceiling_cost > credit:
        raise RuntimeError("planned VESSL Job ceilings exceed live promotional credit")

    snapshot = {
        "schema_version": "1.0.0",
        "observed_at": now.isoformat().replace("+00:00", "Z"),
        "billing": billing,
        "selected_resource_spec": spec,
        "campaign_volume": volume,
        "active_job_count": 0,
        "image": (
            "quay.io/vessl-ai/torch@sha256:"
            "8bd418d49eeed841a607582b614c9d3ba86590b07d1be671816176c47253344a"
        ),
    }
    snapshot_path = args.output_root / "live_snapshots" / "novel-mlip-20260712.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    mount = VesslVolumeMount(
        volume_type="object",
        volume_slug=args.volume,
        mount_path="/campaign",
        read_only=False,
    ).sealed()
    card = VesslCostCard(
        organization="zhang-lab",
        team="Default",
        cluster=str(spec["clusterSlug"]),
        resource_spec_slug=str(spec["slug"]),
        gpu_type=f"{spec['acceleratorName']} {spec['acceleratorOnChipMemoryInGb']}GB",
        gpu_count=1,
        current_hourly_price=hourly_price,
        current_credit=credit,
        currency="CRD",
        image=snapshot["image"],
        expected_max_duration_minutes=minutes_per_job,
        estimated_compute_cost=hourly_price * minutes_per_job / 60.0,
        storage_type="object",
        storage_capacity_gb=5.0,
        storage_hourly_rate=0.0,
        storage_duration_hours=168.0,
        estimated_storage_cost=0.0,
        volume_mounts=[mount],
        timeout_behavior=(
            "The campaign wrapper exits nonzero at 90 minutes and the Job must terminate."
        ),
        cleanup_action=(
            "Verify terminal Job state, pull and hash-check artifacts, then report or remove "
            "the bounded object volume."
        ),
        live_snapshot_sha256=sha256_file(snapshot_path),
        observed_at=now,
    ).sealed()
    card_path = args.output_root / "cost_cards" / "novel-mlip-a100x1-20260712.json"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card.canonical_text())
    approval = VesslCostApproval(
        cost_card_sha256=card.content_sha256,
        approval_artifact_sha256=sha256_file(args.authorization),
        approved_by="repository owner via campaign-specific explicit authorization",
        approved_at=now,
        expires_at=now + timedelta(days=7),
    ).sealed()
    approval_path = (
        args.output_root / "cost_approvals" / "novel-mlip-a100x1-20260712.json"
    )
    approval_path.parent.mkdir(parents=True, exist_ok=True)
    approval_path.write_text(approval.canonical_text())
    assessment = {
        "schema_version": "1.0.0",
        "observed_at": snapshot["observed_at"],
        "live_credit": credit,
        "hourly_price_credit": hourly_price,
        "credit_limited_maximum_gpu_hours": credit / hourly_price,
        "planned_jobs": planned_jobs,
        "planned_ceiling_gpu_hours": planned_jobs * minutes_per_job / 60.0,
        "planned_ceiling_cost_credit": planned_ceiling_cost,
        "credit_reserve_after_all_job_ceilings": credit - planned_ceiling_cost,
        "storage_hard_cap_bytes": 5 * 1024**3,
        "storage_rate_credit_per_hour": 0.0,
        "cost_card_content_sha256": card.content_sha256,
        "cost_approval_content_sha256": approval.content_sha256,
        "authorization_file_sha256": sha256_file(args.authorization),
        "snapshot_file_sha256": sha256_file(snapshot_path),
    }
    assessment_path = args.output_root / "novel_mlip_budget_assessment.json"
    assessment_path.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(assessment_path.read_bytes()).hexdigest()
    print(f"sealed VESSL campaign cost card; budget assessment sha256={digest}")


if __name__ == "__main__":
    main()
