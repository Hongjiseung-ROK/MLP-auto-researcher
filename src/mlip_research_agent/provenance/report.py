"""Concise Markdown run report rendered from grounded artifacts only.

The report is a rendering stage over the event log, manifest, verified claims,
and provenance bundle — never a source of scientific truth (plan.md synthesis).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.runtime.events import EventLog
from mlip_research_agent.schemas.claims import Claim, ClaimStatus
from mlip_research_agent.schemas.events import Event, EventType

REPORT_NAME = "run_report.md"


def _step_summary(events: list[Event]) -> list[dict[str, Any]]:
    steps: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.step_id is None:
            continue
        entry = steps.setdefault(
            event.step_id,
            {"step_id": event.step_id, "skill": "", "attempts": 0, "status": "pending"},
        )
        if event.event_type is EventType.STEP_STARTED:
            entry["skill"] = event.payload.get("skill", entry["skill"])
            entry["attempts"] += 1
        elif event.event_type is EventType.STEP_COMPLETED:
            entry["status"] = "completed"
        elif event.event_type is EventType.STEP_FAILED and entry["status"] != "completed":
            entry["status"] = "failed"
        elif event.event_type is EventType.STEP_RESUMED_FROM_CHECKPOINT:
            entry["status"] = "resumed from checkpoint"
    return list(steps.values())


def _load_claims(run_dir: Path) -> list[Claim]:
    verified = sorted(run_dir.glob("steps/*/verified_claims.json"))
    source = verified[0] if verified else run_dir / "claims.json"
    if not source.is_file():
        return []
    return [Claim.model_validate(item) for item in json.loads(source.read_text())]


def generate_run_report(run_dir: Path) -> Path:
    events = EventLog(run_dir, run_id="report-reader").replay()
    registry = ArtifactRegistry.load(run_dir)
    claims = _load_claims(run_dir)
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}

    run_completed = any(e.event_type is EventType.RUN_COMPLETED for e in events)
    status = "completed" if run_completed else "failed / incomplete"
    recoveries = [e for e in events if e.event_type is EventType.RECOVERY_DECISION]

    lines = [
        f"# Run report: {events[0].run_id if events else 'unknown'}",
        "",
        f"- Status: **{status}**",
        f"- Seed: {provenance.get('seed', 'unknown')}",
        f"- Workflow hash: `{provenance.get('workflow_hash', 'unknown')}`",
        f"- Campaign hash: `{provenance.get('campaign_hash', 'unknown')}`",
        f"- Events: {len(events)} | Artifacts: {len(registry.all())}",
        "",
        "## Steps",
        "",
        "| Step | Skill | Attempts | Status |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {s['step_id']} | {s['skill']} | {s['attempts']} | {s['status']} |"
        for s in _step_summary(events)
    ]

    lines += ["", "## Failures and recovery", ""]
    if recoveries:
        for event in recoveries:
            lines.append(
                f"- Step `{event.step_id}`: decision **{event.payload.get('decision')}** "
                f"(failure `{event.payload.get('failure_id')}`, "
                f"repair `{json.dumps(event.payload.get('repair_params', {}))}`)"
            )
    else:
        lines.append("- No failures recorded.")

    lines += ["", "## Verified claims", ""]
    if claims:
        lines += ["| Claim | Value | Units | Status | Backing artifacts |", "|---|---|---|---|---|"]
        for claim in claims:
            refs = ", ".join(f"`{r}`" for r in claim.artifact_references)
            marker = "" if claim.status is ClaimStatus.VERIFIED else " (NOT VERIFIED)"
            lines.append(
                f"| {claim.claim_id} | {claim.value} | {claim.units} | "
                f"{claim.status.value}{marker} | {refs} |"
            )
    else:
        lines.append("- No claims registered.")

    lines += [
        "",
        "## Artifacts",
        "",
        f"- Manifest: `manifest.json` ({len(registry.all())} artifacts, sha256-addressed)",
        "- Provenance bundle: `provenance.json`",
        "- Event log: `events.jsonl`",
        "",
        "## Reproduce",
        "",
        "```bash",
        "conda run -n mlip-research-agent mlip-agent run \\",
        "  --campaign configs/example_campaign.yaml --output-root artifacts/runs",
        "```",
        "",
        "Rerunning with the same campaign file and seed reproduces all registered",
        "artifact hashes exactly; timestamps and run ids differ by design.",
        "",
    ]
    path = run_dir / REPORT_NAME
    path.write_text("\n".join(lines))
    return path
