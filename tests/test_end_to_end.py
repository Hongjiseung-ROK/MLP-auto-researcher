"""Mock end-to-end campaign execution and reproducibility replay."""

import json
from pathlib import Path

from mlip_research_agent.cli import main
from mlip_research_agent.runtime.events import EventLog
from mlip_research_agent.schemas.events import EventType
from tests.conftest import EXAMPLE_CAMPAIGN


def run_campaign(output_root: Path) -> Path:
    exit_code = main(
        [
            "run",
            "--campaign",
            str(EXAMPLE_CAMPAIGN),
            "--output-root",
            str(output_root),
        ]
    )
    assert exit_code == 0
    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1
    return run_dirs[0]


def test_full_campaign_with_recovery(tmp_path: Path) -> None:
    run_dir = run_campaign(tmp_path / "runs")

    events = EventLog(run_dir, run_id="reader").replay()
    types = [e.event_type for e in events]
    assert types[-1] is EventType.RUN_COMPLETED

    # Exactly one injected recoverable failure, refined and recovered.
    failed = [e for e in events if e.event_type is EventType.STEP_FAILED]
    assert len(failed) == 1
    assert failed[0].step_id == "labeling"
    recoveries = [e for e in events if e.event_type is EventType.RECOVERY_DECISION]
    assert len(recoveries) == 1
    assert recoveries[0].payload["decision"] == "refine"
    assert recoveries[0].payload["repair_params"] == {"scf_damping": 0.7}

    # All required outputs exist.
    required = ("events.jsonl", "manifest.json", "provenance.json", "claims.json", "run_report.md")
    for name in required:
        assert (run_dir / name).is_file(), f"missing {name}"

    # The repair was recorded in the labels artifact, not applied silently.
    labels = json.loads((run_dir / "steps" / "labeling" / "labels.json").read_text())
    assert labels["settings"]["scf_damping"] == 0.7

    # The claim survived verification.
    verified = json.loads(
        (run_dir / "steps" / "verification" / "verified_claims.json").read_text()
    )
    assert len(verified) == 1
    assert verified[0]["status"] == "verified"

    # Provenance records the recovery and the claim.
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["recovery_actions"][0]["decision"] == "refine"
    assert provenance["claims"][0]["claim_id"] == "energy_mae_holdout"
    assert provenance["seed"] == 42

    # Report mentions the recovery and the verified claim.
    report = (run_dir / "run_report.md").read_text()
    assert "refine" in report
    assert "energy_mae_holdout" in report


def test_rerun_reproduces_artifact_hashes(tmp_path: Path) -> None:
    run_a = run_campaign(tmp_path / "a")
    run_b = run_campaign(tmp_path / "b")

    manifest_a = json.loads((run_a / "manifest.json").read_text())
    manifest_b = json.loads((run_b / "manifest.json").read_text())
    hashes_a = {item["artifact_id"]: item["sha256"] for item in manifest_a}
    hashes_b = {item["artifact_id"]: item["sha256"] for item in manifest_b}
    assert hashes_a == hashes_b
    # Documented tolerance: run ids and timestamps differ; content hashes do not.
    assert (run_a / "claims.json").read_text() == (run_b / "claims.json").read_text()
