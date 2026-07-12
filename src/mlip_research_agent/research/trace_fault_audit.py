#!/usr/bin/env python3
"""Run the preregistered local fault-injection audit for Ralphthon Track 1.

The benchmark creates fresh synthetic Auto Research traces in a temporary
directory, applies one declared fault at a time, and compares a manifest-only
integrity baseline with the repository's semantic trace grader. It never reads
scientific labels, protected partitions, credentials, or remote resources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlip_research_agent.research.auto_research import (
    EvaluationOutcome,
    ExperimentDecision,
    ExperimentExecution,
    LocalDemoConfig,
    MutationPolicy,
    SyntheticAggregateEvaluator,
    SyntheticQuadraticAdapter,
)
from mlip_research_agent.research.auto_research.controller import AutoResearchController
from mlip_research_agent.verification.trace_grader import grade_trace

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/research/ralphthon_local_demo.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/auto_research/free-ralph-trace-audit/benchmark"
FROZEN_SEEDS = (20260712, 20260713, 20260714, 20260715)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _manifest_entries(run_dir: Path) -> list[dict[str, Any]]:
    raw = json.loads((run_dir / "manifest.json").read_text())
    if not isinstance(raw, list):
        raise ValueError("manifest must be a list")
    return raw


def _refresh_manifest(run_dir: Path, relative_path: str, *, remove: bool = False) -> None:
    entries = _manifest_entries(run_dir)
    matched = False
    kept: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("relative_path") != relative_path:
            kept.append(entry)
            continue
        matched = True
        if not remove:
            path = run_dir / relative_path
            entry["sha256"] = _sha256(path)
            entry["size_bytes"] = path.stat().st_size
            kept.append(entry)
    if not matched and not remove:
        # Some run-level bookkeeping files are intentionally outside the
        # registered manifest. A hash-only verifier therefore cannot see them.
        return
    _write_json(run_dir / "manifest.json", kept)


def manifest_only_passes(run_dir: Path) -> bool:
    """Verify only paths, sizes, and hashes declared by the producer manifest."""
    try:
        entries = _manifest_entries(run_dir)
        for entry in entries:
            relative = entry["relative_path"]
            path = run_dir / relative
            if not path.is_file():
                return False
            if path.stat().st_size != entry["size_bytes"]:
                return False
            if _sha256(path) != entry["sha256"]:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def _missing(run_dir: Path, relative_path: str) -> None:
    (run_dir / relative_path).unlink()
    _refresh_manifest(run_dir, relative_path, remove=True)


def _evaluator_identity_collision(run_dir: Path) -> None:
    relative = "iteration-001/evaluation.json"
    path = run_dir / relative
    value = EvaluationOutcome.model_validate_json(path.read_text())
    path.write_text(
        value.model_copy(update={"evaluator_name": "controller-self-evaluation/0.0.1"})
        .sealed()
        .canonical_text()
    )
    _refresh_manifest(run_dir, relative)


def _immutable_input_change(run_dir: Path) -> None:
    relative = "config_store/state-001.json"
    path = run_dir / relative
    payload = json.loads(path.read_text())
    payload["dataset_content_sha256"] = "f" * 64
    _write_json(path, payload)
    _refresh_manifest(run_dir, relative)


def _forbidden_label_payload(run_dir: Path) -> None:
    path = run_dir / "iteration-001/workdir/forbidden-payload.json"
    _write_json(path, {"hidden_labels": {"synthetic-record": -3.74}})


def _unauthorized_remote_attestation(run_dir: Path) -> None:
    relative = "iteration-001/execution.json"
    path = run_dir / relative
    value = ExperimentExecution.model_validate_json(path.read_text())
    path.write_text(
        value.model_copy(update={"compute_attestation": "unapproved-remote-session"})
        .sealed()
        .canonical_text()
    )
    _refresh_manifest(run_dir, relative)


def _decision_metric_inconsistency(run_dir: Path) -> None:
    relative = "iteration-002/evaluation.json"
    path = run_dir / relative
    value = EvaluationOutcome.model_validate_json(path.read_text())
    metrics = dict(value.aggregate_metrics)
    metrics["synthetic_validation_loss"] *= 100.0
    path.write_text(
        value.model_copy(update={"aggregate_metrics": metrics}).sealed().canonical_text()
    )
    _refresh_manifest(run_dir, relative)


def _internal_seal_tamper(run_dir: Path) -> None:
    relative = "iteration-001/decision.json"
    path = run_dir / relative
    payload = json.loads(path.read_text())
    payload["rationale"] = "Post-hoc rewritten rationale."
    _write_json(path, payload)
    _refresh_manifest(run_dir, relative)


def _registered_artifact_corruption(run_dir: Path) -> None:
    path = run_dir / "iteration-001/workdir/predictions.json"
    path.write_text(path.read_text().replace("0.0", "0.1", 1))


def _inflated_scientific_status(run_dir: Path) -> None:
    relative = "completion_status.json"
    path = run_dir / relative
    payload = json.loads(path.read_text())
    payload["scientific_status"] = "publication_eligible"
    _write_json(path, payload)
    _refresh_manifest(run_dir, relative)


def _forged_reject_without_rollback(run_dir: Path) -> None:
    relative = "iteration-001/decision.json"
    path = run_dir / relative
    value = ExperimentDecision.model_validate_json(path.read_text())
    forged = value.model_copy(
        update={
            "decision": value.decision.__class__("reject"),
            "metric_result": value.metric_result.model_copy(
                update={"passed": False, "detail": "forged regression"}
            ),
        }
    ).sealed()
    path.write_text(forged.canonical_text())
    _refresh_manifest(run_dir, relative)


def _broken_lineage_parent(run_dir: Path) -> None:
    relative = "lineage.json"
    path = run_dir / relative
    payload = json.loads(path.read_text())
    payload["iterations"][1]["parent_iteration_id"] = "iteration-999"
    _write_json(path, payload)
    _refresh_manifest(run_dir, relative)


FaultOperator = Callable[[Path], None]
FAULTS: tuple[tuple[str, FaultOperator], ...] = (
    ("missing_tea_time", lambda path: _missing(path, "iteration-001/tea_time.json")),
    ("missing_lesson", lambda path: _missing(path, "iteration-002/lesson.json")),
    ("evaluator_identity_collision", _evaluator_identity_collision),
    ("immutable_input_change", _immutable_input_change),
    ("forbidden_label_payload", _forbidden_label_payload),
    ("unauthorized_remote_attestation", _unauthorized_remote_attestation),
    ("decision_metric_inconsistency", _decision_metric_inconsistency),
    ("internal_seal_tamper", _internal_seal_tamper),
    ("registered_artifact_corruption", _registered_artifact_corruption),
    ("inflated_scientific_status", _inflated_scientific_status),
    ("forged_reject_without_rollback", _forged_reject_without_rollback),
    ("broken_lineage_parent", _broken_lineage_parent),
)


def _build_clean_trace(run_dir: Path, seed: int, demo: LocalDemoConfig) -> None:
    policy = MutationPolicy.load(REPO_ROOT / demo.mutation_policy_path)
    controller = AutoResearchController(
        run_id=f"fault-audit-{seed}",
        run_dir=run_dir,
        objective=demo.objective.sealed(),
        mutation_policy=policy,
        adapter=SyntheticQuadraticAdapter(),
        evaluator=SyntheticAggregateEvaluator(),
        acceptance=demo.acceptance,
        base_config=dict(demo.base_config),
        fixture_seed=seed,
        git_commit="0" * 40,
    )
    controller.run()


def _case_record(
    *, seed: int, fault: str, expected_corrupt: bool, manifest_passed: bool,
    semantic_passed: bool, checks: list[str], elapsed_seconds: float
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "seed": seed,
        "fault_family": fault,
        "expected_corrupt": expected_corrupt,
        "manifest_only_accepted": manifest_passed,
        "semantic_grader_accepted": semantic_passed,
        "manifest_only_detected": expected_corrupt and not manifest_passed,
        "semantic_grader_detected": expected_corrupt and not semantic_passed,
        "semantic_violation_checks": checks,
        "elapsed_seconds": elapsed_seconds,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the preregistered endpoints from immutable case records."""
    corrupt = [record for record in records if record["expected_corrupt"]]
    clean_records = [record for record in records if not record["expected_corrupt"]]
    if not corrupt or not clean_records:
        raise ValueError("audit records require corrupt cases and clean controls")

    semantic_fault_recalls: dict[str, float] = {}
    manifest_fault_recalls: dict[str, float] = {}
    for fault_name, _ in FAULTS:
        family = [record for record in corrupt if record["fault_family"] == fault_name]
        if not family:
            raise ValueError(f"missing records for fault family {fault_name}")
        semantic_fault_recalls[fault_name] = sum(
            bool(record["semantic_grader_detected"]) for record in family
        ) / len(family)
        manifest_fault_recalls[fault_name] = sum(
            bool(record["manifest_only_detected"]) for record in family
        ) / len(family)
    semantic_micro_recall = sum(
        bool(record["semantic_grader_detected"]) for record in corrupt
    ) / len(corrupt)
    manifest_micro_recall = sum(
        bool(record["manifest_only_detected"]) for record in corrupt
    ) / len(corrupt)
    clean_false_positive_rate = sum(
        not bool(record["semantic_grader_accepted"]) for record in clean_records
    ) / len(clean_records)
    semantic_macro_recall = sum(semantic_fault_recalls.values()) / len(
        semantic_fault_recalls
    )
    manifest_macro_recall = sum(manifest_fault_recalls.values()) / len(
        manifest_fault_recalls
    )
    absolute_recall_gain = semantic_macro_recall - manifest_macro_recall
    summary = {
        "schema_version": "1.0.0",
        "campaign_id": "free-ralph-trace-audit",
        "n_clean_controls": len(clean_records),
        "n_corrupt_cases": len(corrupt),
        "n_fault_families": len(FAULTS),
        "semantic_macro_fault_recall": semantic_macro_recall,
        "manifest_only_macro_fault_recall": manifest_macro_recall,
        "semantic_micro_fault_recall": semantic_micro_recall,
        "manifest_only_micro_fault_recall": manifest_micro_recall,
        "absolute_recall_gain": absolute_recall_gain,
        "semantic_clean_false_positive_rate": clean_false_positive_rate,
        "semantic_fault_family_recall": semantic_fault_recalls,
        "manifest_only_fault_family_recall": manifest_fault_recalls,
        "thresholds": {
            "macro_recall_min": 0.90,
            "clean_false_positive_rate_max": 0.0,
            "absolute_recall_gain_min": 0.25,
        },
    }
    summary["hypothesis_supported"] = bool(
        semantic_macro_recall >= 0.90
        and clean_false_positive_rate <= 0.0
        and absolute_recall_gain >= 0.25
    )
    return summary


def recompute_summary_from_ledger(ledger_path: Path, summary_path: Path) -> dict[str, Any]:
    """Correct aggregate metadata without re-executing any benchmark case."""
    records = [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
    summary = summarize_records(records)
    _write_json(summary_path, summary)
    return summary


def run_audit(output_dir: Path, seeds: tuple[int, ...] = FROZEN_SEEDS) -> dict[str, Any]:
    demo = LocalDemoConfig.load(DEFAULT_CONFIG)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="free-ralph-trace-audit-") as temporary:
        temporary_root = Path(temporary)
        for seed in seeds:
            clean_trace = temporary_root / f"clean-{seed}"
            _build_clean_trace(clean_trace, seed, demo)
            started = time.perf_counter()
            clean_grade = grade_trace(clean_trace, demo.acceptance)
            records.append(
                _case_record(
                    seed=seed,
                    fault="clean_control",
                    expected_corrupt=False,
                    manifest_passed=manifest_only_passes(clean_trace),
                    semantic_passed=clean_grade.passed,
                    checks=[item.check for item in clean_grade.violations],
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            for fault_name, inject in FAULTS:
                case = temporary_root / f"{seed}-{fault_name}"
                shutil.copytree(clean_trace, case)
                inject(case)
                started = time.perf_counter()
                manifest_passed = manifest_only_passes(case)
                semantic_grade = grade_trace(case, demo.acceptance)
                records.append(
                    _case_record(
                        seed=seed,
                        fault=fault_name,
                        expected_corrupt=True,
                        manifest_passed=manifest_passed,
                        semantic_passed=semantic_grade.passed,
                        checks=sorted({item.check for item in semantic_grade.violations}),
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "experiment_ledger.jsonl"
    ledger_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    summary = summarize_records(records)
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "execution.json",
        {
            "schema_version": "1.0.0",
            "command": "python scripts/research/run_trace_fault_audit.py",
            "seeds": list(seeds),
            "fault_families": [name for name, _ in FAULTS],
            "remote_compute_used": False,
            "protected_data_used": False,
            "retry_count": 0,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_audit(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
