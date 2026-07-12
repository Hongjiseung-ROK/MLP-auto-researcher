#!/usr/bin/env python3
"""Derive paper tables from the frozen trace-audit and real-MACE evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = REPO_ROOT / "artifacts/auto_research/free-ralph-trace-audit/benchmark"
DEFAULT_REAL_TRACE = REPO_ROOT / "artifacts/auto_research/ralphthon-mace-ec2d804"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/auto_research/free-ralph-trace-audit/analysis"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def build_analysis(benchmark: Path, real_trace: Path, output: Path) -> dict[str, Any]:
    ledger_path = benchmark / "experiment_ledger.jsonl"
    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    corrupt = [record for record in records if record["expected_corrupt"]]
    clean = [record for record in records if not record["expected_corrupt"]]
    methods: dict[str, dict[str, int | float]] = {
        "Manifest only": {
            "detected": sum(bool(record["manifest_only_detected"]) for record in corrupt),
            "clean_rejected": sum(not bool(record["manifest_only_accepted"]) for record in clean),
        },
        "Semantic grader": {
            "detected": sum(bool(record["semantic_grader_detected"]) for record in corrupt),
            "clean_rejected": sum(not bool(record["semantic_grader_accepted"]) for record in clean),
        },
    }
    fault_families = sorted({str(record["fault_family"]) for record in corrupt})
    for method, detector_key in (
        ("Manifest only", "manifest_only_detected"),
        ("Semantic grader", "semantic_grader_detected"),
    ):
        family_recalls = []
        for fault_family in fault_families:
            family = [
                record for record in corrupt if record["fault_family"] == fault_family
            ]
            family_recalls.append(
                sum(bool(record[detector_key]) for record in family) / len(family)
            )
        methods[method]["macro_recall"] = sum(family_recalls) / len(family_recalls)
        methods[method]["micro_recall"] = methods[method]["detected"] / len(corrupt)
    output.mkdir(parents=True, exist_ok=True)
    recall_csv = output / "fault_detection_summary.csv"
    with recall_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "detected",
                "corrupt_total",
                "recall",
                "clean_rejected",
                "clean_total",
                "clean_false_positive_rate",
            ],
        )
        writer.writeheader()
        for method, values in methods.items():
            writer.writerow(
                {
                    "method": method,
                    "detected": values["detected"],
                    "corrupt_total": len(corrupt),
                    "recall": values["micro_recall"],
                    "clean_rejected": values["clean_rejected"],
                    "clean_total": len(clean),
                    "clean_false_positive_rate": values["clean_rejected"] / len(clean),
                }
            )

    seed_csv = output / "seed_level_fault_detection.csv"
    seeds = sorted({int(record["seed"]) for record in records})
    with seed_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "method", "detected", "corrupt_total", "recall"],
        )
        writer.writeheader()
        for seed in seeds:
            seed_records = [record for record in corrupt if int(record["seed"]) == seed]
            for method, detector_key in (
                ("Manifest only", "manifest_only_detected"),
                ("Semantic grader", "semantic_grader_detected"),
            ):
                detected = sum(bool(record[detector_key]) for record in seed_records)
                writer.writerow(
                    {
                        "seed": seed,
                        "method": method,
                        "detected": detected,
                        "corrupt_total": len(seed_records),
                        "recall": detected / len(seed_records),
                    }
                )

    case_csv = output / "real_mace_case_study.csv"
    with case_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "relative_improvement",
                "metric_passed",
                "rerun_metric_delta",
                "reproducibility_passed",
                "decision",
            ],
        )
        writer.writeheader()
        for iteration in ("iteration-001", "iteration-002"):
            decision = _read_json(real_trace / iteration / "decision.json")
            evaluation = _read_json(real_trace / iteration / "evaluation.json")
            candidate = float(evaluation["aggregate_metrics"]["force_component_mae_ev_per_a"])
            baseline = float(
                evaluation["baseline_reference"]["metrics"]["force_component_mae_ev_per_a"]
            )
            writer.writerow(
                {
                    "iteration": iteration,
                    "relative_improvement": (baseline - candidate) / baseline,
                    "metric_passed": str(bool(decision["metric_result"]["passed"])).lower(),
                    "rerun_metric_delta": evaluation["aggregate_metrics"]["rerun_metric_delta"],
                    "reproducibility_passed": str(
                        bool(decision["reproducibility"]["passed"])
                    ).lower(),
                    "decision": decision["decision"],
                }
            )

    summary = _read_json(benchmark / "summary.json")
    expected_summary_values = {
        "semantic_macro_fault_recall": methods["Semantic grader"]["macro_recall"],
        "manifest_only_macro_fault_recall": methods["Manifest only"]["macro_recall"],
        "semantic_micro_fault_recall": methods["Semantic grader"]["micro_recall"],
        "manifest_only_micro_fault_recall": methods["Manifest only"]["micro_recall"],
        "absolute_recall_gain": methods["Semantic grader"]["macro_recall"]
        - methods["Manifest only"]["macro_recall"],
    }
    mismatches = {
        key: {"ledger": expected, "summary": summary.get(key)}
        for key, expected in expected_summary_values.items()
        if not isinstance(summary.get(key), int | float)
        or not math.isclose(float(summary[key]), expected, rel_tol=0.0, abs_tol=1e-12)
    }
    if mismatches:
        raise ValueError(f"summary does not match frozen ledger: {mismatches}")
    derived = {
        "schema_version": "1.0.0",
        "headline": {
            "semantic_detected": methods["Semantic grader"]["detected"],
            "manifest_detected": methods["Manifest only"]["detected"],
            "corrupt_total": len(corrupt),
            "clean_total": len(clean),
            "semantic_macro_fault_recall": methods["Semantic grader"]["macro_recall"],
            "manifest_only_macro_fault_recall": methods["Manifest only"]["macro_recall"],
            "semantic_clean_false_positive_rate": methods["Semantic grader"]["clean_rejected"]
            / len(clean),
            "absolute_recall_gain": expected_summary_values["absolute_recall_gain"],
        },
        "summary_consistency": {
            "passed": True,
            "checked_fields": sorted(expected_summary_values),
            "absolute_tolerance": 1e-12,
        },
        "derived_files": {
            str(recall_csv.relative_to(REPO_ROOT)): sha256_file(recall_csv),
            str(seed_csv.relative_to(REPO_ROOT)): sha256_file(seed_csv),
            str(case_csv.relative_to(REPO_ROOT)): sha256_file(case_csv),
        },
        "source_files": {
            str(ledger_path.relative_to(REPO_ROOT)): sha256_file(ledger_path),
            str((benchmark / "summary.json").relative_to(REPO_ROOT)): sha256_file(
                benchmark / "summary.json"
            ),
            str((real_trace / "iteration-001/decision.json").relative_to(REPO_ROOT)): sha256_file(
                real_trace / "iteration-001/decision.json"
            ),
            str((real_trace / "iteration-002/decision.json").relative_to(REPO_ROOT)): sha256_file(
                real_trace / "iteration-002/decision.json"
            ),
            str((real_trace / "iteration-001/evaluation.json").relative_to(REPO_ROOT)): sha256_file(
                real_trace / "iteration-001/evaluation.json"
            ),
            str((real_trace / "iteration-002/evaluation.json").relative_to(REPO_ROOT)): sha256_file(
                real_trace / "iteration-002/evaluation.json"
            ),
        },
    }
    analysis_path = output / "analysis.json"
    analysis_path.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n")
    return derived


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--real-trace", type=Path, default=DEFAULT_REAL_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_analysis(args.benchmark, args.real_trace, args.output), indent=2))


if __name__ == "__main__":
    main()
