from __future__ import annotations

import json
from pathlib import Path

from mlip_research_agent.research.trace_fault_audit import (
    FAULTS,
    manifest_only_passes,
    run_audit,
)


def test_frozen_fault_catalog_has_twelve_unique_families() -> None:
    names = [name for name, _ in FAULTS]
    assert len(names) == 12
    assert len(set(names)) == 12


def test_manifest_only_rejects_missing_manifest(tmp_path: Path) -> None:
    assert not manifest_only_passes(tmp_path)


def test_single_seed_audit_writes_complete_ledger(tmp_path: Path) -> None:
    summary = run_audit(tmp_path, seeds=(20260712,))
    ledger_lines = (tmp_path / "experiment_ledger.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in ledger_lines]
    assert len(records) == 13
    assert summary["n_clean_controls"] == 1
    assert summary["n_corrupt_cases"] == 12
    assert summary["n_fault_families"] == 12
    assert summary["semantic_macro_fault_recall"] == 1.0
    assert summary["manifest_only_macro_fault_recall"] == 1 / 12
    assert summary["absolute_recall_gain"] == 11 / 12
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "execution.json").is_file()
