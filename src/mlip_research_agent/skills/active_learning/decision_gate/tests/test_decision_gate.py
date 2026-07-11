"""Pipeline order, exact budget, and complete reason records for decision_gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.active_learning.decision_gate.implementation import (
    DecisionGateSkill,
)
from mlip_research_agent.skills.active_learning.decision_gate.schema import (
    DecisionGateInput,
    DecisionGateOutput,
)
from mlip_research_agent.skills.base import SkillContext, SkillError


def gate_fixture() -> dict[str, Any]:
    """Six candidates in two descriptor clusters with graded disagreement."""
    ids = [f"c-{i}" for i in range(6)]
    vectors = {
        "c-0": [0.0, 0.0],
        "c-1": [0.1, 0.0],
        "c-2": [0.0, 0.1],
        "c-3": [10.0, 10.0],
        "c-4": [10.1, 10.0],
        "c-5": [10.0, 10.1],
    }
    signals = [
        {"candidate_id": "c-0", "force_disagreement": 0.60, "invalid": False},
        {"candidate_id": "c-1", "force_disagreement": 0.50, "invalid": False},
        {"candidate_id": "c-2", "force_disagreement": 0.40, "invalid": False},
        {"candidate_id": "c-3", "force_disagreement": 0.55, "invalid": False},
        {"candidate_id": "c-4", "force_disagreement": 0.05, "invalid": False},
        {"candidate_id": "c-5", "force_disagreement": 0.90, "invalid": True},
    ]
    return {
        "pool_candidate_ids": ids,
        "metadata": {
            c: {
                "source_group": "left" if int(c[2]) < 3 else "right",
                "n_atoms": 4,
                "formula": "Cu4",
            }
            for c in ids
        },
        "signals": signals,
        "descriptors": {"source": "fixture", "dimension": 2, "vectors": vectors},
        "budget": 2,
        "uncertainty_window_fraction": 0.8,
        "campaign_id": "camp",
        "round_id": "round-1",
    }


def run(tmp_path: Path, payload: dict[str, Any]) -> DecisionGateOutput:
    params = DecisionGateInput.model_validate(payload)
    ctx = SkillContext(
        run_dir=tmp_path, step_id="gate", seed=3, attempt=1, registry=ArtifactRegistry(tmp_path)
    )
    out = DecisionGateSkill().run(params, ctx)
    assert isinstance(out, DecisionGateOutput)
    return out


def payload_of(tmp_path: Path, out: DecisionGateOutput) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((tmp_path / out.selection_path).read_text())
    return payload


def test_invalid_candidates_never_selected(tmp_path: Path) -> None:
    out = run(tmp_path, gate_fixture())
    assert out.n_filtered_invalid == 1
    ids = [r["candidate_id"] for r in payload_of(tmp_path, out)["records"]]
    assert "c-5" not in ids  # highest disagreement but invalid


def test_exact_budget_and_diverse_picks(tmp_path: Path) -> None:
    out = run(tmp_path, gate_fixture())
    recs = payload_of(tmp_path, out)["records"]
    assert out.n_selected == 2 and len(recs) == 2
    groups = {r["source_group"] for r in recs}
    assert groups == {"left", "right"}  # diversity stage spans both clusters


def test_complete_reason_records(tmp_path: Path) -> None:
    out = run(tmp_path, gate_fixture())
    for rank, record in enumerate(payload_of(tmp_path, out)["records"]):
        assert record["candidate_id"]
        assert record["uncertainty_score"] is not None
        assert record["diversity_distance"] is not None
        assert record["source_group"]
        assert record["rank"] == rank
        assert record["selection_reason"]
        assert record["policy_version"] == "decision-gate/1.0.0"


def test_pipeline_order_recorded(tmp_path: Path) -> None:
    out = run(tmp_path, gate_fixture())
    assert payload_of(tmp_path, out)["pipeline"] == [
        "invalid_filtering",
        "uncertainty_window",
        "diversity_selection",
        "exact_budget",
        "reason_artifact",
    ]


def test_budget_unreachable_after_filtering(tmp_path: Path) -> None:
    payload = gate_fixture()
    for signal in payload["signals"]:
        signal["invalid"] = True
    payload["signals"][0]["invalid"] = False
    with pytest.raises(SkillError, match="valid candidates remain"):
        run(tmp_path, payload)


def test_unflagged_nan_disagreement_rejected(tmp_path: Path) -> None:
    payload = gate_fixture()
    payload["signals"][0]["force_disagreement"] = float("nan")
    with pytest.raises(SkillError, match="flagged invalid"):
        run(tmp_path, payload)


def test_signal_pool_mismatch_rejected(tmp_path: Path) -> None:
    payload = gate_fixture()
    payload["signals"] = payload["signals"][:-1]
    with pytest.raises(SkillError, match="do not match the pool"):
        run(tmp_path, payload)


def test_deterministic(tmp_path: Path) -> None:
    out1 = run(tmp_path / "r1", gate_fixture())
    out2 = run(tmp_path / "r2", gate_fixture())
    assert payload_of(tmp_path / "r1", out1)["records"] == payload_of(
        tmp_path / "r2", out2
    )["records"]


def test_window_respects_fraction(tmp_path: Path) -> None:
    out = run(tmp_path, gate_fixture())
    # 5 valid candidates, fraction 0.8 -> window of 4 (>= budget 2).
    assert out.n_in_window == 4
