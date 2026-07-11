"""Member identity, finite signals, and stable ranking for ensemble_uq."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.active_learning.ensemble_uq.implementation import (
    EnsembleUQSkill,
)
from mlip_research_agent.skills.active_learning.ensemble_uq.schema import (
    EnsembleUQInput,
    EnsembleUQOutput,
)
from mlip_research_agent.skills.base import SkillContext, SkillError

IDS = ["cand-a", "cand-b"]


def member(member_id: str, hash_char: str, scale: float) -> dict[str, Any]:
    return {
        "member_id": member_id,
        "provenance_sha256": hash_char * 64,
        "predicted_forces": {
            "cand-a": [[0.0, 0.0, 0.1 * scale], [0.0, 0.0, -0.1 * scale]],
            "cand-b": [[0.5 * scale, 0.0, 0.0], [-0.5 * scale, 0.0, 0.0]],
        },
    }


def base_inputs(**overrides: Any) -> EnsembleUQInput:
    payload: dict[str, Any] = {
        "pool_candidate_ids": IDS,
        "metadata": {
            "cand-a": {"source_group": "g1", "n_atoms": 2, "formula": "Cu2"},
            "cand-b": {"source_group": "g2", "n_atoms": 2, "formula": "Cu2"},
        },
        "members": [member("m1", "a", 1.0), member("m2", "b", 1.2), member("m3", "c", 0.8)],
        "campaign_id": "camp",
        "round_id": "round-1",
    }
    payload.update(overrides)
    return EnsembleUQInput.model_validate(payload)


def run(tmp_path: Path, params: EnsembleUQInput) -> EnsembleUQOutput:
    ctx = SkillContext(
        run_dir=tmp_path, step_id="uq", seed=3, attempt=1, registry=ArtifactRegistry(tmp_path)
    )
    out = EnsembleUQSkill().run(params, ctx)
    assert isinstance(out, EnsembleUQOutput)
    return out


def test_fewer_than_three_members_rejected() -> None:
    with pytest.raises(ValidationError):
        base_inputs(members=[member("m1", "a", 1.0), member("m2", "b", 1.2)])


def test_duplicate_member_identity_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        base_inputs(
            members=[member("m1", "a", 1.0), member("m1", "b", 1.2), member("m3", "c", 0.8)]
        )
    with pytest.raises(ValidationError, match="distinct provenance"):
        base_inputs(
            members=[member("m1", "a", 1.0), member("m2", "a", 1.2), member("m3", "c", 0.8)]
        )


def test_member_pool_mismatch_rejected(tmp_path: Path) -> None:
    bad = member("m3", "c", 0.8)
    del bad["predicted_forces"]["cand-b"]
    with pytest.raises(SkillError, match="do not match the pool"):
        run(tmp_path, base_inputs(members=[member("m1", "a", 1.0), member("m2", "b", 1.2), bad]))


def test_finite_signal_and_ranking(tmp_path: Path) -> None:
    out = run(tmp_path, base_inputs())
    payload = json.loads((tmp_path / out.uq_path).read_text())
    records = payload["records"]
    assert out.n_candidates_ranked == 2 and out.n_candidates_invalid == 0
    for record in records:
        assert math.isfinite(record["force_disagreement"])
        assert record["force_disagreement"] >= 0.0
    # cand-b spreads 0.5*scale across members: larger disagreement than cand-a.
    assert records[0]["candidate_id"] == "cand-b"
    assert records[0]["rank"] == 0


def test_nan_member_flags_candidate_invalid(tmp_path: Path) -> None:
    bad = member("m3", "c", 0.8)
    bad["predicted_forces"]["cand-a"] = [[0.0, 0.0, float("nan")], [0.0, 0.0, 0.0]]
    out = run(tmp_path, base_inputs(members=[member("m1", "a", 1.0), member("m2", "b", 1.2), bad]))
    assert out.n_candidates_invalid == 1
    assert out.invalid_member_flags == {"cand-a": ["m3"]}
    payload = json.loads((tmp_path / out.uq_path).read_text())
    assert [r["candidate_id"] for r in payload["records"]] == ["cand-b"]


def test_stable_ranking_deterministic(tmp_path: Path) -> None:
    out1 = run(tmp_path / "r1", base_inputs())
    out2 = run(tmp_path / "r2", base_inputs())
    p1 = json.loads(((tmp_path / "r1") / out1.uq_path).read_text())
    p2 = json.loads(((tmp_path / "r2") / out2.uq_path).read_text())
    assert p1["records"] == p2["records"]


def test_signal_is_called_disagreement_not_uncertainty(tmp_path: Path) -> None:
    out = run(tmp_path, base_inputs())
    assert out.signal_kind == "ensemble_disagreement"
    payload = json.loads((tmp_path / out.uq_path).read_text())
    assert "not" in payload["signal_caveat"] and "calibrated" in payload["signal_caveat"]


def test_no_label_inputs_possible() -> None:
    with pytest.raises(ValidationError):
        base_inputs(hidden_labels={"cand-a": -3.7})
    with pytest.raises(ValidationError):
        EnsembleUQInput.model_validate(
            {
                **base_inputs().model_dump(),
                "metadata": {
                    "cand-a": {
                        "source_group": "g1",
                        "n_atoms": 2,
                        "formula": "Cu2",
                        "force_error": 0.1,
                    },
                    "cand-b": {"source_group": "g2", "n_atoms": 2, "formula": "Cu2"},
                },
            }
        )


def test_energy_disagreement_optional(tmp_path: Path) -> None:
    members = []
    variants = (("m1", "a", 1.0, -3.70), ("m2", "b", 1.2, -3.72), ("m3", "c", 0.8, -3.68))
    for mid, h, scale, e in variants:
        m = member(mid, h, scale)
        m["predicted_energy_per_atom"] = {"cand-a": e, "cand-b": e + 0.01}
        members.append(m)
    out = run(tmp_path, base_inputs(members=members, include_energy_disagreement=True))
    payload = json.loads((tmp_path / out.uq_path).read_text())
    for record in payload["records"]:
        assert math.isfinite(record["energy_per_atom_disagreement"])
