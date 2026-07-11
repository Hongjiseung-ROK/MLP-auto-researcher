"""Label-free contract, determinism, and exact budget for random_select."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.active_learning.random_select.implementation import (
    RandomSelectSkill,
)
from mlip_research_agent.skills.active_learning.random_select.schema import (
    RandomSelectInput,
    RandomSelectOutput,
)
from mlip_research_agent.skills.base import SkillContext, SkillError


def make_ctx(tmp_path: Path, step_id: str = "select") -> SkillContext:
    return SkillContext(
        run_dir=tmp_path,
        step_id=step_id,
        seed=7,
        attempt=1,
        registry=ArtifactRegistry(tmp_path),
    )


def pool_inputs(n: int = 10, budget: int = 4, seed: int = 20260712) -> RandomSelectInput:
    ids = [f"cand-{i:04d}" for i in range(n)]
    return RandomSelectInput.model_validate(
        {
            "pool_candidate_ids": ids,
            "metadata": {
                c: {"source_group": f"group-{i % 3}", "n_atoms": 8, "formula": "Cu8"}
                for i, c in enumerate(ids)
            },
            "budget": budget,
            "seed": seed,
            "campaign_id": "camp",
            "round_id": "round-1",
        }
    )


def run(tmp_path: Path, params: RandomSelectInput, step_id: str = "select") -> RandomSelectOutput:
    out = RandomSelectSkill().run(params, make_ctx(tmp_path, step_id))
    assert isinstance(out, RandomSelectOutput)
    return out


def selected_ids(tmp_path: Path, out: RandomSelectOutput) -> list[str]:
    import json

    payload = json.loads((tmp_path / out.selection_path).read_text())
    return [r["candidate_id"] for r in payload["records"]]


def test_label_free_input_contract() -> None:
    # Label-shaped metadata is structurally impossible: extra keys are forbidden.
    with pytest.raises(ValidationError):
        RandomSelectInput.model_validate(
            {
                "pool_candidate_ids": ["a"],
                "metadata": {
                    "a": {
                        "source_group": "g",
                        "n_atoms": 2,
                        "formula": "Cu2",
                        "energy_ev": -3.7,
                    }
                },
                "budget": 1,
                "seed": 0,
                "campaign_id": "c",
                "round_id": "r",
            }
        )
    with pytest.raises(ValidationError):
        RandomSelectInput.model_validate(
            {
                "pool_candidate_ids": ["a"],
                "metadata": {"a": {"source_group": "g", "n_atoms": 2}},
                "budget": 1,
                "seed": 0,
                "campaign_id": "c",
                "round_id": "r",
                "hidden_labels": {"a": 1.0},
            }
        )


def test_deterministic_for_same_seed_and_pool(tmp_path: Path) -> None:
    out1 = run(tmp_path / "r1", pool_inputs(seed=11))
    out2 = run(tmp_path / "r2", pool_inputs(seed=11))
    assert selected_ids(tmp_path / "r1", out1) == selected_ids(tmp_path / "r2", out2)


def test_input_order_does_not_change_selection(tmp_path: Path) -> None:
    params = pool_inputs(seed=11)
    shuffled = params.model_copy(
        update={"pool_candidate_ids": list(reversed(params.pool_candidate_ids))}
    )
    out1 = run(tmp_path / "r1", params)
    out2 = run(tmp_path / "r2", shuffled)
    assert selected_ids(tmp_path / "r1", out1) == selected_ids(tmp_path / "r2", out2)


def test_different_seed_changes_selection(tmp_path: Path) -> None:
    out1 = run(tmp_path / "r1", pool_inputs(seed=1))
    out2 = run(tmp_path / "r2", pool_inputs(seed=2))
    assert selected_ids(tmp_path / "r1", out1) != selected_ids(tmp_path / "r2", out2)


def test_exact_budget_and_no_duplicates(tmp_path: Path) -> None:
    out = run(tmp_path, pool_inputs(n=10, budget=4))
    ids = selected_ids(tmp_path, out)
    assert len(ids) == 4
    assert len(set(ids)) == 4
    assert out.n_selected == 4


def test_budget_over_pool_rejected(tmp_path: Path) -> None:
    with pytest.raises(SkillError, match="exceeds pool size"):
        run(tmp_path, pool_inputs(n=3, budget=4))


def test_duplicate_pool_ids_rejected(tmp_path: Path) -> None:
    params = pool_inputs(n=4, budget=2)
    duplicated = params.model_copy(
        update={"pool_candidate_ids": [*params.pool_candidate_ids, "cand-0000"]}
    )
    with pytest.raises(SkillError, match="duplicate"):
        run(tmp_path, duplicated)


def test_reason_records_complete(tmp_path: Path) -> None:
    import json

    out = run(tmp_path, pool_inputs(n=6, budget=3))
    payload = json.loads((tmp_path / out.selection_path).read_text())
    for rank, record in enumerate(payload["records"]):
        assert record["rank"] == rank
        assert record["candidate_id"]
        assert record["source_group"]
        assert record["selection_reason"]
        assert record["policy_version"] == "random-select/1.0.0"
        assert record["uncertainty_score"] is None  # control arm uses no model signal
