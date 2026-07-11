"""Descriptor validation, diversity increase, deterministic tie-breaking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.active_learning.diversity_select.implementation import (
    DiversitySelectSkill,
)
from mlip_research_agent.skills.active_learning.diversity_select.schema import (
    DiversitySelectInput,
    DiversitySelectOutput,
)
from mlip_research_agent.skills.base import SkillContext, SkillError


def clustered_fixture(per_cluster: int = 4) -> tuple[list[str], dict[str, list[float]]]:
    """Three tight, well-separated clusters in 2D."""
    centers = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    ids: list[str] = []
    vectors: dict[str, list[float]] = {}
    for c, (cx, cy) in enumerate(centers):
        for i in range(per_cluster):
            cid = f"cl{c}-{i}"
            ids.append(cid)
            vectors[cid] = [cx + 0.01 * i, cy + 0.02 * i]
    return ids, vectors


def make_inputs(
    ids: list[str], vectors: dict[str, list[float]], budget: int, **overrides: Any
) -> DiversitySelectInput:
    payload: dict[str, Any] = {
        "pool_candidate_ids": ids,
        "metadata": {
            c: {"source_group": c.split("-")[0], "n_atoms": 4, "formula": "Cu4"} for c in ids
        },
        "descriptors": {"source": "fixture", "dimension": 2, "vectors": vectors},
        "budget": budget,
        "campaign_id": "camp",
        "round_id": "round-1",
    }
    payload.update(overrides)
    return DiversitySelectInput.model_validate(payload)


def run(tmp_path: Path, params: DiversitySelectInput) -> DiversitySelectOutput:
    ctx = SkillContext(
        run_dir=tmp_path, step_id="div", seed=3, attempt=1, registry=ArtifactRegistry(tmp_path)
    )
    out = DiversitySelectSkill().run(params, ctx)
    assert isinstance(out, DiversitySelectOutput)
    return out


def records(tmp_path: Path, out: DiversitySelectOutput) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = json.loads((tmp_path / out.selection_path).read_text())[
        "records"
    ]
    return recs


def test_mace_descriptor_source_fails_closed(tmp_path: Path) -> None:
    ids, vectors = clustered_fixture()
    params = make_inputs(
        ids,
        vectors,
        budget=3,
        descriptors={"source": "mace_descriptor_adapter", "dimension": 2, "vectors": vectors},
    )
    with pytest.raises(SkillError, match="future boundary"):
        run(tmp_path, params)


def test_dimension_mismatch_rejected(tmp_path: Path) -> None:
    ids, vectors = clustered_fixture()
    vectors[ids[0]] = [1.0, 2.0, 3.0]
    with pytest.raises(SkillError, match="length"):
        run(tmp_path, make_inputs(ids, vectors, budget=3))


def test_non_finite_descriptor_rejected(tmp_path: Path) -> None:
    ids, vectors = clustered_fixture()
    vectors[ids[0]] = [float("inf"), 0.0]
    with pytest.raises(SkillError, match="non-finite"):
        run(tmp_path, make_inputs(ids, vectors, budget=3))


def test_duplicate_descriptors_rejected(tmp_path: Path) -> None:
    ids, vectors = clustered_fixture()
    vectors[ids[1]] = list(vectors[ids[0]])
    with pytest.raises(SkillError, match="identical descriptor"):
        run(tmp_path, make_inputs(ids, vectors, budget=3))


def test_diversity_increases_on_clustered_fixture(tmp_path: Path) -> None:
    """FPS covers all three clusters; head-of-list picks only one."""
    ids, vectors = clustered_fixture()
    out = run(tmp_path, make_inputs(ids, vectors, budget=3))
    fps_groups = {r["source_group"] for r in records(tmp_path, out)}
    assert fps_groups == {"cl0", "cl1", "cl2"}

    head_of_list = sorted(ids)[:3]
    head_groups = {c.split("-")[0] for c in head_of_list}
    assert len(head_groups) == 1
    assert len(fps_groups) > len(head_groups)


def test_deterministic_tie_breaking(tmp_path: Path) -> None:
    """A perfectly symmetric square: ties must resolve identically every run."""
    ids = ["p-a", "p-b", "p-c", "p-d"]
    vectors = {
        "p-a": [0.0, 0.0],
        "p-b": [1.0, 0.0],
        "p-c": [0.0, 1.0],
        "p-d": [1.0, 1.0],
    }
    out1 = run(tmp_path / "r1", make_inputs(ids, vectors, budget=2))
    out2 = run(tmp_path / "r2", make_inputs(list(reversed(ids)), vectors, budget=2))
    ids1 = [r["candidate_id"] for r in records(tmp_path / "r1", out1)]
    ids2 = [r["candidate_id"] for r in records(tmp_path / "r2", out2)]
    assert ids1 == ids2


def test_records_carry_distance_and_group(tmp_path: Path) -> None:
    ids, vectors = clustered_fixture()
    out = run(tmp_path, make_inputs(ids, vectors, budget=3))
    recs = records(tmp_path, out)
    assert len(recs) == 3
    for rank, record in enumerate(recs):
        assert record["rank"] == rank
        assert record["diversity_distance"] >= 0.0
        assert record["source_group"] in {"cl0", "cl1", "cl2"}
        assert record["policy_version"] == "diversity-select/1.0.0"
    assert out.mean_pairwise_distance > 5.0  # clusters are 10 apart


def test_budget_over_pool_rejected(tmp_path: Path) -> None:
    ids, vectors = clustered_fixture(per_cluster=1)
    with pytest.raises(SkillError, match="exceeds pool size"):
        run(tmp_path, make_inputs(ids, vectors, budget=4))
