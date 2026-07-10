"""Correctness tests for the mock_literature skill."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.skills.base import SkillContext
from mlip_research_agent.skills.literature.implementation import MockLiteratureSkill
from mlip_research_agent.skills.literature.schema import LiteratureInput, LiteratureOutput


def run_skill(
    tmp_path: Path, query: str, max_results: int = 3
) -> tuple[LiteratureOutput, dict[str, Any]]:
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="literature",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )
    inputs = LiteratureInput(query=query, max_results=max_results)
    out = MockLiteratureSkill().run(inputs, ctx)
    assert isinstance(out, LiteratureOutput)
    payload: dict[str, Any] = json.loads((tmp_path / out.references_path).read_text())
    return out, payload


def test_deterministic(tmp_path: Path) -> None:
    _, payload_a = run_skill(tmp_path / "a", "uncertainty active learning")
    _, payload_b = run_skill(tmp_path / "b", "uncertainty active learning")
    assert payload_a == payload_b


def test_ranking_prefers_keyword_overlap(tmp_path: Path) -> None:
    _, payload = run_skill(tmp_path, "uncertainty active learning", max_results=1)
    assert payload["references"][0]["key"] == "zaverkin2024uncertainty"


def test_max_results_bound(tmp_path: Path) -> None:
    out, payload = run_skill(tmp_path, "materials potentials", max_results=2)
    assert out.n_references == 2
    assert len(payload["references"]) == 2


def test_schema_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        LiteratureInput(query="ab")
