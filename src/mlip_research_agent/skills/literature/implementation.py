"""Mock literature retrieval over a fixed fixture corpus.

Deterministic keyword scoring; no network access. The real skill will index
full PDFs (PARNESS-style) once the literature backend is approved.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill
from mlip_research_agent.skills.literature.schema import (
    LiteratureInput,
    LiteratureOutput,
    Reference,
)
from mlip_research_agent.skills.literature.validators import validate_references

REFERENCES_FILENAME = "references.json"

FIXTURE_CORPUS = [
    Reference(
        key="batatia2024mace",
        title="A foundation model for atomistic materials chemistry (MACE-MP-0)",
        venue="arXiv:2401.00096",
        year=2024,
        relevance="foundation MLIP fine-tuning baseline",
    ),
    Reference(
        key="deng2023chgnet",
        title=(
            "CHGNet: pretrained universal neural network potential for "
            "charge-informed atomistic modelling"
        ),
        venue="Nature Machine Intelligence",
        year=2023,
        relevance="materials baseline pretrained on Materials Project trajectories",
    ),
    Reference(
        key="riebesell2024matbench",
        title=(
            "Matbench Discovery -- A framework to evaluate machine learning "
            "crystal stability predictions"
        ),
        venue="arXiv:2308.14920",
        year=2024,
        relevance="public evaluation anchor for materials MLIPs",
    ),
    Reference(
        key="zaverkin2024uncertainty",
        title=(
            "Uncertainty-biased molecular dynamics for learning uniformly "
            "accurate interatomic potentials"
        ),
        venue="npj Computational Materials",
        year=2024,
        relevance="uncertainty-driven active learning for MLIPs",
    ),
    Reference(
        key="neumann2024orb",
        title="Orb: a fast, scalable neural network potential",
        venue="arXiv:2410.22570",
        year=2024,
        relevance="fast committee member for screening",
    ),
]


@register_skill
class MockLiteratureSkill(Skill):
    name = "mock_literature"
    input_model = LiteratureInput
    output_model = LiteratureOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, LiteratureInput)
        terms = {t.lower() for t in params.query.split() if len(t) > 2}

        def score(ref: Reference) -> int:
            text = f"{ref.title} {ref.relevance}".lower()
            return sum(1 for t in terms if t in text)

        ranked = sorted(FIXTURE_CORPUS, key=lambda r: (-score(r), r.key))
        selected = ranked[: params.max_results]
        validate_references(selected)

        path = ctx.step_dir / REFERENCES_FILENAME
        payload = {
            "query": params.query,
            "references": [r.model_dump() for r in selected],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        artifact = ctx.registry.register(path, kind="reference_list", step_id=ctx.step_id)
        return LiteratureOutput(
            references_artifact=artifact.artifact_id,
            references_path=artifact.relative_path,
            n_references=len(selected),
        )
