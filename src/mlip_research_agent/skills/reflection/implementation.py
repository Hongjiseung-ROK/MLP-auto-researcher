"""tea-time-with-reading-poem: a bounded, deterministic creative break.

The skill re-presents the *same* inputs through deliberately estranging
lenses — a public-domain poem, a seeded set of reframing techniques, and
alternative numerical views of the same data — so the next reasoning pass
starts with fresh eyes instead of yesterday's assumptions.

Everything here is agent-text class under the provenance policy: the skill
registers artifacts but never claims, and its output may inspire a hypothesis
but can never serve as evidence for one.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill
from mlip_research_agent.skills.reflection.poems import POEM_CORPUS, Poem
from mlip_research_agent.skills.reflection.schema import TeaTimeInput, TeaTimeOutput
from mlip_research_agent.skills.reflection.validators import (
    assert_no_claims_registered,
    load_data,
    validate_poem_key,
)

REPORT_FILENAME = "tea_time_report.md"
REFRAMINGS_FILENAME = "reframings.json"


class Technique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technique_id: str
    prompt_template: str  # {question} is substituted


TECHNIQUE_CATALOG: list[Technique] = [
    Technique(
        technique_id="inversion",
        prompt_template=(
            "Assume the opposite of your current belief about: {question}. "
            "What in the data would you now read as support? Whatever you find "
            "was invisible a minute ago."
        ),
    ),
    Technique(
        technique_id="beginners_mind",
        prompt_template=(
            "Explain {question} to someone who knows no jargon. Every term you "
            "cannot replace with a plain word is an assumption you have stopped seeing."
        ),
    ),
    Technique(
        technique_id="outlier_first",
        prompt_template=(
            "Ignore the average behavior entirely. Start the story of {question} "
            "from the single strangest point and work outward."
        ),
    ),
    Technique(
        technique_id="null_model",
        prompt_template=(
            "What would this data look like if nothing interesting were happening in "
            "{question}? Describe the boring universe precisely; your result is only "
            "the distance from it."
        ),
    ),
    Technique(
        technique_id="axis_swap",
        prompt_template=(
            "Swap the roles of the variables in {question}: treat the thing you "
            "predict as the thing you know, and vice versa. What question is the "
            "data now answering?"
        ),
    ),
    Technique(
        technique_id="negative_space",
        prompt_template=(
            "List what is absent from the data about {question}: configurations never "
            "sampled, regimes never visited, failures never recorded. The absence has "
            "a shape; sketch it."
        ),
    ),
    Technique(
        technique_id="falsification",
        prompt_template=(
            "Write the single measurement that would kill your favorite answer to "
            "{question}. If you cannot afford to make it, say so out loud."
        ),
    ),
]


def _extract_series(data: dict[str, Any]) -> list[float]:
    """Pull a numeric series out of a labels- or metrics-shaped artifact."""
    if isinstance(data.get("labels"), list):
        return [
            float(rec["energy"])
            for rec in data["labels"]
            if isinstance(rec, dict) and isinstance(rec.get("energy"), int | float)
        ]
    return [float(v) for v in data.values() if isinstance(v, int | float)]


def _alternative_views(series: list[float]) -> dict[str, Any]:
    """The same numbers, summarized every way except the habitual one."""
    arr = np.asarray(series, dtype=float)
    median = float(np.median(arr))
    deviations = np.abs(arr - median)
    order = np.argsort(arr, kind="stable")
    return {
        "n_values": int(arr.size),
        "mean": round(float(arr.mean()), 10),
        "median": round(median, 10),
        "mean_minus_median": round(float(arr.mean()) - median, 10),
        "min": round(float(arr.min()), 10),
        "max": round(float(arr.max()), 10),
        "spread_std": round(float(arr.std()), 10),
        "strangest_index": int(np.argmax(deviations)),
        "strangest_value": round(float(arr[int(np.argmax(deviations))]), 10),
        "rank_bottom3_indices": [int(i) for i in order[:3]],
        "rank_top3_indices": [int(i) for i in order[-3:][::-1]],
    }


@register_skill
class TeaTimeWithReadingPoemSkill(Skill):
    name = "tea_time_with_reading_poem"
    input_model = TeaTimeInput
    output_model = TeaTimeOutput
    cost_class = "trivial"
    permission_level = "auto"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, TeaTimeInput)
        validate_poem_key(params.poem_key)
        data = load_data(ctx.run_dir, params.data_path)

        rng = np.random.default_rng([ctx.seed, 99])
        poem_keys = sorted(POEM_CORPUS)
        poem: Poem = POEM_CORPUS[
            params.poem_key
            if params.poem_key is not None
            else poem_keys[int(rng.integers(len(poem_keys)))]
        ]
        chosen = rng.choice(len(TECHNIQUE_CATALOG), size=params.n_provocations, replace=False)
        techniques = [TECHNIQUE_CATALOG[int(i)] for i in sorted(int(i) for i in chosen)]
        provocations = [
            {
                "technique": tech.technique_id,
                "prompt": tech.prompt_template.format(question=params.focus_question),
                "poem_line": poem.lines[int(rng.integers(len(poem.lines)))],
            }
            for tech in techniques
        ]
        series = _extract_series(data) if data is not None else []
        views = _alternative_views(series) if len(series) >= 2 else None

        reframings = {
            "focus_question": params.focus_question,
            "poem": poem.model_dump(),
            "provocations": provocations,
            "alternative_data_views": views,
            "evidence_class": "agent_text",
        }
        reframings_path = ctx.step_dir / REFRAMINGS_FILENAME
        reframings_path.write_text(json.dumps(reframings, indent=2, sort_keys=True) + "\n")
        report_path = ctx.step_dir / REPORT_FILENAME
        report_path.write_text(_render_report(params.focus_question, poem, provocations, views))

        reframings_artifact = ctx.registry.register(
            reframings_path, kind="reflection", step_id=ctx.step_id
        )
        report_artifact = ctx.registry.register(
            report_path, kind="reflection", step_id=ctx.step_id
        )
        assert_no_claims_registered(len(ctx.claims))
        return TeaTimeOutput(
            report_artifact=report_artifact.artifact_id,
            report_path=report_artifact.relative_path,
            reframings_artifact=reframings_artifact.artifact_id,
            reframings_path=reframings_artifact.relative_path,
            poem_key=poem.key,
            techniques=[t.technique_id for t in techniques],
            n_provocations=len(provocations),
        )


def _render_report(
    question: str,
    poem: Poem,
    provocations: list[dict[str, str]],
    views: dict[str, Any] | None,
) -> str:
    lines = [
        "# Tea time",
        "",
        "> This report is agent-text class: it may inspire the next experiment,",
        "> but it is never evidence and registers no claims.",
        "",
        f"Put the question down for a moment: *{question}*",
        "",
        f"## The poem — {poem.title}",
        f"*{poem.author}*",
        "",
        *[f"> {line}" for line in poem.lines],
        "",
        f"Why this poem now: {poem.why_it_helps}.",
        "",
        "## Provocations",
        "",
    ]
    for i, item in enumerate(provocations, 1):
        lines += [
            f"### {i}. {item['technique'].replace('_', ' ')}",
            "",
            item["prompt"],
            "",
            f"*(hold this next to: “{item['poem_line']}”)*",
            "",
        ]
    if views is not None:
        lines += [
            "## The same boring data, re-viewed",
            "",
            "| view | value |",
            "|---|---|",
            *[f"| {k} | {v} |" for k, v in views.items()],
            "",
            "If the mean and the median disagree, the story you tell depends on",
            "which one you befriended first.",
            "",
        ]
    lines += [
        "## Back to work",
        "",
        "Pick the one provocation that annoyed you most — annoyance is usually",
        "a prior defending itself — and spend ten minutes taking it seriously.",
        "",
    ]
    return "\n".join(lines)
