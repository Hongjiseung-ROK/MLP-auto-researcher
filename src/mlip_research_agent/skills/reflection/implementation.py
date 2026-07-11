"""Tea Time: a bounded, deterministic pause-and-review layer.

The skill combines a lightweight creative reset with a structured audit of
benchmark lock-in, bounded alternative paths, and pending owner questions.
It intentionally has no data-artifact input, so hidden labels, detailed test
metrics, and secrets cannot enter this reflection boundary.

Everything here is agent-text class under the provenance policy: the skill
registers artifacts but never claims, and its output may inspire a hypothesis
but can never serve as evidence for one.
"""

from __future__ import annotations

import json

import numpy as np
from pydantic import BaseModel, ConfigDict

from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill
from mlip_research_agent.skills.reflection.poems import POEM_CORPUS, Poem
from mlip_research_agent.skills.reflection.schema import (
    AlternativePath,
    OwnerQuestionDraft,
    TeaTimeInput,
    TeaTimeOutput,
)
from mlip_research_agent.skills.reflection.validators import (
    assert_claim_count_unchanged,
    assert_no_sensitive_payload,
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


def _benchmark_audit(params: TeaTimeInput) -> dict[str, object]:
    """Return a deterministic, intentionally conservative lock-in audit."""
    reusable = sorted(set(params.reusable_components))
    benchmark_specific = sorted(set(params.benchmark_specific_components))
    if benchmark_specific and not reusable:
        risk = "high"
    elif benchmark_specific and len(benchmark_specific) >= len(reusable):
        risk = "medium"
    else:
        risk = "low"
    return {
        "risk": risk,
        "benchmark_role": params.benchmark_role,
        "reusable_components": reusable,
        "benchmark_specific_components": benchmark_specific,
        "audit_rule": (
            "high when only benchmark-specific components are named; medium when they "
            "equal or outnumber reusable components; low otherwise"
        ),
        "controls": [
            "controller, domain skill, and execution backend remain decoupled",
            "benchmark-specific identifiers stay in configuration and fixtures",
            "no hidden labels or detailed test metrics enter this reflection",
            "reflection artifacts cannot support claims",
        ],
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
        claims_before = len(ctx.claims)

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
        audit = _benchmark_audit(params)
        owner_packet = {
            "status": (
                "questions_pending"
                if params.open_owner_questions
                else "no_consequential_questions"
            ),
            "questions": [
                question.model_dump(mode="json") for question in params.open_owner_questions
            ],
        }

        reframings = {
            "schema_version": "2.0.0",
            "trigger": params.trigger.value,
            "focus_question": params.focus_question,
            "current_stage_purpose": params.stage_objective,
            "benchmark_overfitting_audit": audit,
            "alternative_paths": [
                alternative.model_dump(mode="json") for alternative in params.alternatives
            ],
            "owner_question_packet_draft": owner_packet,
            "poem": poem.model_dump(),
            "provocations": provocations,
            "evidence_class": "agent_text",
            "claim_eligible": False,
        }
        assert_no_sensitive_payload(reframings)
        reframings_path = ctx.step_dir / REFRAMINGS_FILENAME
        reframings_path.write_text(json.dumps(reframings, indent=2, sort_keys=True) + "\n")
        report_path = ctx.step_dir / REPORT_FILENAME
        report_path.write_text(
            _render_report(
                params,
                poem,
                provocations,
                audit,
                params.alternatives,
                params.open_owner_questions,
            )
        )

        reframings_artifact = ctx.registry.register(
            reframings_path, kind="reflection", step_id=ctx.step_id
        )
        report_artifact = ctx.registry.register(
            report_path, kind="reflection", step_id=ctx.step_id
        )
        assert_claim_count_unchanged(claims_before, len(ctx.claims))
        return TeaTimeOutput(
            report_artifact=report_artifact.artifact_id,
            report_path=report_artifact.relative_path,
            reframings_artifact=reframings_artifact.artifact_id,
            reframings_path=reframings_artifact.relative_path,
            poem_key=poem.key,
            techniques=[t.technique_id for t in techniques],
            n_provocations=len(provocations),
            trigger=params.trigger,
            purpose_summary=params.stage_objective,
            benchmark_overfitting_risk=str(audit["risk"]),
            alternative_path_ids=[path.path_id for path in params.alternatives],
            n_owner_questions=len(params.open_owner_questions),
        )


def _render_report(
    params: TeaTimeInput,
    poem: Poem,
    provocations: list[dict[str, str]],
    audit: dict[str, object],
    alternatives: list[AlternativePath],
    owner_questions: list[OwnerQuestionDraft],
) -> str:
    lines = [
        "# Tea time",
        "",
        "> This report is agent-text class: it may inspire the next experiment,",
        "> but it is never evidence and registers no claims.",
        "",
        f"Trigger: `{params.trigger.value}`",
        "",
        "## 1. Current-stage purpose",
        "",
        params.stage_objective,
        "",
        f"Put the question down for a moment: *{params.focus_question}*",
        "",
        "## 2. Benchmark-overfitting audit",
        "",
        f"Risk: **{audit['risk']}**",
        "",
        f"Benchmark role: {params.benchmark_role}",
        "",
        "Reusable components: "
        + (", ".join(sorted(set(params.reusable_components))) or "none declared"),
        "",
        "Benchmark-specific components: "
        + (
            ", ".join(sorted(set(params.benchmark_specific_components)))
            or "none declared"
        ),
        "",
        f"## Creative reset — {poem.title}",
        f"*{poem.author}*",
        "",
        *[f"> {line}" for line in poem.lines],
        "",
        f"Why this poem now: {poem.why_it_helps}.",
        "",
        "## Provocations for self-critique",
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
    lines += [
        "## 3. Bounded alternative paths",
        "",
    ]
    for alternative in alternatives:
        lines += [
            f"### {alternative.path_id}",
            "",
            alternative.summary,
            "",
            f"Cost: {alternative.cost}. Risk: {alternative.risk}",
            "",
        ]
    lines += ["## 4. Owner question packet draft", ""]
    if owner_questions:
        for question in owner_questions:
            lines += [
                f"### {question.question_id}: {question.question}",
                "",
                f"Why: {question.why_needed}",
                "",
                f"Evidence: {question.current_evidence}",
                "",
                f"Recommended default: {question.recommended_default}",
                "",
                *[f"- {choice}" for choice in question.choices],
                "",
            ]
    else:
        lines += [
            "No new consequential owner question was identified at this pause point.",
            "",
        ]
    lines += [
        "## Back to work",
        "",
        "Proceed only with the reversible path supported by the current approvals;",
        "pause again before the next irreversible or human-gated action.",
        "",
    ]
    return "\n".join(lines)
