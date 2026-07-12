"""Tea Time boundary for the Auto Research loop.

Tea Time is a metacognitive co-regulator, not a scientific evaluator. This
boundary builds the *only* inputs it may see (objective text, proposal
summary, mutation category, aggregate prior decisions, failure category,
remaining budget, protected-constant keys) and invokes the existing
``tea_time_with_reading_poem`` reflection SKILL — it never receives api.env,
raw hidden labels, frozen-test record details, candidate true labels, secret
values, or unapproved checkpoint performance rankings. Its output remains
agent-text and can never satisfy an evidence gate.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from mlip_research_agent.research.auto_research.mutation import MutationClass
from mlip_research_agent.research.auto_research.objective import ResearchObjective
from mlip_research_agent.research.auto_research.proposal import ExperimentProposal
from mlip_research_agent.research.auto_research.validators import ContentAddressedModel
from mlip_research_agent.skills.base import SkillContext, get_skill
from mlip_research_agent.skills.reflection.schema import (
    TeaTimeInput,
    TeaTimeOutput,
    TeaTimeTrigger,
)

#: Boundaries at which the controller must pause for Tea Time.
REQUIRED_TRIGGERS: frozenset[TeaTimeTrigger] = frozenset(
    {
        TeaTimeTrigger.AUTO_RESEARCH_FIRST_PROPOSAL,
        TeaTimeTrigger.AUTO_RESEARCH_REMOTE_PRELAUNCH,
        TeaTimeTrigger.AUTO_RESEARCH_REPEATED_MUTATION_CLASS,
        TeaTimeTrigger.AUTO_RESEARCH_CONSECUTIVE_FAILURES,
        TeaTimeTrigger.AUTO_RESEARCH_PIVOT_REQUEST,
        TeaTimeTrigger.AUTO_RESEARCH_CHECKPOINT_FAMILY_CHANGE,
        TeaTimeTrigger.AUTO_RESEARCH_CLAIM_RENDERING,
        TeaTimeTrigger.AUTO_RESEARCH_ITERATION_BOUNDARY,
    }
)

#: How many uses of the same mutation class in a row force a pause.
REPEATED_MUTATION_CLASS_THRESHOLD = 3
#: Consecutive failed iterations that force a pause.
CONSECUTIVE_FAILURE_THRESHOLD = 2


class TeaTimeReviewRecord(ContentAddressedModel):
    """Typed reflection output stored in the iteration trace. Agent text only."""

    model_config = ConfigDict(extra="forbid")

    review_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    proposal_id: str = Field(min_length=3)
    trigger: TeaTimeTrigger
    evidence_class: str = "agent_text"
    claim_eligible: bool = False
    objective_alignment: str = Field(min_length=5, max_length=2000)
    benchmark_lock_in_risk: str = Field(min_length=2, max_length=2000)
    post_hoc_selection_risk: str = Field(min_length=5, max_length=2000)
    alternative_routes: list[str] = Field(min_length=2, max_length=3)
    continue_or_refine_recommendation: str = Field(pattern=r"^(continue|refine)$")
    pivot_request_rationale: str = Field(
        default="", max_length=2000, description="Non-empty only on a pivot-request pause"
    )
    owner_question: str = Field(min_length=5, max_length=1000)
    skill_report_artifact: str = Field(min_length=1)
    skill_reframings_artifact: str = Field(min_length=1)


def select_trigger(
    *,
    is_first_proposal: bool,
    proposal_is_remote: bool,
    prior_mutation_classes: list[MutationClass],
    consecutive_failures: int,
    is_pivot_request: bool,
    changes_checkpoint_family: bool,
    renders_claims: bool,
) -> TeaTimeTrigger:
    """Pick the most specific required boundary for this pause."""
    if is_pivot_request:
        return TeaTimeTrigger.AUTO_RESEARCH_PIVOT_REQUEST
    if changes_checkpoint_family:
        return TeaTimeTrigger.AUTO_RESEARCH_CHECKPOINT_FAMILY_CHANGE
    if renders_claims:
        return TeaTimeTrigger.AUTO_RESEARCH_CLAIM_RENDERING
    if proposal_is_remote:
        return TeaTimeTrigger.AUTO_RESEARCH_REMOTE_PRELAUNCH
    if consecutive_failures >= CONSECUTIVE_FAILURE_THRESHOLD:
        return TeaTimeTrigger.AUTO_RESEARCH_CONSECUTIVE_FAILURES
    if is_first_proposal:
        return TeaTimeTrigger.AUTO_RESEARCH_FIRST_PROPOSAL
    if prior_mutation_classes:
        latest = prior_mutation_classes[-1]
        streak = 0
        for mutation_class in reversed(prior_mutation_classes):
            if mutation_class is not latest:
                break
            streak += 1
        if streak >= REPEATED_MUTATION_CLASS_THRESHOLD:
            return TeaTimeTrigger.AUTO_RESEARCH_REPEATED_MUTATION_CLASS
    return TeaTimeTrigger.AUTO_RESEARCH_ITERATION_BOUNDARY


def run_tea_time_review(
    *,
    ctx: SkillContext,
    objective: ResearchObjective,
    proposal: ExperimentProposal,
    trigger: TeaTimeTrigger,
    prior_decisions_summary: dict[str, int],
    failure_category: str | None,
    remaining_iterations: int,
    remaining_compute_seconds: float,
) -> TeaTimeReviewRecord:
    """Invoke the reflection SKILL with allowed inputs only and record the review.

    ``prior_decisions_summary`` is aggregate counts (e.g. {"accept": 1}) —
    never metric values or record-level results.
    """
    mutation_classes = sorted({m.target_key for m in proposal.proposed_mutations})
    focus_question = (
        f"Does mutating {', '.join(mutation_classes)} still serve the objective "
        f"'{objective.title}'?"
    )[:500]
    skill_cls = get_skill("tea_time_with_reading_poem")
    inputs = TeaTimeInput(
        focus_question=focus_question,
        trigger=trigger,
        stage_objective=(
            "Bounded Auto Research iteration: propose, review, execute, evaluate "
            "and learn without expanding scientific claims."
        ),
        benchmark_role=(
            "The fixture/benchmark is a pipeline-hardening scaffold, "
            "not the final scientific topic."
        ),
        reusable_components=[
            "auto_research contracts",
            "mutation policy",
            "acceptance policy",
            "trace grader",
        ],
        benchmark_specific_components=[objective.benchmark_adapter],
    )
    output = skill_cls().run(inputs, ctx)
    assert isinstance(output, TeaTimeOutput)

    decisions_text = (
        ", ".join(f"{k}={v}" for k, v in sorted(prior_decisions_summary.items())) or "none yet"
    )
    failure_text = failure_category or "none"
    return TeaTimeReviewRecord(
        review_id=f"tea-time-{proposal.proposal_id}"[:64],
        proposal_id=proposal.proposal_id,
        trigger=trigger,
        objective_alignment=(
            f"Proposal {proposal.proposal_id} mutates {', '.join(mutation_classes)} toward "
            f"'{objective.title}'. Prior decisions: {decisions_text}. Last failure category: "
            f"{failure_text}."
        ),
        benchmark_lock_in_risk=output.benchmark_overfitting_risk,
        post_hoc_selection_risk=(
            "Selection uses the preregistered acceptance policy and aggregate metrics only; "
            "no per-record test errors reach the controller, so post-hoc metric shopping is "
            "structurally blocked. Residual risk: repeated bounded mutations of one knob can "
            "still overfit the fixture landscape."
        ),
        alternative_routes=output.alternative_path_ids,
        continue_or_refine_recommendation=(
            "refine" if failure_category is not None else "continue"
        ),
        pivot_request_rationale=(
            "Pivot pause: a scientific method change is being requested and remains owner-gated."
            if trigger is TeaTimeTrigger.AUTO_RESEARCH_PIVOT_REQUEST
            else ""
        ),
        owner_question=(
            f"Budget remaining: {remaining_iterations} iterations, "
            f"{remaining_compute_seconds:.0f} compute-seconds. Continue this mutation "
            "direction, or ask for a bounded alternative?"
        ),
        skill_report_artifact=output.report_artifact,
        skill_reframings_artifact=output.reframings_artifact,
    ).sealed()
