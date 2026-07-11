"""Deterministic proposal generation: iteration N+1 descends from iteration N.

The policy is a typed function of (objective, mutation policy, current
accepted configuration, prior outcome). It never reads per-record metrics,
labels, or protected data — only the prior decision value, the prior lesson,
and aggregate evaluation deltas that already passed the evaluator boundary.

Rules (first match wins):

- no prior iteration        → conservative first move: reduce ``learning_rate``
- prior REFINE/ESCALATE     → stability repair: halve ``learning_rate``; if at
                              the lower bound, increase ``gradient_clip``
- prior REJECT (constraint) → keep the scientific direction, shrink the
                              resource driver: halve ``batch_size``
- prior REJECT (no effect)  → rotate to the next allowed bounded key
- prior ACCEPT              → bounded nearby value: same key, same direction,
                              half the multiplicative step
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mlip_research_agent.research.auto_research.decision import (
    DecisionValue,
    ExperimentDecision,
)
from mlip_research_agent.research.auto_research.lesson import ResearchLesson
from mlip_research_agent.research.auto_research.mutation import (
    AllowedRange,
    Mutation,
    MutationPolicy,
)
from mlip_research_agent.research.auto_research.objective import ResearchObjective
from mlip_research_agent.research.auto_research.proposal import (
    EstimatedCompute,
    ExpectedObservable,
    ExperimentProposal,
)
from mlip_research_agent.research.auto_research.validators import ConfigValue

PROPOSAL_POLICY_VERSION = "proposal-policy/1.0.0"

#: Deterministic rotation for "try a different knob" moves.
KEY_ROTATION: tuple[str, ...] = (
    "learning_rate",
    "gradient_clip",
    "force_loss_weight",
    "batch_size",
    "scheduler_patience",
    "energy_loss_weight",
)


class PriorOutcome(BaseModel):
    """The only cross-iteration information the proposal policy may read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iteration_id: str
    decision: ExperimentDecision
    lesson: ResearchLesson
    constraint_failed: bool
    mutated_keys: list[str]


def _clamp(value: float, allowed: AllowedRange) -> float:
    assert allowed.minimum is not None and allowed.maximum is not None
    return min(max(value, allowed.minimum), allowed.maximum)


def _numeric_mutation(
    policy: MutationPolicy,
    config: dict[str, ConfigValue],
    key: str,
    factor: float,
    scientific_effect: str,
    engineering_effect: str,
) -> Mutation | None:
    """Multiplicative move on one bounded numeric key; None if it cannot move."""
    allowed = policy.bounded_mutable.get(key)
    if allowed is None or allowed.value_type == "choice":
        return None
    old = config.get(key)
    if isinstance(old, bool) or not isinstance(old, int | float):
        return None
    new_value: ConfigValue
    if allowed.value_type == "int":
        new_value = round(_clamp(float(old) * factor, allowed))
    else:
        new_value = _clamp(float(old) * factor, allowed)
    if new_value == old:
        return None
    return Mutation(
        target_type="config",
        target_path="config_store/current.json",
        target_key=key,
        old_value=old,
        new_value=new_value,
        allowed_range=allowed,
        reversibility="reversible",
        scientific_effect=scientific_effect,
        engineering_effect=engineering_effect,
    ).sealed()


def _first_move(
    policy: MutationPolicy, config: dict[str, ConfigValue]
) -> tuple[Mutation, str, str]:
    mutation = _numeric_mutation(
        policy,
        config,
        "learning_rate",
        0.2,
        "A lower learning rate should move optimization toward the loss basin, "
        "reducing the synthetic validation loss.",
        "Smaller update steps; identical memory and runtime profile.",
    )
    if mutation is None:
        raise ValueError("first move impossible: learning_rate not mutable in this policy")
    hypothesis = (
        "The base learning rate overshoots the optimum of the fixture landscape; "
        "reducing it by 5x will lower the synthetic validation loss."
    )
    falsification = (
        "If the synthetic validation loss does not decrease by the minimum relative "
        "improvement after the reduction, the overshoot hypothesis is wrong."
    )
    return mutation, hypothesis, falsification


def _next_move(
    policy: MutationPolicy,
    config: dict[str, ConfigValue],
    prior: PriorOutcome,
) -> tuple[Mutation, str, str]:
    decision = prior.decision.decision
    prior_key = prior.mutated_keys[0] if prior.mutated_keys else "learning_rate"

    if decision in {DecisionValue.REFINE, DecisionValue.ESCALATE}:
        mutation = _numeric_mutation(
            policy,
            config,
            "learning_rate",
            0.5,
            "Halving the learning rate should restore stable optimization after the "
            f"failure observed in {prior.iteration_id}.",
            "Smaller update steps reduce divergence risk.",
        ) or _numeric_mutation(
            policy,
            config,
            "gradient_clip",
            2.0,
            "Stronger gradient clipping should bound the update magnitude that "
            f"destabilized {prior.iteration_id}.",
            "Clipping caps per-step parameter change.",
        )
        if mutation is None:
            raise ValueError("no stability repair available within bounds")
        return (
            mutation,
            f"Iteration {prior.iteration_id} failed for stability reasons "
            f"({prior.lesson.anti_pattern}); a bounded stability repair will let the same "
            "scientific direction execute.",
            "If the repaired configuration still fails or the loss does not improve, "
            "instability was not the binding problem.",
        )

    if decision is DecisionValue.REJECT and prior.constraint_failed:
        mutation = _numeric_mutation(
            policy,
            config,
            "batch_size",
            0.5,
            "Keeping the accepted scientific direction while halving the batch size "
            "should bring resource usage inside the constraint that rejected "
            f"{prior.iteration_id}.",
            "Smaller batches cut simulated memory and runtime.",
        )
        if mutation is None:
            raise ValueError("no resource repair available within bounds")
        return (
            mutation,
            f"Iteration {prior.iteration_id} improved the metric but violated a resource "
            "constraint; shrinking the resource driver preserves the science within budget.",
            "If resource usage still violates the constraint after halving the batch size, "
            "batch size is not the dominant resource driver.",
        )

    if decision is DecisionValue.REJECT:
        # Legal but ineffective: rotate to the next allowed bounded key.
        start = KEY_ROTATION.index(prior_key) if prior_key in KEY_ROTATION else -1
        for offset in range(1, len(KEY_ROTATION) + 1):
            key = KEY_ROTATION[(start + offset) % len(KEY_ROTATION)]
            if key == prior_key:
                continue
            mutation = _numeric_mutation(
                policy,
                config,
                key,
                0.5,
                f"Mutating {key} explores a different bounded direction after "
                f"{prior_key} proved ineffective in {prior.iteration_id}.",
                "Bounded halving of one configuration knob.",
            )
            if mutation is not None:
                return (
                    mutation,
                    f"Iteration {prior.iteration_id} showed {prior_key} is not the binding "
                    f"factor; a different bounded knob ({key}) may be.",
                    f"If mutating {key} also fails to improve the metric, neither knob is "
                    "the binding factor and the mutation class itself is suspect.",
                )
        raise ValueError("no alternative bounded key can move within bounds")

    # ACCEPT: bounded nearby value — same key, same direction, half the step.
    mutation = _numeric_mutation(
        policy,
        config,
        prior_key,
        0.5 if _accepted_direction_was_decrease(prior) else 2.0,
        f"Iteration {prior.iteration_id} accepted a move of {prior_key}; a smaller step in "
        "the same direction tests whether the improvement continues toward the optimum.",
        "Half-sized multiplicative step on the accepted knob.",
    )
    if mutation is None:
        # The accepted knob hit its bound: fall back to key rotation.
        return _rotate_after_bound(policy, config, prior)
    return (
        mutation,
        f"The accepted direction on {prior_key} in {prior.iteration_id} has not been "
        "exhausted; a bounded nearby value should yield a further, smaller improvement.",
        f"If the further move on {prior_key} does not improve the metric, the optimum "
        "lies between the two accepted values and the direction is exhausted.",
    )


def _accepted_direction_was_decrease(prior: PriorOutcome) -> bool:
    """The lesson records the accepted move; parse its direction marker."""
    return "decrease" in prior.lesson.observed_result.lower() or (
        "reduc" in prior.lesson.attempt_summary.lower()
    )


def _rotate_after_bound(
    policy: MutationPolicy,
    config: dict[str, ConfigValue],
    prior: PriorOutcome,
) -> tuple[Mutation, str, str]:
    prior_key = prior.mutated_keys[0] if prior.mutated_keys else "learning_rate"
    for key in KEY_ROTATION:
        if key == prior_key:
            continue
        mutation = _numeric_mutation(
            policy,
            config,
            key,
            0.5,
            f"{prior_key} reached its bound after {prior.iteration_id}; {key} is the next "
            "bounded knob in the deterministic rotation.",
            "Bounded halving of one configuration knob.",
        )
        if mutation is not None:
            return (
                mutation,
                f"{prior_key} is at its bound; testing {key} continues the search "
                "without violating the mutation policy.",
                f"If {key} does not improve the metric, this rotation step is exhausted too.",
            )
    raise ValueError("every rotation key is immovable within bounds")


def generate_proposal(
    *,
    objective: ResearchObjective,
    policy: MutationPolicy,
    current_config: dict[str, ConfigValue],
    iteration_index: int,
    prior: PriorOutcome | None,
    seed: int,
) -> ExperimentProposal:
    """Deterministic: same inputs always yield the same sealed proposal."""
    if prior is None:
        mutation, hypothesis, falsification = _first_move(policy, current_config)
        parent = None
    else:
        mutation, hypothesis, falsification = _next_move(policy, current_config, prior)
        parent = prior.iteration_id

    primary_metric = objective.success_criteria[0].metric
    return ExperimentProposal(
        proposal_id=f"{objective.objective_id}-p{iteration_index + 1:03d}"[:64],
        objective_id=objective.objective_id,
        parent_iteration_id=parent,
        hypothesis=hypothesis,
        proposed_mutations=[mutation],
        expected_mechanism=mutation.scientific_effect,
        expected_observables=[
            ExpectedObservable(
                metric=primary_metric,
                expected_direction="decrease",
                rationale="The mutation is chosen to move the primary metric toward the "
                "objective's success criterion.",
            )
        ],
        falsification_condition=falsification,
        estimated_compute=EstimatedCompute(device="cpu", estimated_seconds=30.0, remote=False),
        required_skills=["tea_time_with_reading_poem"],
        risk_class="low",
        required_approval=None,
        seed=seed,
        proposal_policy_version=PROPOSAL_POLICY_VERSION,
    ).sealed()
