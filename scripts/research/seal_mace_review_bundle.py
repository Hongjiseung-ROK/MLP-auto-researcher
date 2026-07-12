#!/usr/bin/env python
"""Validate three aggregate-only reviews and seal the iteration-two proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.research.auto_research import (
    AgentReview,
    EstimatedCompute,
    ExpectedObservable,
    ExperimentProposal,
    LocalDemoConfig,
    Mutation,
    MutationPolicy,
    ReviewPacket,
    ReviewSynthesis,
)
from mlip_research_agent.research.auto_research.mutation import TransactionalConfigStore
from mlip_research_agent.research.auto_research.validators import ConfigValue

REVIEW_ROLES = (
    "mlip_scientist",
    "active_learning_scientist",
    "scientific_auditor",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _bounded_value(
    *, key: str, direction: str, old: ConfigValue, policy: MutationPolicy
) -> ConfigValue:
    allowed = policy.bounded_mutable[key]
    direction_lower = direction.lower()
    increase_words = any(word in direction_lower for word in ("increase", "raise", "larger"))
    decrease_words = any(word in direction_lower for word in ("decrease", "reduce", "smaller"))
    if increase_words == decrease_words and allowed.value_type != "choice":
        raise ValueError("numeric review direction must unambiguously increase or decrease")
    increase = increase_words
    if key == "batch_size" and not increase:
        raise ValueError(
            "batch_size cannot decrease below the fixed 32-record replay training set"
        )
    if allowed.value_type == "choice":
        assert allowed.choices is not None
        prefix = "set_to:"
        if not direction_lower.startswith(prefix):
            raise ValueError("choice review direction must use set_to:<allowed-choice>")
        selected = direction_lower.removeprefix(prefix).strip()
        if selected == old or selected not in allowed.choices:
            raise ValueError(f"invalid choice direction for {key}: {selected}")
        return selected
    if isinstance(old, bool) or not isinstance(old, int | float):
        raise ValueError(f"current value for {key} is not numeric")
    assert allowed.minimum is not None and allowed.maximum is not None
    factor = 2.0 if increase else 0.5
    candidate = min(max(float(old) * factor, allowed.minimum), allowed.maximum)
    value: ConfigValue = round(candidate) if allowed.value_type == "int" else candidate
    if value == old:
        alternate = allowed.maximum if increase else allowed.minimum
        value = round(alternate) if allowed.value_type == "int" else alternate
    if value == old or not allowed.permits(value):
        raise ValueError(f"review direction cannot produce a legal change for {key}")
    return value


def seal_bundle(
    *, run_dir: Path, drafts_dir: Path, output_dir: Path, config_path: Path
) -> Path:
    config = LocalDemoConfig.load(config_path)
    policy = MutationPolicy.load(config_path.parent.parent.parent / config.mutation_policy_path)
    reviews: dict[str, AgentReview] = {}
    packet = ReviewPacket.model_validate_json((run_dir / "review_packet.json").read_text())
    if not packet.verify_seal():
        raise ValueError("review packet seal mismatch")
    allowed_evidence = set(packet.allowed_evidence_artifact_ids)
    for role in REVIEW_ROLES:
        raw = _load_json(drafts_dir / f"{role}.json")
        raw["role"] = role
        raw["evidence_class"] = "agent_text"
        raw["claim_eligible"] = False
        raw["review_packet_sha256"] = packet.content_sha256
        reviews[role] = AgentReview.model_validate(raw).sealed()
        if not set(reviews[role].evidence_artifact_ids) <= allowed_evidence:
            raise ValueError(f"{role} cites evidence outside the review packet")

    unsafe = [
        f"{role}:{review.review_status}"
        for role, review in reviews.items()
        if review.review_status in {"stop", "escalate"}
    ]
    if unsafe:
        raise ValueError(f"review boundary stops the replay: {unsafe}")

    synthesis_draft = _load_json(drafts_dir / "synthesis.json")
    recommended_key = synthesis_draft["recommended_mutation_class"]
    if recommended_key not in policy.bounded_mutable:
        raise ValueError(f"synthesis chose an unapproved mutation: {recommended_key}")
    direction = str(synthesis_draft["recommended_direction"])
    iteration_dir = run_dir / "iteration-001"
    evidence_paths = {
        "proposal": iteration_dir / "proposal.json",
        "evaluation": iteration_dir / "evaluation.json",
        "decision": iteration_dir / "decision.json",
        "lesson": iteration_dir / "lesson.json",
    }
    synthesis = ReviewSynthesis(
        review_hashes={role: review.content_sha256 for role, review in reviews.items()},
        recommended_mutation_class=recommended_key,
        recommended_direction=direction,
        rationale=str(synthesis_draft["rationale"]),
        accepted_recommendations=list(synthesis_draft["accepted_recommendations"]),
        rejected_recommendations=list(synthesis_draft["rejected_recommendations"]),
        iteration_evidence_sha256={
            name: sha256_file(path) for name, path in evidence_paths.items()
        },
        review_packet_sha256=packet.content_sha256,
    ).sealed()

    current = TransactionalConfigStore(run_dir / "config_store").current_config()
    old = current[recommended_key]
    new = _bounded_value(key=recommended_key, direction=direction, old=old, policy=policy)
    mutation = Mutation(
        target_type="config",
        target_path="config_store/current.json",
        target_key=recommended_key,
        old_value=old,
        new_value=new,
        allowed_range=policy.bounded_mutable[recommended_key],
        reversibility="reversible",
        scientific_effect=(
            f"Test the aggregate-only review synthesis direction for {recommended_key}; "
            "this is an infrastructure experiment, not a scientific selection."
        ),
        engineering_effect="Apply one bounded configuration change in the exact resumed run.",
    ).sealed()
    primary_metric = config.objective.success_criteria[0].metric
    proposal = ExperimentProposal(
        proposal_id=f"{config.objective.objective_id}-p002",
        objective_id=config.objective.objective_id,
        parent_iteration_id="iteration-001",
        hypothesis=(
            f"The evidence-bound review synthesis recommends a bounded {direction} move "
            f"for {recommended_key}; one exact-commit replay step will test that direction."
        ),
        proposed_mutations=[mutation],
        expected_mechanism=mutation.scientific_effect,
        expected_observables=[
            ExpectedObservable(
                metric=primary_metric,
                expected_direction="decrease",
                rationale="Only the independently evaluated aggregate metric can test the move.",
            )
        ],
        falsification_condition=(
            "The recommendation is falsified for this bounded infrastructure replay if the "
            "independent aggregate evaluator does not improve the primary metric or a resource "
            "or reproducibility constraint fails."
        ),
        estimated_compute=EstimatedCompute(device="gpu", estimated_seconds=600.0, remote=True),
        required_skills=[
            "tea_time_with_reading_poem",
            "mace_finetune",
            "mlip_metrics",
            f"review_synthesis:{synthesis.content_sha256}",
        ],
        risk_class="medium",
        required_approval=None,
        seed=config.fixture_seed + 1,
        proposal_policy_version="host-review-synthesis/1.0.0",
    ).sealed()

    legality = policy.check_mutations(
        proposal.proposal_id,
        proposal.proposed_mutations,
        current,
        allowed_classes=config.objective.allowed_mutation_classes,
    )
    if not legality.legal:
        raise ValueError(f"sealed iteration-two proposal is illegal: {legality.violations}")

    output_dir.mkdir(parents=True, exist_ok=False)
    for role, review in reviews.items():
        (output_dir / f"{role}.json").write_text(review.model_dump_json(indent=2) + "\n")
    (output_dir / "synthesis.json").write_text(synthesis.model_dump_json(indent=2) + "\n")
    (output_dir / "review_packet.json").write_text(packet.canonical_text())
    (output_dir / "proposal.json").write_text(proposal.model_dump_json(indent=2) + "\n")
    (output_dir / "READY").write_text("sealed\n")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--drafts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/ralphthon_mace_replay.yaml")
    )
    args = parser.parse_args()
    print(
        seal_bundle(
            run_dir=args.run_dir,
            drafts_dir=args.drafts_dir,
            output_dir=args.output_dir,
            config_path=args.config.resolve(),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
