"""Contract-level tests: seals, mutation policy, transactional config store."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_research_agent.research.auto_research import (
    AllowedRange,
    IterationLineage,
    LineageNode,
    LineageNodeKind,
    Mutation,
    MutationClass,
    MutationPolicy,
    ResearchLesson,
    TransactionalConfigStore,
)
from mlip_research_agent.research.auto_research.objective import LocalDemoConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def policy() -> MutationPolicy:
    return MutationPolicy.load(REPO_ROOT / "configs/research/ralphthon_mutation_policy.yaml")


def make_mutation(key: str = "learning_rate", old: float = 0.02, new: float = 0.004) -> Mutation:
    return Mutation(
        target_type="config",
        target_path="config_store/current.json",
        target_key=key,
        old_value=old,
        new_value=new,
        allowed_range=AllowedRange(value_type="float", minimum=1e-6, maximum=0.1),
        reversibility="reversible",
        scientific_effect="Moves optimization toward the loss basin.",
        engineering_effect="Smaller update steps.",
    ).sealed()


def test_deterministic_serialization_and_stable_hash() -> None:
    m1 = make_mutation()
    m2 = make_mutation()
    assert m1.content_sha256 == m2.content_sha256
    assert m1.canonical_text() == m2.canonical_text()
    assert m1.verify_seal()


def test_tampered_seal_fails() -> None:
    mutation = make_mutation()
    tampered = mutation.model_copy(update={"new_value": 0.05})
    assert not tampered.verify_seal()


def test_invalid_parent_lineage_rejected() -> None:
    node = LineageNode(
        kind=LineageNodeKind.PROPOSAL,
        node_id="iteration-002:proposal.json",
        artifact_id="iteration-002:proposal.json",
        content_sha256="0" * 64,
    )
    with pytest.raises(ValidationError, match=r"out of chain order|repeat"):
        IterationLineage(
            iteration_id="iteration-002",
            parent_iteration_id="iteration-001",
            nodes=[node, node],
        )


def test_immutable_parameter_rejected(policy: MutationPolicy) -> None:
    mutation = Mutation(
        target_type="config",
        target_path="config_store/current.json",
        target_key="dataset_content_sha256",
        old_value="a" * 64,
        new_value="b" * 64,
        allowed_range=None,
        reversibility="reversible",
        scientific_effect="Swap the dataset silently.",
        engineering_effect="Should never be possible.",
    ).sealed()
    result = policy.check_mutations(
        "prop-x",
        [mutation],
        {"dataset_content_sha256": "a" * 64},
        allowed_classes=[MutationClass.BOUNDED_MUTABLE],
    )
    assert not result.legal
    assert any("immutable" in v for v in result.violations)


def test_unknown_key_is_immutable_fail_closed(policy: MutationPolicy) -> None:
    assert policy.classify("brand_new_key") is MutationClass.IMMUTABLE


def test_owner_gated_checkpoint_rejected(policy: MutationPolicy) -> None:
    mutation = Mutation(
        target_type="config",
        target_path="config_store/current.json",
        target_key="checkpoint_id",
        old_value="mace-mp-0-small",
        new_value="mace-mpa-0-medium",
        allowed_range=None,
        reversibility="reversible",
        scientific_effect="Change the checkpoint family.",
        engineering_effect="Different weights.",
    ).sealed()
    result = policy.check_mutations(
        "prop-x",
        [mutation],
        {"checkpoint_id": "mace-mp-0-small"},
        allowed_classes=[MutationClass.BOUNDED_MUTABLE],
    )
    assert not result.legal
    assert any("owner-gated" in v for v in result.violations)


def test_bounded_mutation_accepted(policy: MutationPolicy) -> None:
    mutation = Mutation(
        target_type="config",
        target_path="config_store/current.json",
        target_key="learning_rate",
        old_value=0.02,
        new_value=0.004,
        allowed_range=policy.bounded_mutable["learning_rate"],
        reversibility="reversible",
        scientific_effect="Moves optimization toward the loss basin.",
        engineering_effect="Smaller update steps.",
    ).sealed()
    result = policy.check_mutations(
        "prop-x",
        [mutation],
        {"learning_rate": 0.02},
        allowed_classes=[MutationClass.BOUNDED_MUTABLE],
    )
    assert result.legal


def test_out_of_bounds_mutation_rejected(policy: MutationPolicy) -> None:
    mutation = Mutation(
        target_type="config",
        target_path="config_store/current.json",
        target_key="learning_rate",
        old_value=0.02,
        new_value=0.5,  # above the 0.1 maximum
        allowed_range=policy.bounded_mutable["learning_rate"],
        reversibility="reversible",
        scientific_effect="Way too large a step.",
        engineering_effect="Would escape bounds.",
    ).sealed()
    result = policy.check_mutations(
        "prop-x",
        [mutation],
        {"learning_rate": 0.02},
        allowed_classes=[MutationClass.BOUNDED_MUTABLE],
    )
    assert not result.legal
    assert any("outside declared bounds" in v for v in result.violations)


def test_rejected_mutation_rolls_back(tmp_path: Path) -> None:
    store = TransactionalConfigStore(tmp_path / "store")
    store.initialize({"learning_rate": 0.02})
    store.apply("prop-1", [make_mutation()])
    assert store.candidate_config() == {"learning_rate": 0.004}
    store.rollback()
    assert store.current_config() == {"learning_rate": 0.02}
    assert not store.has_pending_candidate()
    # History is preserved, not rewritten.
    assert (tmp_path / "store" / "state-001.json").is_file()


def test_accepted_mutation_persists_and_history_kept(tmp_path: Path) -> None:
    store = TransactionalConfigStore(tmp_path / "store")
    store.initialize({"learning_rate": 0.02})
    store.apply("prop-1", [make_mutation()])
    store.commit()
    assert store.current_config() == {"learning_rate": 0.004}
    # Previous accepted state remains readable and unchanged.
    assert store.read_state(0) == {"learning_rate": 0.02}


def test_history_files_are_append_only(tmp_path: Path) -> None:
    store = TransactionalConfigStore(tmp_path / "store")
    store.initialize({"learning_rate": 0.02})
    store.apply("prop-1", [make_mutation()])
    with pytest.raises(ValueError, match="pending"):
        store.apply("prop-2", [make_mutation()])


def test_lesson_cannot_recommend_protected_classes() -> None:
    with pytest.raises(ValidationError, match="immutable or owner-gated"):
        ResearchLesson(
            lesson_id="lesson-x",
            iteration_id="iteration-001",
            attempt_summary="Tried mutating the learning rate downward.",
            observed_result="The loss decreased on the synthetic fixture.",
            evidence_references=["iteration-001:decision.json"],
            likely_explanation="Smaller steps land nearer the basin optimum.",
            alternative_explanations=["Noise floor masks the true effect."],
            limits="Synthetic fixture only; no scientific transfer.",
            applicability_conditions="This fixture landscape and policy bounds only.",
            anti_pattern="Do not repeat the identical move.",
            recommended_next_mutation_class=MutationClass.OWNER_GATED,
        )


def test_demo_config_loads_and_objective_is_bounded() -> None:
    demo = LocalDemoConfig.load(REPO_ROOT / "configs/research/ralphthon_local_demo.yaml")
    objective = demo.objective.sealed()
    assert objective.verify_seal()
    assert objective.budget.max_remote_jobs == 0
    assert objective.scientific_status_ceiling.value == "non_scientific"
    assert MutationClass.IMMUTABLE not in objective.allowed_mutation_classes
