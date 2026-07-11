"""Tests for auto_research validators."""
import pytest

from mlip_research_agent.research.auto_research.mutation_policy import (
    Mutability,
    MutationPolicy,
    MutationTransaction,
)
from mlip_research_agent.research.auto_research.validators import validate_mutation


def test_validate_mutation_legal() -> None:
    policy = MutationPolicy(rules={"path/to/val": Mutability.MUTABLE})
    transaction = MutationTransaction(
        config_path="path/to/val",
        old_value=1,
        new_value=2
    )
    # Should not raise
    assert validate_mutation(policy, transaction, {}) is True

def test_validate_mutation_illegal() -> None:
    policy = MutationPolicy(rules={"path/to/val": Mutability.IMMUTABLE})
    transaction = MutationTransaction(
        config_path="path/to/val",
        old_value=1,
        new_value=2
    )
    with pytest.raises(ValueError, match=r"Mutation illegal: path path/to/val is not mutable\."):
        validate_mutation(policy, transaction, {})

    policy = MutationPolicy(rules={"path/to/val": Mutability.CONDITIONALLY_MUTABLE})
    with pytest.raises(ValueError, match=r"Mutation illegal: path path/to/val is not mutable\."):
        validate_mutation(policy, transaction, {})

    policy = MutationPolicy(rules={})
    transaction = MutationTransaction(
        config_path="unspecified_path",
        old_value=1,
        new_value=2
    )
    with pytest.raises(ValueError, match=r"Mutation illegal: path unspecified_path is not mutable\."):  # noqa: E501
        validate_mutation(policy, transaction, {})
