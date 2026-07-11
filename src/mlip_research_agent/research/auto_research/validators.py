"""Validation logic for proposed mutations."""

from __future__ import annotations

from typing import Any

from mlip_research_agent.research.auto_research.mutation_policy import (
    MutationPolicy,
    MutationTransaction,
)


def validate_mutation(
    policy: MutationPolicy,
    transaction: MutationTransaction,
    current_config: dict[str, Any]
) -> bool:
    """Check if the proposed mutation is legal under the policy.

    Fails before execution if illegal.
    """
    if not policy.is_mutable(transaction.config_path):
        raise ValueError(f"Mutation illegal: path {transaction.config_path} is not mutable.")

    # In a full implementation, this would also check bounds.
    return True
