"""Mutation policy and transaction modeling."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Mutability(StrEnum):
    MUTABLE = "mutable"
    CONDITIONALLY_MUTABLE = "conditionally_mutable"
    IMMUTABLE = "immutable"


class MutationPolicy(BaseModel):
    """Configuration mapping config paths to mutability levels."""

    model_config = ConfigDict(extra="forbid")

    rules: dict[str, Mutability]

    def is_mutable(self, path: str) -> bool:
        """Check if a path is fully mutable."""
        return self.rules.get(path, Mutability.IMMUTABLE) == Mutability.MUTABLE


class MutationTransaction(BaseModel):
    """An exact, reversible mutation diff."""

    model_config = ConfigDict(extra="forbid")

    config_path: str = Field(min_length=1)
    old_value: Any
    new_value: Any

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
