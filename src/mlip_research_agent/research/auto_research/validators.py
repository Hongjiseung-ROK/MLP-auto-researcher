"""Shared validation plumbing for Auto Research contracts.

Every lineage node is content-addressed with the repository's canonical-JSON
idiom (sort_keys, compact separators — the same convention as
``research/preregistration.py`` and ``research/program_guardrail.py``). The
``content_sha256`` field is excluded from its own hash so a sealed record can
be re-verified byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, TypeVar

from pydantic import BaseModel, Field

#: JSON scalar permitted in configurations and mutations. ``bool`` precedes
#: ``int`` so pydantic does not coerce True/False into 1/0.
ConfigValue = bool | int | float | str | None

SHA256_HEX_LENGTH = 64

_UNSEALED = ""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


M = TypeVar("M", bound="ContentAddressedModel")


class ContentAddressedModel(BaseModel):
    """Base for lineage nodes: deterministic serialization + content hash."""

    schema_version: str = "1.0.0"
    content_sha256: str = Field(
        default=_UNSEALED,
        description="SHA-256 over the canonical JSON of every other field; empty until sealed",
    )

    def compute_content_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("content_sha256", None)
        return sha256_of_text(canonical_json(payload))

    def sealed(self: M) -> M:
        """Return a copy with ``content_sha256`` recorded."""
        update = {"content_sha256": self.compute_content_sha256()}
        return self.model_copy(update=update)

    @property
    def is_sealed(self) -> bool:
        return self.content_sha256 != _UNSEALED

    def verify_seal(self) -> bool:
        return self.is_sealed and self.content_sha256 == self.compute_content_sha256()

    def canonical_text(self) -> str:
        """Deterministic serialization used for artifact files (stable across reruns)."""
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


class SealError(Exception):
    """Raised when a lineage node's content hash is missing or wrong."""


def require_sealed(model: ContentAddressedModel, label: str) -> None:
    if not model.verify_seal():
        raise SealError(f"{label} is not sealed or its content_sha256 does not verify")
