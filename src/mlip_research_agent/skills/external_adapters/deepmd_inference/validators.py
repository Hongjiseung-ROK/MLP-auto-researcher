"""Validation for the DeePMD inference adapter."""

from __future__ import annotations

from pathlib import Path

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.skills.external_adapters.common import adapter_error
from mlip_research_agent.skills.external_adapters.validators_shared import (
    finite_structure_records,
)


def resolve_pinned_model(model_path: str, expected_sha256: str) -> Path:
    """Verify the local model file's hash before any deserialization."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise adapter_error(f"DeePMD model file not found: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise adapter_error(
            f"DeePMD model hash mismatch: expected {expected_sha256}, got {actual}"
        )
    return path


__all__ = ["finite_structure_records", "resolve_pinned_model"]
