"""Shared label-free selection vocabulary for the WP6 acquisition skills.

Selection skills never see energies, forces, stresses, hidden labels, or any
label-derived error. Candidate metadata is restricted to an explicit
allowlist; anything else is rejected fail-closed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError

#: The only candidate metadata keys a selection skill may receive.
ALLOWED_METADATA_KEYS = frozenset({"source_group", "n_atoms", "formula"})

#: Key fragments that indicate label or label-derived data.
FORBIDDEN_METADATA_FRAGMENTS = (
    "energy",
    "force",
    "stress",
    "label",
    "error",
    "mae",
    "rmse",
    "target",
)


class CandidateMeta(BaseModel):
    """Permitted, label-free candidate metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_group: str = Field(min_length=1)
    n_atoms: int = Field(gt=0)
    formula: str = ""


class SelectionRecord(BaseModel):
    """One machine-readable selection decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    uncertainty_score: float | None = None
    diversity_distance: float | None = None
    source_group: str = Field(min_length=1)
    rank: int = Field(ge=0)
    selection_reason: str = Field(min_length=5, max_length=500)
    policy_version: str = Field(min_length=1)


def reject(message: str, *, retryable: bool = False) -> SkillError:
    return SkillError(
        message,
        failure_class=FailureClass.VALIDATION_ERROR,
        severity=Severity.HIGH,
        retryable=retryable,
    )


def validate_candidate_pool(
    candidate_ids: list[str], metadata: dict[str, CandidateMeta]
) -> None:
    """Duplicates, unknown metadata, and label-shaped keys are all fatal."""
    if len(set(candidate_ids)) != len(candidate_ids):
        duplicates = sorted({c for c in candidate_ids if candidate_ids.count(c) > 1})
        raise reject(f"duplicate candidate ids: {duplicates[:5]}")
    unknown = sorted(set(metadata) - set(candidate_ids))
    if unknown:
        raise reject(f"metadata for ids outside the pool: {unknown[:5]}")


def assert_label_free_keys(keys: list[str], *, context: str) -> None:
    """Fail-closed guard against label-shaped payload keys."""
    for key in keys:
        lowered = key.lower()
        for fragment in FORBIDDEN_METADATA_FRAGMENTS:
            if fragment in lowered:
                raise reject(
                    f"{context}: key {key!r} looks label-derived ({fragment!r}); "
                    "selection skills are label-free by contract"
                )
