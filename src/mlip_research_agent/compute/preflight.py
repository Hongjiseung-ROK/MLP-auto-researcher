"""Colab preflight records: VESSL submission requires a passing, non-stale one.

A preflight is valid for a VESSL job only when repo commit, dependency spec,
MLIP backend, CUDA compatibility class, and workflow schema version all match,
it passed, and it is younger than the policy's max age.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mlip_research_agent.compute.schemas import JobSpec

PREFLIGHT_RECORD_SUFFIX = ".preflight.json"


class PreflightRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1.0"
    run_id: str
    passed: bool
    repo_commit: str
    dependency_hash: str
    mlip_backend: str
    cuda_compat_class: str
    workflow_schema_version: str
    created_at: str  # ISO-8601 UTC

    def matches(self, spec: JobSpec) -> bool:
        return (
            self.repo_commit == spec.repo_commit
            and self.dependency_hash == spec.dependency_hash
            and self.mlip_backend == spec.mlip_backend
            and self.cuda_compat_class == spec.cuda_compat_class
            and self.workflow_schema_version == spec.workflow_schema_version
        )

    def is_stale(self, max_age_hours: int, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        created = datetime.fromisoformat(self.created_at)
        return now - created > timedelta(hours=max_age_hours)


class PreflightStore:
    """Directory-backed store of preflight records."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, record: PreflightRecord) -> Path:
        path = self.directory / f"{record.run_id}{PREFLIGHT_RECORD_SUFFIX}"
        path.write_text(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
        return path

    def all(self) -> list[PreflightRecord]:
        records = [
            PreflightRecord.model_validate_json(path.read_text())
            for path in sorted(self.directory.glob(f"*{PREFLIGHT_RECORD_SUFFIX}"))
        ]
        return sorted(records, key=lambda r: r.created_at)

    def find_valid(
        self, spec: JobSpec, max_age_hours: int, now: datetime | None = None
    ) -> PreflightRecord | None:
        """Newest passing, matching, non-stale preflight for this spec, if any."""
        for record in reversed(self.all()):
            if record.passed and record.matches(spec) and not record.is_stale(max_age_hours, now):
                return record
        return None
