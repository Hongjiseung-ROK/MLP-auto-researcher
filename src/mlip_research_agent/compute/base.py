"""ComputeProvider protocol and shared remote-provider machinery."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from mlip_research_agent.compute.schemas import (
    ArtifactManifest,
    AuthStatus,
    CancellationResult,
    CapacityReport,
    JobHandle,
    JobSpec,
    JobStatus,
    LogEvent,
    RemoteArtifact,
)


class ComputeError(Exception):
    """Base error for the compute layer."""


class ComputePolicyDenied(ComputeError):
    """A job was refused by policy. Carries structured reasons; fail-closed."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons) or "denied by compute policy")
        self.reasons = reasons


@runtime_checkable
class ComputeProvider(Protocol):
    def validate_auth(self) -> AuthStatus: ...

    def inspect_capacity(self) -> CapacityReport: ...

    def submit(self, spec: JobSpec) -> JobHandle: ...

    def status(self, handle: JobHandle) -> JobStatus: ...

    def stream_logs(self, handle: JobHandle) -> Iterable[LogEvent]: ...

    def collect_artifacts(self, handle: JobHandle) -> ArtifactManifest: ...

    def cancel(self, handle: JobHandle) -> CancellationResult: ...


def new_run_id(spec: JobSpec) -> str:
    return f"{spec.name}-{uuid.uuid4().hex[:8]}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_from_directory(job_id: str, directory: Path) -> ArtifactManifest:
    artifacts = []
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                artifacts.append(
                    RemoteArtifact(
                        path=path.relative_to(directory).as_posix(),
                        sha256=sha256_bytes(path.read_bytes()),
                        size_bytes=path.stat().st_size,
                    )
                )
    return ArtifactManifest(job_id=job_id, artifacts=artifacts)


def write_job_record(run_dir: Path, name: str, payload: dict[str, object]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path
