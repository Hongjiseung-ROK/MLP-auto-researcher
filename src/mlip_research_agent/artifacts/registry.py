"""Content-addressed artifact registry.

Manifest entries deliberately contain no timestamps so that a rerun with the
same inputs and seed produces a byte-identical manifest (timestamps live in
the event log).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

MANIFEST_NAME = "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    relative_path: str
    sha256: str
    size_bytes: int
    kind: str
    created_by_step: str


class ArtifactRegistry:
    """Registers files produced under a run directory and persists a manifest."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._entries: dict[str, Artifact] = {}

    def register(self, path: Path, kind: str, step_id: str) -> Artifact:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"cannot register missing artifact file: {path}")
        rel = path.relative_to(self.run_dir.resolve()).as_posix()
        artifact = Artifact(
            artifact_id=f"{step_id}:{path.name}",
            relative_path=rel,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            kind=kind,
            created_by_step=step_id,
        )
        existing = self._entries.get(artifact.artifact_id)
        if existing is not None and existing != artifact:
            raise ValueError(
                f"artifact id collision with different content: {artifact.artifact_id}"
            )
        self._entries[artifact.artifact_id] = artifact
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        return self._entries.get(artifact_id)

    def verify(self, artifact_id: str) -> bool:
        """True iff the artifact exists on disk and its hash still matches."""
        entry = self._entries.get(artifact_id)
        if entry is None:
            return False
        path = self.run_dir / entry.relative_path
        return path.is_file() and sha256_file(path) == entry.sha256

    def all(self) -> list[Artifact]:
        return sorted(self._entries.values(), key=lambda a: a.artifact_id)

    def save(self) -> Path:
        manifest_path = self.run_dir / MANIFEST_NAME
        payload = [a.model_dump() for a in self.all()]
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return manifest_path

    @classmethod
    def load(cls, run_dir: Path) -> ArtifactRegistry:
        registry = cls(run_dir)
        manifest_path = run_dir / MANIFEST_NAME
        if manifest_path.is_file():
            raw = json.loads(manifest_path.read_text())
            for item in raw:
                artifact = Artifact.model_validate(item)
                registry._entries[artifact.artifact_id] = artifact
        return registry
