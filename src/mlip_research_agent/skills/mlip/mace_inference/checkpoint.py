"""Content-addressed MACE checkpoint manifest and fail-closed resolver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class MACECheckpointManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    family: str
    checkpoint_filename: str
    source_url: str
    checkpoint_sha256: str = Field(min_length=64, max_length=64)
    checkpoint_size_bytes: int = Field(gt=0)
    license_spdx: str
    mace_torch_version: str
    supported_species: list[str] = Field(min_length=1)
    training_data_statement: str
    scientific_status: str

    @classmethod
    def load(cls, path: Path) -> MACECheckpointManifest:
        return cls.model_validate(json.loads(path.read_text()))

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


class ResolvedCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    size_bytes: int
    manifest_sha256: str
    model_id: str


class CheckpointResolutionError(ValueError):
    """Checkpoint bytes or environment violate the pinned manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(
    manifest: MACECheckpointManifest, checkpoint_path: Path
) -> ResolvedCheckpoint:
    """Verify exact filename, size, and SHA before MACE can deserialize bytes."""
    if not checkpoint_path.is_file():
        raise CheckpointResolutionError(
            f"checkpoint not found: {checkpoint_path}; fetch the pinned source URL into "
            "the gitignored model cache first"
        )
    if checkpoint_path.name != manifest.checkpoint_filename:
        raise CheckpointResolutionError(
            f"checkpoint filename mismatch: expected {manifest.checkpoint_filename}, "
            f"got {checkpoint_path.name}"
        )
    size = checkpoint_path.stat().st_size
    if size != manifest.checkpoint_size_bytes:
        raise CheckpointResolutionError(
            f"checkpoint size mismatch: expected {manifest.checkpoint_size_bytes}, got {size}"
        )
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != manifest.checkpoint_sha256:
        raise CheckpointResolutionError(
            f"checkpoint SHA-256 mismatch: expected {manifest.checkpoint_sha256}, got {actual_sha}"
        )
    return ResolvedCheckpoint(
        path=str(checkpoint_path.resolve()),
        sha256=actual_sha,
        size_bytes=size,
        manifest_sha256=manifest.content_hash(),
        model_id=manifest.model_id,
    )
