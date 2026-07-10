"""Deterministic artifact hashing and manifest tests."""

from pathlib import Path

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry, sha256_file


def test_hash_depends_only_on_content(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{\"x\": 1}")
    (tmp_path / "b.json").write_text("{\"x\": 1}")
    assert sha256_file(tmp_path / "a.json") == sha256_file(tmp_path / "b.json")
    (tmp_path / "c.json").write_text("{\"x\": 2}")
    assert sha256_file(tmp_path / "a.json") != sha256_file(tmp_path / "c.json")


def test_manifest_roundtrip_is_byte_identical(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    step_dir = tmp_path / "steps" / "s1"
    step_dir.mkdir(parents=True)
    (step_dir / "data.json").write_text("{}")
    registry.register(step_dir / "data.json", kind="test", step_id="s1")
    manifest_path = registry.save()
    first = manifest_path.read_bytes()

    reloaded = ArtifactRegistry.load(tmp_path)
    assert [a.artifact_id for a in reloaded.all()] == ["s1:data.json"]
    reloaded.save()
    assert manifest_path.read_bytes() == first


def test_verify_detects_tampering(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    target = tmp_path / "steps" / "s1" / "data.json"
    target.parent.mkdir(parents=True)
    target.write_text("original")
    artifact = registry.register(target, kind="test", step_id="s1")
    assert registry.verify(artifact.artifact_id)
    target.write_text("tampered")
    assert not registry.verify(artifact.artifact_id)
    assert not registry.verify("s1:never-registered.json")


def test_missing_file_rejected(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.register(tmp_path / "ghost.json", kind="test", step_id="s1")


def test_id_collision_with_different_content_rejected(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    target = tmp_path / "steps" / "s1" / "data.json"
    target.parent.mkdir(parents=True)
    target.write_text("v1")
    registry.register(target, kind="test", step_id="s1")
    target.write_text("v2")
    with pytest.raises(ValueError, match="collision"):
        registry.register(target, kind="test", step_id="s1")
