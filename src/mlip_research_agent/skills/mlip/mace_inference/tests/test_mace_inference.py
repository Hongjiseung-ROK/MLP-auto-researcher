"""Checkpoint-integrity tests plus opt-in real CPU MACE invariance fixture."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.schemas.predictions import PredictionBatch
from mlip_research_agent.skills.atomistics.structures_io import (
    StructureSet,
    atoms_to_record,
)
from mlip_research_agent.skills.base import SkillContext
from mlip_research_agent.skills.mlip.mace_inference.checkpoint import (
    CheckpointResolutionError,
    MACECheckpointManifest,
    resolve_checkpoint,
)
from mlip_research_agent.skills.mlip.mace_inference.implementation import MACEInferenceSkill
from mlip_research_agent.skills.mlip.mace_inference.schema import (
    MACEInferenceInput,
    MACEInferenceOutput,
)

ROOT = Path(__file__).resolve().parents[6]
MANIFEST_PATH = ROOT / "configs/models/mace_mp_0_small_candidate.json"
MPA_MANIFEST_PATH = ROOT / "configs/models/mace_mpa_0_medium_candidate.json"


def test_checkpoint_hash_mismatch_aborts_before_model_load(tmp_path: Path) -> None:
    manifest = MACECheckpointManifest.load(MANIFEST_PATH)
    fake = tmp_path / manifest.checkpoint_filename
    fake.write_bytes(b"not a model")
    with pytest.raises(CheckpointResolutionError, match="size mismatch"):
        resolve_checkpoint(manifest, fake)


def test_checkpoint_manifest_is_candidate_not_h2_selection() -> None:
    manifest = MACECheckpointManifest.load(MANIFEST_PATH)
    assert manifest.scientific_status == "candidate_only"
    assert manifest.mace_torch_version == "0.3.16"
    assert "Cu" in manifest.supported_species
    comparison = MACECheckpointManifest.load(MPA_MANIFEST_PATH)
    assert comparison.scientific_status == "candidate_only"
    assert comparison.checkpoint_sha256 == (
        "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"
    )
    assert "Cu" in comparison.supported_species


def _real_checkpoint() -> Path:
    value = os.environ.get("MACE_TEST_CHECKPOINT")
    if value is None:
        pytest.skip("set MACE_TEST_CHECKPOINT for the opt-in real CPU fixture")
    return Path(value).resolve()


def _real_manifest() -> Path:
    value = os.environ.get("MACE_TEST_MANIFEST")
    return MANIFEST_PATH if value is None else Path(value).resolve()


def test_real_cpu_inference_invariance_and_provenance(tmp_path: Path) -> None:
    checkpoint = _real_checkpoint()
    manifest_path = _real_manifest()
    base = bulk("Cu", "fcc", a=3.6, cubic=True)
    translated = base.copy()
    translated.translate([0.31, -0.27, 0.19])
    rotated = base.copy()
    rotated.rotate(90.0, "z", rotate_cell=True)
    permutation = [2, 0, 3, 1]
    permuted = base[permutation]
    structures = StructureSet(
        systems=[
            atoms_to_record(base, 0, 0.0),
            atoms_to_record(translated, 1, 0.0),
            atoms_to_record(rotated, 2, 0.0),
            atoms_to_record(permuted, 3, 0.0),
        ]
    )
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    structures_path = inputs_dir / "structures.json"
    structures.save(structures_path)
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="mace-inference",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )
    output = MACEInferenceSkill().run(
        MACEInferenceInput(
            structures_path="inputs/structures.json",
            checkpoint_manifest_path=str(manifest_path),
            checkpoint_path=str(checkpoint),
            dataset_id="mace-inference-fixture",
            dataset_content_sha256="d" * 64,
            split_semantic_sha256="e" * 64,
            record_ids=["base", "translated", "rotated", "permuted"],
            device="cpu",
            default_dtype="float64",
        ),
        ctx,
    )
    assert isinstance(output, MACEInferenceOutput)
    payload = json.loads((tmp_path / output.predictions_path).read_text())
    PredictionBatch.model_validate(payload)
    predictions = payload["predictions"]
    assert [item["record_id"] for item in predictions] == [
        "base",
        "translated",
        "rotated",
        "permuted",
    ]
    assert payload["energy_unit"] == "eV"
    assert payload["force_unit"] == "eV/angstrom"
    energies = np.asarray([item["energy_ev"] for item in predictions])
    assert np.allclose(energies, energies[0], rtol=0.0, atol=1e-8)
    base_forces = np.asarray(predictions[0]["forces_ev_per_a"])
    assert np.allclose(predictions[1]["forces_ev_per_a"], base_forces, atol=1e-8)
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert np.allclose(
        predictions[2]["forces_ev_per_a"], base_forces @ rotation.T, atol=1e-7
    )
    permuted_forces = np.asarray(predictions[3]["forces_ev_per_a"])
    assert np.allclose(permuted_forces, base_forces[permutation], atol=1e-8)
    assert output.n_structures == 4
    assert len(ctx.registry.all()) == 2
    assert all(ctx.registry.verify(item.artifact_id) for item in ctx.registry.all())
    model_manifest = json.loads((tmp_path / output.model_manifest_path).read_text())
    assert model_manifest["checkpoint"]["sha256"] == output.checkpoint_sha256
    assert model_manifest["device"] == "cpu"
