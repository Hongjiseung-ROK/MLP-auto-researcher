#!/usr/bin/env python3
"""Run the H2 evidence-only MACE checkpoint/E0 comparison on D0 labels."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.data.manifests import NormalizedDataset
from mlip_research_agent.data.registry import NormalizedManifest, load_qualified_dataset
from mlip_research_agent.data.split import PartitionName, SplitManifest
from mlip_research_agent.research.preregistration import load_approvals
from mlip_research_agent.schemas.predictions import PredictionBatch
from mlip_research_agent.skills.atomistics.structures_io import StructureRecord, StructureSet
from mlip_research_agent.skills.base import SkillContext
from mlip_research_agent.skills.mlip.mace_inference.checkpoint import (
    MACECheckpointManifest,
    resolve_checkpoint,
)
from mlip_research_agent.skills.mlip.mace_inference.diagnostics import (
    compute_e0_diagnostic,
)
from mlip_research_agent.skills.mlip.mace_inference.implementation import MACEInferenceSkill
from mlip_research_agent.skills.mlip.mace_inference.schema import (
    MACEInferenceInput,
    MACEInferenceOutput,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = (
    (
        ROOT / "configs/models/mace_mp_0_small_candidate.json",
        ROOT / "data/cache/models/2023-12-10-mace-128-L0_energy_epoch-249.model",
    ),
    (
        ROOT / "configs/models/mace_mpa_0_medium_candidate.json",
        ROOT / "data/cache/models/mace-mpa-0-medium.model",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "data_registry/datasets/cu_phase2",
    )
    parser.add_argument(
        "--approvals",
        type=Path,
        default=ROOT / "docs/research/approvals.jsonl",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "artifacts/research/mace-checkpoint-comparison",
    )
    return parser.parse_args()


def _d0_view(dataset: NormalizedDataset, split: SplitManifest) -> NormalizedDataset:
    by_id = dataset.by_id()
    record_ids = split.record_ids[PartitionName.INITIAL_LABELED.value]
    return NormalizedDataset(
        dataset_id=dataset.dataset_id,
        level_of_theory=dataset.level_of_theory,
        configurations=[by_id[record_id] for record_id in record_ids],
    )


def _structure_set(dataset: NormalizedDataset) -> StructureSet:
    return StructureSet(
        systems=[
            StructureRecord(
                index=index,
                symbols=record.symbols,
                positions=record.positions,
                cell=record.cell,
                pbc=record.pbc,
                perturbation_scale=0.0,
            )
            for index, record in enumerate(dataset.configurations)
        ]
    )


def _cu_atomic_reference_energy(checkpoint_path: Path) -> float:
    import torch

    model: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    numbers_value: Any = model.atomic_numbers
    if hasattr(numbers_value, "detach"):
        numbers_value = numbers_value.detach().cpu().tolist()
    atomic_numbers = [int(number) for number in numbers_value]
    try:
        cu_index = atomic_numbers.index(29)
    except ValueError as exc:
        raise ValueError(f"checkpoint {checkpoint_path.name} does not contain Cu") from exc
    energies: Any = model.atomic_energies_fn.atomic_energies
    if hasattr(energies, "detach"):
        energies = energies.detach().cpu()
    if energies.ndim == 1:
        return float(energies[cu_index])
    if energies.ndim == 2 and energies.shape[0] == 1:
        return float(energies[0, cu_index])
    raise ValueError(
        f"checkpoint {checkpoint_path.name} has unsupported atomic-energy shape "
        f"{tuple(energies.shape)}"
    )


def main() -> int:
    args = _parse_args()
    dataset_dir = args.dataset_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = ArtifactRegistry(run_dir)

    approvals = load_approvals(args.approvals.resolve())
    qualified_dataset = load_qualified_dataset(dataset_dir, approvals)
    normalized_manifest_path = dataset_dir / "normalized_manifest.json"
    split_manifest_path = dataset_dir / "split_manifest.json"
    normalized_manifest = NormalizedManifest.load(normalized_manifest_path)
    split = SplitManifest.load(split_manifest_path)
    split.validate_against(normalized_manifest)
    d0 = _d0_view(qualified_dataset, split)

    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(normalized_manifest_path, inputs_dir / normalized_manifest_path.name)
    shutil.copyfile(split_manifest_path, inputs_dir / split_manifest_path.name)
    structures_path = inputs_dir / "initial_labeled_structures.json"
    _structure_set(d0).save(structures_path)
    registry.register(structures_path, "structure_set", "inputs")
    registry.register(
        inputs_dir / normalized_manifest_path.name,
        "normalized_dataset_manifest",
        "inputs",
    )
    registry.register(inputs_dir / split_manifest_path.name, "split_manifest", "inputs")
    registry.save()

    batches: list[PredictionBatch] = []
    cu_e0_by_model: dict[str, float] = {}
    for manifest_path, checkpoint_path in DEFAULT_CANDIDATES:
        manifest = MACECheckpointManifest.load(manifest_path)
        resolve_checkpoint(manifest, checkpoint_path)
        step_id = f"predict-{manifest.model_id}"
        ctx = SkillContext(
            run_dir=run_dir,
            step_id=step_id,
            seed=0,
            attempt=0,
            registry=registry,
        )
        output = MACEInferenceSkill().run(
            MACEInferenceInput(
                structures_path=structures_path.relative_to(run_dir).as_posix(),
                checkpoint_manifest_path=str(manifest_path),
                checkpoint_path=str(checkpoint_path),
                dataset_id=qualified_dataset.dataset_id,
                dataset_content_sha256=normalized_manifest.dataset_content_sha256,
                split_semantic_sha256=split.semantic_hash(),
                record_ids=[record.config_id for record in d0.configurations],
                device="cpu",
                default_dtype="float64",
            ),
            ctx,
        )
        if not isinstance(output, MACEInferenceOutput):
            raise TypeError("MACE inference returned an unexpected output model")
        prediction_path = run_dir / output.predictions_path
        batches.append(PredictionBatch.model_validate_json(prediction_path.read_text()))
        cu_e0_by_model[manifest.model_id] = _cu_atomic_reference_energy(checkpoint_path)
        registry.save()
        gc.collect()

    report = compute_e0_diagnostic(
        d0,
        normalized_manifest.dataset_content_sha256,
        split,
        batches,
        cu_e0_by_model,
    )
    diagnostic_dir = run_dir / "steps/e0-diagnostic"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    report_path = diagnostic_dir / "e0_diagnostic.json"
    report.save(report_path)
    registry.register(report_path, "e0_diagnostic", "e0-diagnostic")
    registry.save()

    summary = {
        "run_dir": str(run_dir),
        "selection_status": report.selection_status,
        "candidates": [
            {
                "model_id": candidate.model_id,
                "checkpoint_sha256": candidate.checkpoint_sha256,
                "mean_offset_ev_per_atom": candidate.mean_offset_ev_per_atom,
                "std_offset_ev_per_atom": candidate.std_offset_ev_per_atom,
                "mae_offset_ev_per_atom": candidate.mae_offset_ev_per_atom,
            }
            for candidate in report.candidates
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
