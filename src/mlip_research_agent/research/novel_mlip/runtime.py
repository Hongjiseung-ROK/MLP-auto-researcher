"""Real-GPU baseline and sequential round execution for the frozen campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from mlip_research_agent.artifacts.registry import ArtifactRegistry, sha256_file
from mlip_research_agent.data.manifests import LabeledConfiguration, NormalizedDataset
from mlip_research_agent.data.split import PartitionName, SplitManifest
from mlip_research_agent.research.novel_mlip.metrics import (
    evaluate_predictions,
    max_prediction_difference,
    mean_predictions,
)
from mlip_research_agent.research.novel_mlip.schema import (
    ArmName,
    ArmState,
    CampaignState,
    ModelIdentity,
    SelectionRound,
)
from mlip_research_agent.research.novel_mlip.selection import (
    committee_scores,
    select_candidates,
    viability_summary,
)
from mlip_research_agent.skills.base import SkillContext
from mlip_research_agent.skills.mlip.mace_finetune.runner import run_controlled_training
from mlip_research_agent.skills.mlip.mace_inference.checkpoint import (
    MACECheckpointManifest,
    resolve_checkpoint,
)
from mlip_research_agent.skills.mlip.mace_inference.implementation import _make_calculator
from mlip_research_agent.skills.reflection.implementation import (
    TeaTimeWithReadingPoemSkill,
)
from mlip_research_agent.skills.reflection.schema import (
    AlternativePath,
    TeaTimeInput,
    TeaTimeTrigger,
)

SPEC_SHA256 = "9dea3391d6285a2ab7590a3cf18ad0f76992585d31d047eca892e63fb97a92d3"
DATASET_CONTENT_SHA256 = "bc9b78ca5e3b95f91bf34bbc3641a3d6e3f92338b4e3d97065165157848cfc48"
DATASET_FILE_SHA256 = "6c0fc583ead5e028ed72227ab864ea88fb8a9942a84d03d0558a890f5b4dba8f"
SPLIT_SEMANTIC_SHA256 = "3f909d7aaf10ee959aaebf4538e82fb4f391cba4cad5f5fd44850939629ef457"
CHECKPOINT_SHA256 = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"
SEEDS = (42, 43, 44)
ARMS: tuple[ArmName, ...] = (
    "random",
    "disagreement",
    "disagreement_fps",
    "tail_risk_fps",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_sha(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return sha256_file(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_inputs(
    *,
    dataset_path: Path,
    split_path: Path,
    checkpoint_path: Path,
    checkpoint_manifest_path: Path,
    spec_path: Path,
    authorization_path: Path,
) -> tuple[NormalizedDataset, SplitManifest, MACECheckpointManifest]:
    if sha256_file(spec_path) != SPEC_SHA256:
        raise ValueError("frozen research-spec hash mismatch")
    authorization = json.loads(authorization_path.read_text())
    if authorization.get("authorization_status") != "approved":
        raise ValueError("campaign authorization is not approved")
    if authorization.get("research_spec_sha256") != SPEC_SHA256:
        raise ValueError("campaign authorization does not bind the frozen research spec")
    required_actions = {
        "real_gpu_inference",
        "real_gpu_fine_tuning",
        "real_active_learning_selection",
        "oracle_label_reveal_within_frozen_budget",
        "multi_round_retraining",
    }
    if not required_actions <= set(authorization.get("authorized_actions", [])):
        raise ValueError("campaign authorization is missing required scientific actions")
    expires_at = datetime.fromisoformat(
        str(authorization["expires_at"]).replace("Z", "+00:00")
    )
    if expires_at < datetime.now(UTC):
        raise ValueError("campaign authorization has expired")
    if sha256_file(dataset_path) != DATASET_FILE_SHA256:
        raise ValueError("normalized dataset file hash mismatch")
    dataset = NormalizedDataset.load(dataset_path)
    if dataset.content_hash() != DATASET_CONTENT_SHA256:
        raise ValueError("normalized dataset content hash mismatch")
    split = SplitManifest.load(split_path)
    if split.semantic_hash() != SPLIT_SEMANTIC_SHA256:
        raise ValueError("frozen split semantic hash mismatch")
    if split.qualified_dataset_content_sha256 != DATASET_CONTENT_SHA256:
        raise ValueError("split is not bound to the campaign dataset")
    checkpoint_manifest = MACECheckpointManifest.load(checkpoint_manifest_path)
    resolved = resolve_checkpoint(checkpoint_manifest, checkpoint_path)
    if resolved.sha256 != CHECKPOINT_SHA256:
        raise ValueError("foundation checkpoint hash mismatch")
    return dataset, split, checkpoint_manifest


def _hardware() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the scientific campaign requires a visible CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("the scientific campaign permits exactly one visible GPU")
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
    }


def _atoms(record: LabeledConfiguration) -> Any:
    return Atoms(
        symbols=record.symbols,
        positions=np.asarray(record.positions),
        cell=np.asarray(record.cell),
        pbc=record.pbc,
    )


def _predict(
    model_path: Path,
    records: list[LabeledConfiguration],
    *,
    calculator: Any | None = None,
) -> dict[str, dict[str, Any]]:
    local_calculator = calculator or _make_calculator(model_path, "cuda", "float32")
    predictions: dict[str, dict[str, Any]] = {}
    for record in records:
        atoms: Any = _atoms(record)
        atoms.calc = local_calculator
        predictions[record.config_id] = {
            "energy_ev": float(atoms.get_potential_energy()),
            "forces_ev_per_a": np.asarray(atoms.get_forces(), dtype=float).tolist(),
        }
    return predictions


def _training_data_sha(records: list[LabeledConfiguration]) -> str:
    return _canonical_sha(
        [
            {"record_id": record.config_id, "content_sha256": record.content_hash()}
            for record in sorted(records, key=lambda item: item.config_id)
        ]
    )


def _fit_model(
    *,
    root: Path,
    foundation_path: Path,
    train_records: list[LabeledConfiguration],
    validation_records: list[LabeledConfiguration],
    arm: str,
    round_index: int,
    seed: int,
    learning_rate: float,
) -> ModelIdentity:
    import torch

    output_dir = root / "models" / arm / f"round-{round_index}" / f"seed-{seed}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite model fit: {output_dir}")
    output_dir.mkdir(parents=True)
    model_path = output_dir / "model.model"
    checkpoint_path = output_dir / "training-checkpoint.pt"
    n_batches = math.ceil(len(train_records) / 4)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = run_controlled_training(
        foundation_path=foundation_path,
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        train_records=train_records,
        validation_records=validation_records,
        resume_checkpoint_path=None,
        resume_contract_sha256=_canonical_sha(
            {
                "campaign": "cu-tail-risk-ralph-20260712",
                "arm": arm,
                "round": round_index,
                "seed": seed,
                "training_data_sha256": _training_data_sha(train_records),
                "epoch_mode": "full_epoch",
            }
        ),
        seed=seed,
        optimizer_name="adamw",
        learning_rate=learning_rate,
        gradient_clip=10.0,
        batch_size=4,
        valid_batch_size=4,
        max_epochs=20,
        max_optimizer_steps=20 * n_batches,
        patience=20,
        device_name="cuda",
        default_dtype="float32",
        max_wall_seconds=1200,
        energy_loss_weight=1.0,
        force_loss_weight=100.0,
        trainable_layer_policy="last_interaction_and_readout",
        epoch_mode="full_epoch",
    )
    torch.cuda.synchronize()
    runtime_seconds = time.monotonic() - started
    training_data_sha256 = _training_data_sha(train_records)
    manifest_path = output_dir / "model_manifest.json"
    manifest = {
        "schema_version": "1.0.0",
        "scientific_status": "campaign_experiment",
        "campaign_id": "cu-tail-risk-ralph-20260712",
        "research_spec_sha256": SPEC_SHA256,
        "arm": arm,
        "round_index": round_index,
        "seed": seed,
        "base_checkpoint_sha256": CHECKPOINT_SHA256,
        "model_sha256": sha256_file(model_path),
        "training_checkpoint_sha256": sha256_file(checkpoint_path),
        "training_data_sha256": training_data_sha256,
        "training_record_ids": sorted(record.config_id for record in train_records),
        "validation_record_ids": sorted(record.config_id for record in validation_records),
        "optimizer": "adamw",
        "learning_rate": learning_rate,
        "epochs": 20,
        "optimizer_steps": result.optimizer_steps,
        "batch_size": 4,
        "epoch_mode": "full_epoch",
        "trainable_layer_policy": "last_interaction_and_readout",
        "energy_loss_weight": 1.0,
        "force_loss_weight": 100.0,
        "e0_policy": "foundation",
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "best_validation_force_component_mae_ev_per_a": (
            result.best_validation_force_mae_ev_per_a
        ),
        "changed_parameter_tensors": result.changed_parameter_tensors,
        "changed_frozen_parameter_tensors": result.changed_frozen_parameter_tensors,
        "training_records": result.records,
        "hardware": _hardware(),
    }
    _write_json(manifest_path, manifest)
    return ModelIdentity(
        arm=arm,
        round_index=round_index,
        seed=seed,
        model_path=_relative(model_path, root),
        model_sha256=sha256_file(model_path),
        training_checkpoint_path=_relative(checkpoint_path, root),
        training_checkpoint_sha256=sha256_file(checkpoint_path),
        manifest_path=_relative(manifest_path, root),
        manifest_sha256=sha256_file(manifest_path),
        training_data_sha256=training_data_sha256,
        optimizer_steps=result.optimizer_steps,
        completed_epochs=result.completed_epochs,
        runtime_seconds=runtime_seconds,
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024**2,
    )


def _evaluate_models(
    *,
    root: Path,
    models: list[ModelIdentity],
    records: list[LabeledConfiguration],
    evaluation_name: str,
) -> str:
    output_dir = root / "evaluations" / evaluation_name
    output_dir.mkdir(parents=True, exist_ok=False)
    member_predictions: list[dict[str, dict[str, Any]]] = []
    seed_metrics: list[dict[str, Any]] = []
    member_artifacts: list[dict[str, Any]] = []
    for model in sorted(models, key=lambda item: item.seed):
        predictions = _predict(root / model.model_path, records)
        path = output_dir / f"predictions-seed-{model.seed}.json"
        _write_json(path, predictions)
        member_predictions.append(predictions)
        seed_metrics.append(
            {
                "seed": model.seed,
                "model_sha256": model.model_sha256,
                "metrics": evaluate_predictions(records, predictions),
            }
        )
        member_artifacts.append(
            {"seed": model.seed, "path": _relative(path, root), "sha256": sha256_file(path)}
        )
    ensemble = mean_predictions(member_predictions)
    ensemble_path = output_dir / "ensemble_predictions.json"
    _write_json(ensemble_path, ensemble)
    metrics_path = output_dir / "metrics.json"
    _write_json(
        metrics_path,
        {
            "schema_version": "1.0.0",
            "evaluation_name": evaluation_name,
            "evaluation_record_ids": sorted(record.config_id for record in records),
            "seed_metrics": seed_metrics,
            "ensemble_metrics": evaluate_predictions(records, ensemble),
            "member_prediction_artifacts": member_artifacts,
            "ensemble_prediction_artifact": {
                "path": _relative(ensemble_path, root),
                "sha256": sha256_file(ensemble_path),
            },
        },
    )
    return _relative(metrics_path, root)


def _predict_committee(
    root: Path,
    models: list[ModelIdentity],
    records: list[LabeledConfiguration],
    output_dir: Path,
) -> list[dict[str, dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions: list[dict[str, dict[str, Any]]] = []
    for model in sorted(models, key=lambda item: item.seed):
        member = _predict(root / model.model_path, records)
        path = output_dir / f"pool-predictions-seed-{model.seed}.json"
        _write_json(path, member)
        predictions.append(member)
    return predictions


def _raw_descriptors(
    checkpoint_path: Path, records: list[LabeledConfiguration]
) -> dict[str, list[float]]:
    calculator = _make_calculator(checkpoint_path, "cuda", "float32")
    num_interactions = int(calculator.models[0].num_interactions)
    descriptors: dict[str, list[float]] = {}
    for record in records:
        values = np.asarray(
            calculator.get_descriptors(
                _atoms(record), invariants_only=True, num_layers=num_interactions
            ),
            dtype=float,
        )
        descriptors[record.config_id] = values.mean(axis=0).tolist()
    return descriptors


def _tea_time(root: Path, step_id: str, trigger: TeaTimeTrigger) -> None:
    registry = ArtifactRegistry.load(root)
    context = SkillContext(
        run_dir=root,
        step_id=step_id,
        seed=20260712,
        attempt=0,
        registry=registry,
    )
    inputs = TeaTimeInput(
        focus_question=(
            "Does the next frozen acquisition or compute action still test force-tail "
            "selection without using protected outcomes?"
        ),
        trigger=trigger,
        stage_objective=(
            "Execute the preregistered Cu acquisition comparison while preserving equal "
            "budgets, oracle isolation, and finite-benchmark claim scope."
        ),
        benchmark_role=(
            "mlearn Cu is a finite two-trajectory tail benchmark, not evidence of universal "
            "acquisition performance."
        ),
        reusable_components=["oracle ledger", "independent metrics", "hash-bound models"],
        benchmark_specific_components=["mlearn Cu split", "MACE-MP-0-small checkpoint"],
        alternatives=[
            AlternativePath(
                path_id="execute_frozen",
                summary="Execute the already frozen intervention exactly as preregistered.",
                cost="bounded GPU allocation",
                risk="The result may be null or benchmark-specific.",
            ),
            AlternativePath(
                path_id="stop_before_reveal",
                summary="Stop before revealing labels if a viability or provenance gate fails.",
                cost="no additional label or GPU spend",
                risk="The main hypothesis remains unanswered.",
            ),
        ],
    )
    TeaTimeWithReadingPoemSkill().run(inputs, context)
    registry.save()


def _calibration(
    checkpoint_path: Path, records: list[LabeledConfiguration]
) -> dict[str, Any]:
    calculator = _make_calculator(checkpoint_path, "cuda", "float32")
    first = _predict(checkpoint_path, records, calculator=calculator)
    repeated = _predict(checkpoint_path, records, calculator=calculator)
    reloaded = _predict(checkpoint_path, records)
    first_metrics = evaluate_predictions(records, first)
    repeated_metrics = evaluate_predictions(records, repeated)
    return {
        "schema_version": "1.0.0",
        "records": sorted(first),
        "same_calculator_repeat": max_prediction_difference(first, repeated),
        "model_reload": max_prediction_difference(first, reloaded),
        "aggregate_metric_absolute_differences": {
            key: abs(float(first_metrics[key]) - float(repeated_metrics[key]))
            for key in (
                "energy_mae_ev_per_atom",
                "force_component_mae_ev_per_a",
                "force_component_rmse_ev_per_a",
                "force_vector_error_p95_ev_per_a",
            )
        },
        "reference_predictions": first,
        "hardware": _hardware(),
    }


def run_baseline(
    *,
    root: Path,
    dataset_path: Path,
    split_path: Path,
    checkpoint_path: Path,
    checkpoint_manifest_path: Path,
    spec_path: Path,
    authorization_path: Path,
    provider: str,
) -> CampaignState:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("baseline output root must be empty")
    root.mkdir(parents=True, exist_ok=True)
    dataset, split, _ = _load_inputs(
        dataset_path=dataset_path,
        split_path=split_path,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        spec_path=spec_path,
        authorization_path=authorization_path,
    )
    _tea_time(root, "tea-time-baseline-prelaunch", TeaTimeTrigger.AUTO_RESEARCH_REMOTE_PRELAUNCH)
    records = dataset.by_id()
    initial_ids = split.record_ids[PartitionName.INITIAL_LABELED.value]
    validation_ids = split.record_ids[PartitionName.VALIDATION.value]
    pool_ids = split.record_ids[PartitionName.ACQUISITION_POOL.value]
    initial = [records[record_id] for record_id in initial_ids]
    validation = [records[record_id] for record_id in validation_ids]
    pool = [records[record_id] for record_id in pool_ids]
    calibration_path = root / "calibration" / "colab_same_session.json"
    calibration = _calibration(checkpoint_path, initial[:3])
    _write_json(calibration_path, calibration)

    zero_predictions = _predict(checkpoint_path, validation)
    zero_dir = root / "evaluations" / "zero_shot" / "validation"
    zero_dir.mkdir(parents=True)
    zero_prediction_path = zero_dir / "predictions.json"
    _write_json(zero_prediction_path, zero_predictions)
    zero_metrics_path = zero_dir / "metrics.json"
    _write_json(
        zero_metrics_path,
        {
            "schema_version": "1.0.0",
            "model_sha256": CHECKPOINT_SHA256,
            "metrics": evaluate_predictions(validation, zero_predictions),
            "predictions_path": _relative(zero_prediction_path, root),
            "predictions_sha256": sha256_file(zero_prediction_path),
        },
    )

    learning_rate = 0.0005
    d0_models = [
        _fit_model(
            root=root,
            foundation_path=checkpoint_path,
            train_records=initial,
            validation_records=validation,
            arm="static_initial",
            round_index=0,
            seed=seed,
            learning_rate=learning_rate,
        )
        for seed in SEEDS
    ]
    static_metrics_path = _evaluate_models(
        root=root,
        models=d0_models,
        records=validation,
        evaluation_name="static_initial/round-0/validation",
    )
    descriptor_path = root / "acquisition" / "foundation_raw_descriptors.json"
    descriptors = _raw_descriptors(checkpoint_path, pool)
    _write_json(descriptor_path, descriptors)
    d0_pool_predictions = _predict_committee(
        root,
        d0_models,
        pool,
        root / "acquisition" / "d0_pool_predictions",
    )
    d0_scores = committee_scores(d0_pool_predictions)
    score_path = root / "acquisition" / "d0_scores.json"
    _write_json(score_path, d0_scores)
    numerical_floor = max(
        calibration["same_calculator_repeat"][
            "maximum_absolute_force_component_difference_ev_per_a"
        ],
        calibration["model_reload"][
            "maximum_absolute_force_component_difference_ev_per_a"
        ],
    )
    viability = viability_summary(d0_scores, numerical_floor)
    viability_path = root / "acquisition" / "d0_committee_viability.json"
    _write_json(viability_path, viability)
    if not viability["passed"]:
        raise RuntimeError(
            "D0 committee failed the preregistered outcome-blind viability gate; "
            "no pool labels were revealed"
        )

    experiment_ledger = root / "experiment_ledger.jsonl"
    oracle_ledger = root / "oracle_reveal_ledger.jsonl"
    _append_jsonl(
        experiment_ledger,
        {
            "event": "baseline_frozen",
            "observed_at": _utc_now(),
            "provider": provider,
            "research_spec_sha256": SPEC_SHA256,
            "model_hashes": [model.model_sha256 for model in d0_models],
            "zero_shot_metrics_sha256": sha256_file(zero_metrics_path),
            "static_metrics_sha256": sha256_file(root / static_metrics_path),
            "viability_sha256": sha256_file(viability_path),
        },
    )
    _append_jsonl(
        oracle_ledger,
        {
            "event": "initial_labels_frozen",
            "observed_at": _utc_now(),
            "record_ids": sorted(initial_ids),
            "training_head_sha256": _canonical_sha(sorted(initial_ids)),
            "charged_pool_labels": 0,
        },
    )
    arm_states = {
        arm: ArmState(
            arm=arm,
            cumulative_training_record_ids=sorted(initial_ids),
            current_models=d0_models,
            validation_history_paths=[static_metrics_path],
        )
        for arm in ARMS
    }
    state = CampaignState(
        research_spec_sha256=SPEC_SHA256,
        dataset_content_sha256=DATASET_CONTENT_SHA256,
        dataset_file_sha256=DATASET_FILE_SHA256,
        split_semantic_sha256=SPLIT_SEMANTIC_SHA256,
        base_checkpoint_sha256=CHECKPOINT_SHA256,
        provider=provider,
        completed_round=0,
        frozen_initial_record_ids=sorted(initial_ids),
        d0_models=d0_models,
        arms=arm_states,
        zero_shot_validation_path=_relative(zero_metrics_path, root),
        static_validation_path=static_metrics_path,
        raw_descriptor_path=_relative(descriptor_path, root),
        raw_descriptor_sha256=sha256_file(descriptor_path),
        numerical_calibration_path=_relative(calibration_path, root),
        oracle_ledger_path=_relative(oracle_ledger, root),
        experiment_ledger_path=_relative(experiment_ledger, root),
        learning_rate_used=learning_rate,
    )
    state_path = root / "state.json"
    _write_json(state_path, state.model_dump(mode="json"))
    return state


def run_round(
    *,
    root: Path,
    dataset_path: Path,
    split_path: Path,
    checkpoint_path: Path,
    checkpoint_manifest_path: Path,
    spec_path: Path,
    authorization_path: Path,
    provider: str,
    round_index: int,
) -> CampaignState:
    dataset, split, _ = _load_inputs(
        dataset_path=dataset_path,
        split_path=split_path,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        spec_path=spec_path,
        authorization_path=authorization_path,
    )
    state_path = root / "state.json"
    state = CampaignState.model_validate_json(state_path.read_text())
    if round_index != state.completed_round + 1 or round_index not in {1, 2, 3}:
        raise ValueError("round request is not the next frozen sequential round")
    if state.research_spec_sha256 != SPEC_SHA256:
        raise ValueError("campaign state research-spec mismatch")
    for model in [state.d0_models, *[arm.current_models for arm in state.arms.values()]]:
        for identity in model:
            if sha256_file(root / identity.model_path) != identity.model_sha256:
                raise ValueError("campaign state model hash mismatch")
    _tea_time(
        root,
        f"tea-time-round-{round_index}-pre-reveal",
        TeaTimeTrigger.AUTO_RESEARCH_ITERATION_BOUNDARY,
    )
    records = dataset.by_id()
    validation = [
        records[record_id]
        for record_id in split.record_ids[PartitionName.VALIDATION.value]
    ]
    pool_ids = set(split.record_ids[PartitionName.ACQUISITION_POOL.value])
    raw_descriptor_path = root / state.raw_descriptor_path
    if sha256_file(raw_descriptor_path) != state.raw_descriptor_sha256:
        raise ValueError("raw descriptor artifact hash mismatch")
    raw_descriptors: dict[str, list[float]] = json.loads(raw_descriptor_path.read_text())

    new_arms: dict[ArmName, ArmState] = {}
    for arm_name in ARMS:
        arm = state.arms[arm_name]
        already_selected = set(arm.cumulative_training_record_ids) - set(
            state.frozen_initial_record_ids
        )
        remaining_ids = sorted(pool_ids - already_selected)
        remaining_records = [records[record_id] for record_id in remaining_ids]
        prediction_dir = (
            root / "rounds" / f"round-{round_index}" / arm_name / "pool_predictions"
        )
        member_predictions = _predict_committee(
            root, arm.current_models, remaining_records, prediction_dir
        )
        scores = committee_scores(member_predictions)
        selected, selection_details = select_candidates(
            policy=arm_name,
            remaining_ids=remaining_ids,
            scores=scores,
            raw_descriptors=raw_descriptors,
            round_index=round_index,
        )
        selection_path = (
            root / "rounds" / f"round-{round_index}" / arm_name / "selection.json"
        )
        selection_payload = {
            "schema_version": "1.0.0",
            "campaign_id": state.campaign_id,
            "research_spec_sha256": SPEC_SHA256,
            "round_index": round_index,
            "policy": arm_name,
            "label_access": "none during scoring and selection",
            "remaining_pool_record_ids": remaining_ids,
            "committee_model_sha256": [model.model_sha256 for model in arm.current_models],
            "scores": scores,
            "selected_record_ids": selected,
            "selection_details": selection_details,
        }
        _write_json(selection_path, selection_payload)
        prior_head = sorted(arm.cumulative_training_record_ids)
        new_head = sorted([*prior_head, *selected])
        if len(new_head) != 32 + round_index * 12:
            raise ValueError("oracle reveal would violate the frozen per-arm label budget")
        reveal_event = {
            "event": "oracle_reveal",
            "observed_at": _utc_now(),
            "campaign_id": state.campaign_id,
            "arm": arm_name,
            "round_index": round_index,
            "selection_artifact_sha256": sha256_file(selection_path),
            "selected_record_ids": selected,
            "selected_record_content_sha256": {
                record_id: records[record_id].content_hash() for record_id in selected
            },
            "prior_training_head_sha256": _canonical_sha(prior_head),
            "new_training_head_sha256": _canonical_sha(new_head),
            "round_charge": 12,
            "cumulative_pool_charge": round_index * 12,
        }
        _append_jsonl(root / state.oracle_ledger_path, reveal_event)
        train_records = [records[record_id] for record_id in new_head]
        models = [
            _fit_model(
                root=root,
                foundation_path=checkpoint_path,
                train_records=train_records,
                validation_records=validation,
                arm=arm_name,
                round_index=round_index,
                seed=seed,
                learning_rate=state.learning_rate_used,
            )
            for seed in SEEDS
        ]
        validation_path = _evaluate_models(
            root=root,
            models=models,
            records=validation,
            evaluation_name=f"{arm_name}/round-{round_index}/validation",
        )
        selection = SelectionRound(
            round_index=round_index,
            policy=arm_name,
            selected_record_ids=selected,
            selection_artifact_path=_relative(selection_path, root),
            selection_artifact_sha256=sha256_file(selection_path),
            prior_training_head_sha256=_canonical_sha(prior_head),
            new_training_head_sha256=_canonical_sha(new_head),
        )
        new_arms[arm_name] = ArmState(
            arm=arm_name,
            cumulative_training_record_ids=new_head,
            selections=[*arm.selections, selection],
            current_models=models,
            validation_history_paths=[*arm.validation_history_paths, validation_path],
        )
        _append_jsonl(
            root / state.experiment_ledger_path,
            {
                "event": "arm_round_completed",
                "observed_at": _utc_now(),
                "provider": provider,
                "round_index": round_index,
                "arm": arm_name,
                "selection_sha256": sha256_file(selection_path),
                "model_hashes": [model.model_sha256 for model in models],
                "validation_metrics_sha256": sha256_file(root / validation_path),
            },
        )
    next_state = state.model_copy(
        update={
            "provider": provider,
            "completed_round": round_index,
            "arms": new_arms,
        }
    )
    next_state = CampaignState.model_validate(next_state.model_dump(mode="json"))
    _write_json(state_path, next_state.model_dump(mode="json"))
    _append_jsonl(
        root / state.experiment_ledger_path,
        {
            "event": "ralph_round_frozen",
            "observed_at": _utc_now(),
            "round_index": round_index,
            "state_sha256": sha256_file(state_path),
            "decision": "await_host_scientific_review_before_next_round",
        },
    )
    return next_state
