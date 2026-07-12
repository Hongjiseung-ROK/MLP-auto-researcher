"""Transactional, lineage-bound MACE fine-tuning boundary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mlip_research_agent.schemas.failure import (
    FailureClass,
    RecoveryDecision,
    Severity,
)
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.mlip.mace_finetune.runner import (
    MONITOR,
    SUPPORTED_MACE_VERSION,
    run_controlled_training,
)
from mlip_research_agent.skills.mlip.mace_finetune.schema import (
    FineTuneResumeState,
    MACEFineTuneInput,
    MACEFineTuneOutput,
)
from mlip_research_agent.skills.mlip.mace_finetune.validators import (
    FineTuneDataBundle,
    load_and_validate_data,
    registered_artifact,
    validation_error,
)
from mlip_research_agent.skills.mlip.mace_inference.checkpoint import (
    MACECheckpointManifest,
    resolve_checkpoint,
)
from mlip_research_agent.skills.mlip.mace_inference.validators import local_file


def _resume_contract(
    params: MACEFineTuneInput,
    bundle: FineTuneDataBundle,
    base_checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": "1.0.0",
        "execution_mode": params.execution_mode,
        "run_name": params.run_name,
        "seed": params.seed,
        "optimizer": params.optimizer,
        "learning_rate": params.learning_rate,
        "gradient_clip": params.gradient_clip,
        "batch_size": params.batch_size,
        "valid_batch_size": params.valid_batch_size,
        "early_stopping_patience": params.early_stopping_patience,
        "monitor": MONITOR,
        "e0_policy": params.e0_policy,
        "energy_loss_weight": params.energy_loss_weight,
        "force_loss_weight": params.force_loss_weight,
        "trainable_layer_policy": params.trainable_layer_policy,
        "device": params.device,
        "default_dtype": params.default_dtype,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "dataset_content_sha256": bundle.dataset.content_hash(),
        "split_semantic_sha256": bundle.split.semantic_hash(),
        "training_lineage_artifact": params.training_lineage_artifact,
        "train_record_ids": bundle.train_subset.record_ids,
        "validation_record_ids": bundle.validation_subset.record_ids,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _failure_properties(
    exc: Exception, params: MACEFineTuneInput
) -> tuple[FailureClass, bool, dict[str, Any], RecoveryDecision]:
    lowered = str(exc).lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        can_reduce = params.batch_size > 1
        repairs = {"batch_size": max(1, params.batch_size // 2)} if can_reduce else {}
        return (
            FailureClass.RESOURCE_EXHAUSTED,
            can_reduce,
            repairs,
            RecoveryDecision.REFINE if can_reduce else RecoveryDecision.ESCALATE,
        )
    if "non-finite" in lowered or "nan" in lowered:
        return (
            FailureClass.SIMULATION_INSTABILITY,
            True,
            {
                "learning_rate": params.learning_rate * 0.5,
                "gradient_clip": min(params.gradient_clip, 5.0),
            },
            RecoveryDecision.REFINE,
        )
    return FailureClass.TOOL_ERROR, False, {}, RecoveryDecision.ESCALATE


def _record_failure(
    *,
    stage_dir: Path,
    ctx: SkillContext,
    params: MACEFineTuneInput,
    exc: Exception,
) -> None:
    failure_class, retryable, repair_params, action = _failure_properties(exc, params)
    failed_dir = ctx.step_dir / f"failed-attempt-{ctx.attempt:03d}"
    if failed_dir.exists():
        raise validation_error(f"fine-tune failure directory already exists: {failed_dir}")
    os.replace(stage_dir, failed_dir)
    failure_path = failed_dir / f"fine_tune_failure-attempt-{ctx.attempt:03d}.json"
    failure_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "scientific_status": "training_boundary_failure",
                "failure_class": failure_class.value,
                "retryable": retryable,
                "recommended_action": action.value,
                "repair_params": repair_params,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:4000],
                "successful_model_emitted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    evidence = ctx.registry.register(failure_path, "mace_fine_tune_failure", ctx.step_id)
    raise SkillError(
        f"MACE fine-tuning failed; evidence: {evidence.artifact_id}",
        failure_class=failure_class,
        severity=Severity.HIGH,
        retryable=retryable,
        recommended_action=action,
        repair_params=repair_params,
        likely_causes=["see the registered fine-tune failure artifact"],
    )


@register_skill
class MACEFineTuneSkill(Skill):
    """Run a bounded optimizer-step diagnostic without crossing the H2 gate."""

    name = "mace_finetune"
    input_model = MACEFineTuneInput
    output_model = MACEFineTuneOutput
    cost_class = "expensive"
    permission_level = "human_approval"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, MACEFineTuneInput)
        if params.execution_mode == "pilot":
            raise validation_error(
                "pilot fine-tuning is disabled until H2 selects a checkpoint and freezes "
                "the preregistration"
            )
        bundle = load_and_validate_data(params, ctx.registry)
        if params.batch_size < len(bundle.train_records):
            raise validation_error(
                "boundary test requires batch_size >= the training subset size so resume "
                "occurs only at complete epoch boundaries"
            )
        manifest_path = local_file(params.checkpoint_manifest_path, "checkpoint manifest")
        foundation_path = local_file(params.checkpoint_path, "foundation checkpoint")
        try:
            checkpoint_manifest = MACECheckpointManifest.load(manifest_path)
            resolved = resolve_checkpoint(checkpoint_manifest, foundation_path)
        except (OSError, ValueError) as exc:
            raise validation_error(f"fine-tune checkpoint validation failed: {exc}") from exc
        if checkpoint_manifest.mace_torch_version != SUPPORTED_MACE_VERSION:
            raise validation_error("foundation checkpoint manifest has unsupported MACE version")
        if checkpoint_manifest.scientific_status != "candidate_only":
            raise validation_error(
                "boundary test expects a candidate-only checkpoint before H2 selection"
            )

        contract_sha = _resume_contract(params, bundle, resolved.sha256)
        resume_state: FineTuneResumeState | None = None
        resume_checkpoint_path: Path | None = None
        if params.resume_state_artifact is not None:
            _, state_path = registered_artifact(
                ctx.registry, params.resume_state_artifact, "mace_training_state"
            )
            try:
                resume_state = FineTuneResumeState.model_validate_json(state_path.read_text())
            except (OSError, ValueError) as exc:
                raise validation_error(f"fine-tune resume state is invalid: {exc}") from exc
            if resume_state.resume_contract_sha256 != contract_sha:
                raise validation_error("fine-tune resume contract hash mismatch")
            if params.max_epochs <= resume_state.completed_epochs:
                raise validation_error("resumed max_epochs must exceed completed epochs")
            if params.max_optimizer_steps <= resume_state.optimizer_steps:
                raise validation_error(
                    "resumed max_optimizer_steps must exceed completed optimizer steps"
                )
            _, resume_checkpoint_path = registered_artifact(
                ctx.registry,
                resume_state.checkpoint_artifact,
                "mace_training_checkpoint",
            )

        transaction_parent = ctx.step_dir / ".transactions"
        transaction_parent.mkdir(parents=True, exist_ok=True)
        stage_dir = Path(tempfile.mkdtemp(prefix="mace-fine-tune-", dir=transaction_parent))
        suffix = f"attempt-{ctx.attempt:03d}"
        try:
            checkpoint_path = stage_dir / f"controlled-checkpoint-{suffix}.pt"
            model_path = stage_dir / f"fine-tuned-model-{suffix}.model"
            result = run_controlled_training(
                foundation_path=foundation_path,
                checkpoint_path=checkpoint_path,
                model_path=model_path,
                train_records=bundle.train_records,
                validation_records=bundle.validation_records,
                resume_checkpoint_path=resume_checkpoint_path,
                resume_contract_sha256=contract_sha,
                seed=params.seed,
                optimizer_name=params.optimizer,
                learning_rate=params.learning_rate,
                gradient_clip=params.gradient_clip,
                batch_size=params.batch_size,
                valid_batch_size=params.valid_batch_size,
                max_epochs=params.max_epochs,
                max_optimizer_steps=params.max_optimizer_steps,
                patience=params.early_stopping_patience,
                device_name=params.device,
                default_dtype=params.default_dtype,
                max_wall_seconds=params.max_wall_seconds,
                energy_loss_weight=params.energy_loss_weight,
                force_loss_weight=params.force_loss_weight,
                trainable_layer_policy=params.trainable_layer_policy,
            )
            config_path = stage_dir / f"fine-tune-config-{suffix}.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "scientific_status": "training_boundary_only",
                        "execution_mode": params.execution_mode,
                        "mace_torch_version": SUPPORTED_MACE_VERSION,
                        "resume_contract_sha256": contract_sha,
                        "monitor": MONITOR,
                        "input": params.model_dump(mode="json"),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            metrics_path = stage_dir / f"training-metrics-{suffix}.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "scientific_status": "training_boundary_only",
                        "monitor": MONITOR,
                        "best_validation_force_mae_ev_per_a": (
                            result.best_validation_force_mae_ev_per_a
                        ),
                        "completed_epochs": result.completed_epochs,
                        "optimizer_steps": result.optimizer_steps,
                        "patience_count": result.patience_count,
                        "stopped_early": result.stopped_early,
                        "records": result.records,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            final_dir = ctx.step_dir / suffix
            if final_dir.exists():
                raise ValueError(f"fine-tune final directory already exists: {final_dir}")
            os.replace(stage_dir, final_dir)
            model_path = final_dir / model_path.name
            checkpoint_path = final_dir / checkpoint_path.name
            config_path = final_dir / config_path.name
            metrics_path = final_dir / metrics_path.name
            model_artifact = ctx.registry.register(model_path, "fine_tuned_mace_model", ctx.step_id)
            checkpoint_artifact = ctx.registry.register(
                checkpoint_path, "mace_training_checkpoint", ctx.step_id
            )
            config_artifact = ctx.registry.register(
                config_path, "mace_fine_tune_config", ctx.step_id
            )
            metrics_artifact = ctx.registry.register(
                metrics_path, "mace_training_metrics", ctx.step_id
            )

            model_manifest_path = final_dir / f"fine-tuned-model-manifest-{suffix}.json"
            model_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "scientific_status": "training_boundary_only",
                        "model_id": f"{checkpoint_manifest.model_id}-boundary-{params.seed}",
                        "base_checkpoint": resolved.model_dump(mode="json"),
                        "model_artifact": model_artifact.artifact_id,
                        "model_sha256": model_artifact.sha256,
                        "checkpoint_artifact": checkpoint_artifact.artifact_id,
                        "checkpoint_sha256": checkpoint_artifact.sha256,
                        "dataset_content_sha256": bundle.dataset.content_hash(),
                        "split_semantic_sha256": bundle.split.semantic_hash(),
                        "training_lineage_artifact": params.training_lineage_artifact,
                        "train_subset_manifest_artifact": (params.train_subset_manifest_artifact),
                        "validation_subset_manifest_artifact": (
                            params.validation_subset_manifest_artifact
                        ),
                        "config_artifact": config_artifact.artifact_id,
                        "metrics_artifact": metrics_artifact.artifact_id,
                        "e0_policy": params.e0_policy,
                        "monitor": MONITOR,
                        "completed_epochs": result.completed_epochs,
                        "optimizer_steps": result.optimizer_steps,
                        "changed_parameter_tensors": result.changed_parameter_tensors,
                        "max_abs_parameter_change": result.max_abs_parameter_change,
                        "trainable_parameter_names": result.trainable_parameter_names,
                        "frozen_parameter_names": result.frozen_parameter_names,
                        "changed_frozen_parameter_tensors": (
                            result.changed_frozen_parameter_tensors
                        ),
                        "resumed": resume_state is not None,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            model_manifest_artifact = ctx.registry.register(
                model_manifest_path, "fine_tuned_model_manifest", ctx.step_id
            )
            state_path = final_dir / f"fine-tune-state-{suffix}.json"
            state = FineTuneResumeState(
                run_name=params.run_name,
                seed=params.seed,
                completed_epochs=result.completed_epochs,
                optimizer_steps=result.optimizer_steps,
                best_validation_force_mae_ev_per_a=(result.best_validation_force_mae_ev_per_a),
                patience_count=result.patience_count,
                checkpoint_artifact=checkpoint_artifact.artifact_id,
                model_artifact=model_artifact.artifact_id,
                resume_contract_sha256=contract_sha,
                base_checkpoint_sha256=resolved.sha256,
                dataset_content_sha256=bundle.dataset.content_hash(),
                split_semantic_sha256=bundle.split.semantic_hash(),
            )
            state_path.write_text(
                json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
            )
            state_artifact = ctx.registry.register(state_path, "mace_training_state", ctx.step_id)
            return MACEFineTuneOutput(
                model_artifact=model_artifact.artifact_id,
                model_path=model_artifact.relative_path,
                model_manifest_artifact=model_manifest_artifact.artifact_id,
                model_manifest_path=model_manifest_artifact.relative_path,
                checkpoint_artifact=checkpoint_artifact.artifact_id,
                checkpoint_path=checkpoint_artifact.relative_path,
                training_state_artifact=state_artifact.artifact_id,
                training_state_path=state_artifact.relative_path,
                training_metrics_artifact=metrics_artifact.artifact_id,
                training_metrics_path=metrics_artifact.relative_path,
                config_artifact=config_artifact.artifact_id,
                config_path=config_artifact.relative_path,
                n_train=len(bundle.train_records),
                n_validation=len(bundle.validation_records),
                completed_epochs=result.completed_epochs,
                optimizer_steps=result.optimizer_steps,
                changed_parameter_tensors=result.changed_parameter_tensors,
                resumed=resume_state is not None,
                trainable_parameter_names=list(result.trainable_parameter_names),
                frozen_parameter_names=list(result.frozen_parameter_names),
                changed_frozen_parameter_tensors=result.changed_frozen_parameter_tensors,
            )
        except SkillError:
            raise
        except Exception as exc:
            if stage_dir.exists():
                _record_failure(
                    stage_dir=stage_dir,
                    ctx=ctx,
                    params=params,
                    exc=exc,
                )
            raise
        finally:
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
