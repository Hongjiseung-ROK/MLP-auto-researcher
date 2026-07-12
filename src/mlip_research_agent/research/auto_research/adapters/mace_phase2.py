"""Infrastructure-only real-MACE adapter over the bounded Phase 2 label view."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ase import Atoms

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.data.bounded_view import (
    EXPECTED_DATASET_CONTENT_SHA256,
    EXPECTED_SPLIT_MANIFEST_SHA256,
    BoundedLabelView,
)
from mlip_research_agent.research.auto_research.adapters.base import AdapterRunResult
from mlip_research_agent.research.auto_research.mutation import MutationPolicy
from mlip_research_agent.research.auto_research.objective import ResearchObjective
from mlip_research_agent.research.auto_research.operations import (
    ExactlyOnceOperation,
    OptimizerOperationRequest,
)
from mlip_research_agent.research.auto_research.proposal import (
    EstimatedCompute,
    ExpectedObservable,
    ExperimentProposal,
)
from mlip_research_agent.research.auto_research.proposal_policy import generate_proposal
from mlip_research_agent.research.auto_research.validators import (
    ConfigValue,
    canonical_json,
    sha256_of_text,
)
from mlip_research_agent.skills.mlip.mace_finetune.runner import run_controlled_training
from mlip_research_agent.skills.mlip.mace_inference.checkpoint import (
    MACECheckpointManifest,
    resolve_checkpoint,
)
from mlip_research_agent.skills.mlip.mace_inference.implementation import _make_calculator

EXPECTED_CHECKPOINT_SHA256 = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"


def build_initial_mace_proposal(
    *,
    objective: ResearchObjective,
    policy: MutationPolicy,
    config: dict[str, ConfigValue],
    seed: int,
) -> ExperimentProposal:
    """Build the sealed one-step MACE proposal without synthetic-fixture prose."""
    generic = generate_proposal(
        objective=objective,
        policy=policy,
        current_config=config,
        iteration_index=0,
        prior=None,
        seed=seed,
    )
    mutation = generic.proposed_mutations[0].model_copy(
        update={
            "scientific_effect": (
                "A smaller update may improve the independently evaluated aggregate "
                "validation force metric for this bounded infrastructure replay."
            ),
            "engineering_effect": (
                "Use smaller optimizer steps without changing the checkpoint, data, or "
                "trainable-layer policy."
            ),
        }
    ).sealed()
    proposal = generic.model_copy(
        update={
            "hypothesis": (
                "A smaller learning rate may reduce the aggregate validation force error "
                "after one bounded MACE optimizer step from the same foundation checkpoint."
            ),
            "expected_mechanism": (
                "Reducing the update magnitude may improve one-step numerical stability "
                "without changing the data, checkpoint, seed, or trainable-layer policy."
            ),
            "expected_observables": [
                ExpectedObservable(
                    metric=objective.success_criteria[0].metric,
                    expected_direction="decrease",
                    rationale=(
                        "The independent evaluator will compare only aggregate validation "
                        "metrics under the fixed infrastructure contract."
                    ),
                )
            ],
            "falsification_condition": (
                "If the independently computed aggregate validation force error does not "
                "improve by the precommitted threshold, the one-step smaller-update "
                "hypothesis is rejected for this infrastructure trace."
            ),
            "estimated_compute": EstimatedCompute(
                device="gpu",
                estimated_seconds=900.0,
                remote=True,
            ),
            "required_skills": ["tea_time_with_reading_poem", "mace_finetune", "mlip_metrics"],
            "risk_class": "medium",
            "required_approval": None,
            "proposal_policy_version": "mace-infrastructure-replay/1.0.0",
            "proposed_mutations": [mutation],
        }
    )
    return proposal.sealed()


def _required_number(config: dict[str, ConfigValue], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"real MACE config {key!r} must be numeric")
    return float(value)


def _tensor_subset_hash(
    model: Any, marker: str | None = None, *, decimal_places: int | None = None
) -> str:
    digest = hashlib.sha256()
    found = 0
    for name, tensor in sorted(model.state_dict().items()):
        if marker is not None and marker not in name.casefold():
            continue
        found += 1
        digest.update(name.encode())
        values = tensor.detach().cpu().contiguous().numpy()
        if decimal_places is not None:
            values = values.astype("float64").round(decimal_places)
        digest.update(values.tobytes())
    if found == 0:
        raise ValueError(f"model state contains no tensors matching {marker!r}")
    return digest.hexdigest()


def _tensor_subset_values(model: Any, marker: str) -> list[float]:
    values: list[float] = []
    for name, tensor in sorted(model.state_dict().items()):
        if marker in name.casefold():
            values.extend(float(value) for value in tensor.detach().cpu().reshape(-1))
    if not values:
        raise ValueError(f"model state contains no tensors matching {marker!r}")
    return values


class MACEPhase2Adapter:
    """One-step same-foundation comparison; never decides acceptance."""

    def __init__(
        self,
        *,
        label_view_path: Path,
        checkpoint_manifest_path: Path,
        checkpoint_path: Path,
        label_view_sha256: str,
        git_commit: str,
        compute_attestation: str,
    ) -> None:
        if len(git_commit) != 40:
            raise ValueError("MACE replay requires a full 40-character commit")
        self.label_view_path = label_view_path
        self.checkpoint_manifest_path = checkpoint_manifest_path
        self.checkpoint_path = checkpoint_path
        self.label_view_sha256 = label_view_sha256
        self.git_commit = git_commit
        self.compute_attestation = compute_attestation

    @property
    def name(self) -> str:
        return "mace_phase2_infrastructure_replay/1.0.0"

    @property
    def remote(self) -> bool:
        return True

    def _verify_inputs(self) -> tuple[BoundedLabelView, MACECheckpointManifest]:
        if sha256_file(self.label_view_path) != self.label_view_sha256:
            raise ValueError("bounded label-view SHA-256 mismatch")
        view = BoundedLabelView.load(self.label_view_path)
        if view.source_dataset_content_sha256 != EXPECTED_DATASET_CONTENT_SHA256:
            raise ValueError("bounded view dataset identity mismatch")
        if view.split_manifest_sha256 != EXPECTED_SPLIT_MANIFEST_SHA256:
            raise ValueError("bounded view split identity mismatch")
        manifest = MACECheckpointManifest.load(self.checkpoint_manifest_path)
        resolved = resolve_checkpoint(manifest, self.checkpoint_path)
        if resolved.sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("infrastructure candidate checkpoint SHA-256 mismatch")
        if manifest.scientific_status != "candidate_only":
            raise ValueError("checkpoint must remain candidate_only")
        return view, manifest

    def execute(self, config: dict[str, ConfigValue], seed: int, workdir: Path) -> AdapterRunResult:
        started = time.monotonic()
        workdir.mkdir(parents=True, exist_ok=True)
        view, manifest = self._verify_inputs()
        definition_path = workdir / "fixture_definition.json"
        definition = {
            "adapter": self.name,
            "scientific_status": "infrastructure_only",
            "claim_eligible": False,
            "uses_frozen_test_data": False,
            "uses_hidden_labels": False,
            "uses_protected_partitions": False,
            "uses_acquisition_pool_labels": False,
            "dataset_content_sha256": view.source_dataset_content_sha256,
            "bounded_label_view_sha256": self.label_view_sha256,
            "normalized_manifest_sha256": view.normalized_manifest_sha256,
            "split_manifest_sha256": view.split_manifest_sha256,
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "n_initial_labeled": len(view.initial_labeled),
            "n_validation": len(view.validation),
            "e0_policy": "foundation",
            "config": dict(sorted(config.items())),
        }
        definition_path.write_text(json.dumps(definition, indent=2, sort_keys=True) + "\n")
        model_path = self.checkpoint_path
        operation_receipt_path: Path | None = None
        if workdir.name != "baseline":
            config_hash = sha256_of_text(canonical_json(config))
            proposal = ExperimentProposal.model_validate_json(
                (workdir.parent / "proposal.json").read_text()
            )
            if not proposal.verify_seal():
                raise ValueError("optimizer request proposal seal mismatch")
            request = OptimizerOperationRequest(
                operation_id=f"{workdir.parent.name}-{workdir.parent.parent.name}"[:64].lower(),
                git_commit=self.git_commit,
                proposal_fingerprint=proposal.content_sha256,
                config_fingerprint=config_hash,
                seed=seed,
                dataset_content_sha256=view.source_dataset_content_sha256,
                split_manifest_sha256=view.split_manifest_sha256,
                checkpoint_sha256=manifest.checkpoint_sha256,
            ).sealed()
            operation = ExactlyOnceOperation(workdir / "operation", request)
            reused = operation.begin()
            if reused is not None:
                raise RuntimeError(
                    "completed operation receipt exists but model reuse is unavailable"
                )
            import torch

            foundation = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            e0_before_values = _tensor_subset_values(foundation, "atomic_energies")
            e0_before = _tensor_subset_hash(
                foundation, "atomic_energies", decimal_places=6
            )
            model_path = workdir / "fine_tuned.model"
            checkpoint_path = workdir / "optimizer_checkpoint.pt"
            result = run_controlled_training(
                foundation_path=self.checkpoint_path,
                checkpoint_path=checkpoint_path,
                model_path=model_path,
                train_records=[r.to_training_record() for r in view.initial_labeled],
                validation_records=[r.to_training_record() for r in view.validation],
                resume_checkpoint_path=None,
                resume_contract_sha256=request.content_sha256,
                seed=seed,
                optimizer_name="adam",
                learning_rate=_required_number(config, "learning_rate"),
                gradient_clip=_required_number(config, "gradient_clip"),
                batch_size=int(_required_number(config, "batch_size")),
                valid_batch_size=len(view.validation),
                max_epochs=1,
                max_optimizer_steps=1,
                patience=int(_required_number(config, "scheduler_patience")),
                device_name="cuda",
                default_dtype="float32",
                max_wall_seconds=900,
                energy_loss_weight=_required_number(config, "energy_loss_weight"),
                force_loss_weight=_required_number(config, "force_loss_weight"),
                trainable_layer_policy=str(config.get("trainable_layer_policy")),
            )
            if result.optimizer_steps != 1:
                raise RuntimeError("infrastructure replay must perform exactly one optimizer step")
            trained = torch.load(model_path, map_location="cpu", weights_only=False)
            e0_after_values = _tensor_subset_values(trained, "atomic_energies")
            e0_after = _tensor_subset_hash(
                trained, "atomic_energies", decimal_places=6
            )
            reloaded = torch.load(model_path, map_location="cpu", weights_only=False)
            e0_reload_values = _tensor_subset_values(reloaded, "atomic_energies")
            e0_reload = _tensor_subset_hash(
                reloaded, "atomic_energies", decimal_places=6
            )
            if not (
                len(e0_before_values)
                == len(e0_after_values)
                == len(e0_reload_values)
            ):
                raise RuntimeError("foundation E0 tensor shape changed")
            e0_max_delta = max(
                abs(before - after)
                for before, after in zip(
                    e0_before_values, e0_after_values, strict=True
                )
            )
            e0_reload_delta = max(
                abs(after - reload)
                for after, reload in zip(
                    e0_after_values, e0_reload_values, strict=True
                )
            )
            if e0_max_delta > 1.0e-6 or e0_reload_delta != 0.0:
                raise RuntimeError("foundation E0 tensor state changed during training/save/reload")
            state_hash = _tensor_subset_hash(trained)
            reload_hash = _tensor_subset_hash(reloaded)
            if state_hash != reload_hash:
                raise RuntimeError("saved MACE model does not round-trip exactly")
            (workdir / "training_evidence.json").write_text(
                json.dumps(
                    {
                        "scientific_status": "infrastructure_only",
                        "claim_eligible": False,
                        "optimizer_steps": result.optimizer_steps,
                        "changed_parameter_tensors": result.changed_parameter_tensors,
                        "changed_frozen_parameter_tensors": result.changed_frozen_parameter_tensors,
                        "trainable_parameter_names": result.trainable_parameter_names,
                        "frozen_parameter_names": result.frozen_parameter_names,
                        "e0_before_sha256": e0_before,
                        "e0_after_sha256": e0_after,
                        "e0_reload_sha256": e0_reload,
                        "e0_max_abs_delta_ev": e0_max_delta,
                        "e0_reload_max_abs_delta_ev": e0_reload_delta,
                        "e0_comparison_tolerance_ev": 1.0e-6,
                        "model_state_sha256": state_hash,
                        "reloaded_model_state_sha256": reload_hash,
                        "reload_optimizer_steps": 0,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            operation.complete(
                model_sha256=sha256_file(model_path),
                checkpoint_sha256=sha256_file(checkpoint_path),
            )
            operation_receipt_path = operation.completed_path
        predictions_path = workdir / "predictions.json"
        rerun_path = workdir / "predictions_rerun.json"
        self._predict(view, manifest, model_path, predictions_path, workdir)
        self._predict(view, manifest, model_path, rerun_path, workdir)
        peak_mb = 0.0
        try:
            import torch

            peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        except (ImportError, RuntimeError):
            pass
        return AdapterRunResult(
            succeeded=True,
            fixture_definition_path=str(definition_path),
            predictions_path=str(predictions_path),
            predictions_rerun_path=str(rerun_path),
            failure_path=None,
            failure_category=None,
            repair_available=False,
            simulated_wall_seconds=time.monotonic() - started,
            simulated_peak_memory_mb=peak_mb,
            split_path=(
                str(
                    (workdir.parent if workdir.name == "baseline" else workdir.parents[1])
                    / "split_verification.json"
                )
                if (
                    (workdir.parent if workdir.name == "baseline" else workdir.parents[1])
                    / "split_verification.json"
                ).is_file()
                else None
            ),
            model_path=(str(model_path) if workdir.name != "baseline" else None),
            compute_attestation=self.compute_attestation,
            device="gpu",
            scientific_status="infrastructure_only",
            claim_eligible=False,
            operation_receipt_path=(
                str(operation_receipt_path) if operation_receipt_path is not None else None
            ),
            additional_artifact_paths=[
                str(path)
                for path in (
                    workdir / "model_manifest.json",
                    workdir / "training_evidence.json",
                    workdir / "operation" / "operation_request.json",
                    workdir / "operation" / "operation_started.json",
                )
                if path.is_file()
            ],
        )

    def _predict(
        self,
        view: BoundedLabelView,
        manifest: MACECheckpointManifest,
        model_path: Path,
        output_path: Path,
        workdir: Path,
    ) -> None:
        calculator = _make_calculator(model_path, "cuda", "float64")
        predictions: list[dict[str, object]] = []
        structures: list[dict[str, object]] = []
        for record in view.validation:
            atoms = Atoms(
                symbols=record.symbols,
                positions=record.positions,
                cell=record.cell,
                pbc=record.pbc,
            )
            atoms.calc = calculator
            predictions.append(
                {
                    "record_id": record.config_id,
                    "energy_ev": float(atoms.get_potential_energy()),  # type: ignore[no-untyped-call]
                    "forces_ev_per_a": [
                        [float(value) for value in row]
                        for row in atoms.get_forces()  # type: ignore[no-untyped-call]
                    ],
                }
            )
            structures.append(
                {
                    "record_id": record.config_id,
                    "symbols": record.symbols,
                    "positions": record.positions,
                    "cell": record.cell,
                    "pbc": record.pbc,
                }
            )
        structure_hash = sha256_of_text(canonical_json(structures))
        checkpoint_sha = sha256_file(model_path)
        model_id = f"{manifest.model_id}-infrastructure-{checkpoint_sha[:12]}"
        step_id = "baseline" if workdir.name == "baseline" else workdir.parent.name
        model_manifest_id = f"{step_id}:model_manifest.json"
        prediction_artifact_id = f"{step_id}:predictions.json"
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": "2.0.0",
                    "dataset_id": view.dataset_id,
                    "dataset_content_sha256": view.source_dataset_content_sha256,
                    "split_semantic_sha256": view.split_semantic_sha256,
                    "model_id": model_id,
                    "model_manifest_artifact": model_manifest_id,
                    "checkpoint_sha256": checkpoint_sha,
                    "energy_unit": "eV",
                    "force_unit": "eV/angstrom",
                    "device": "cuda",
                    "default_dtype": "float64",
                    "structure_set_sha256": structure_hash,
                    "predictions": predictions,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (workdir / "model_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "2.0.0",
                    "model_id": model_id,
                    "scientific_status": "infrastructure_only",
                    "claim_eligible": False,
                    "checkpoint": {"sha256": checkpoint_sha},
                    "predictions_artifact": prediction_artifact_id,
                    "structure_set_sha256": structure_hash,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
