"""WP4 gates: lineage/leakage validation, failure taxonomy, resume contract,
plus the opt-in real CPU one-step + checkpoint round-trip fixture."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from ase.build import bulk

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.data.manifests import LabeledConfiguration, NormalizedDataset
from mlip_research_agent.data.oracle import TrainingLineageManifest
from mlip_research_agent.data.registry import NormalizedManifest
from mlip_research_agent.data.split import (
    PartitionName,
    SplitManifest,
    SplitSpec,
    build_split_manifest,
)
from mlip_research_agent.schemas.failure import FailureClass, RecoveryDecision
from mlip_research_agent.skills.base import SkillContext, SkillError, get_skill
from mlip_research_agent.skills.mlip.mace_finetune.implementation import MACEFineTuneSkill
from mlip_research_agent.skills.mlip.mace_finetune.runner import (
    ControlledTrainingResult,
    run_controlled_training,
)
from mlip_research_agent.skills.mlip.mace_finetune.schema import (
    FineTuneResumeState,
    FineTuneSubsetManifest,
    MACEFineTuneInput,
    MACEFineTuneOutput,
)
from mlip_research_agent.skills.mlip.mace_inference.checkpoint import (
    MACECheckpointManifest,
    resolve_checkpoint,
)

ROOT = Path(__file__).resolve().parents[6]
REAL_MANIFEST_PATH = ROOT / "configs/models/mace_mp_0_small_candidate.json"

RUNNER_TARGET = (
    "mlip_research_agent.skills.mlip.mace_finetune.implementation.run_controlled_training"
)


def make_dataset() -> NormalizedDataset:
    records: list[LabeledConfiguration] = []
    for index in range(12):
        records.append(
            LabeledConfiguration(
                config_id=f"finetune-fixture-{index:02d}",
                source_id=f"fixture.json[{index}]",
                top_group=f"family-{index // 4}",
                group_id=f"trajectory-{index // 2}",
                split_unit_id=f"unit-{index:02d}",
                symbols=["Cu", "Cu"],
                positions=[[0.0, 0.0, 0.0], [1.0 + index / 100, 1.0, 1.0]],
                cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                pbc=[True, True, True],
                energy_ev=-7.0 - index / 10,
                forces_ev_per_a=[[index / 100, 0.0, 0.0], [0.0, -index / 100, 0.0]],
                virial_stress_kbar=None,
                level_of_theory="synthetic-test",
                source_record_sha256=f"{index + 100:064x}",
            )
        )
    return NormalizedDataset(
        dataset_id="finetune_fixture",
        level_of_theory="synthetic-test",
        configurations=records,
    )


class Fixture:
    """Registered dataset/split/subset/lineage inputs for one fine-tune run."""

    def __init__(self, tmp_path: Path, *, train_ids: list[str] | None = None) -> None:
        inputs = tmp_path / "inputs"
        self.registry = ArtifactRegistry(tmp_path)
        self.dataset = make_dataset()
        dataset_path = inputs / "normalized_dataset.json"
        dataset_file_sha = self.dataset.save(dataset_path)
        manifest = NormalizedManifest.from_dataset(self.dataset, dataset_file_sha)
        manifest_path = inputs / "normalized_manifest.json"
        manifest.save(manifest_path)
        self.split: SplitManifest = build_split_manifest(
            manifest,
            SplitSpec(
                initial_labeled=2,
                acquisition_pool=6,
                validation=2,
                frozen_test=2,
                seed=55,
            ),
            qualified_manifest_path=manifest_path,
            qualified_manifest_reference="inputs/normalized_manifest.json",
        )
        split_path = inputs / "split_manifest.json"
        self.split.save(split_path)

        initial = self.split.record_ids[PartitionName.INITIAL_LABELED.value]
        self.train_ids = sorted(train_ids if train_ids is not None else initial)
        lineage = TrainingLineageManifest(
            campaign_id="pilot-random",
            dataset_id=self.dataset.dataset_id,
            dataset_content_sha256=self.dataset.content_hash(),
            split_semantic_sha256=self.split.semantic_hash(),
            initial_record_ids=sorted(initial),
            reveal_history=[],
            training_record_ids=self.train_ids,
        )
        lineage_path = inputs / "training_lineage.json"
        lineage_path.write_text(
            json.dumps(lineage.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )

        self.dataset_artifact = self.registry.register(
            dataset_path, "normalized_dataset", "inputs"
        )
        self.manifest_artifact = self.registry.register(
            manifest_path, "normalized_dataset_manifest", "inputs"
        )
        self.split_artifact = self.registry.register(split_path, "split_manifest", "inputs")
        self.lineage_artifact = self.registry.register(
            lineage_path, "training_lineage_manifest", "inputs"
        )

        train_subset = FineTuneSubsetManifest(
            role="train",
            dataset_id=self.dataset.dataset_id,
            dataset_content_sha256=self.dataset.content_hash(),
            split_semantic_sha256=self.split.semantic_hash(),
            record_ids=self.train_ids,
            training_lineage_artifact=self.lineage_artifact.artifact_id,
        )
        train_subset_path = inputs / "train_subset_manifest.json"
        train_subset.save(train_subset_path)
        validation_subset = FineTuneSubsetManifest(
            role="validation",
            dataset_id=self.dataset.dataset_id,
            dataset_content_sha256=self.dataset.content_hash(),
            split_semantic_sha256=self.split.semantic_hash(),
            record_ids=sorted(self.split.record_ids[PartitionName.VALIDATION.value]),
        )
        validation_subset_path = inputs / "validation_subset_manifest.json"
        validation_subset.save(validation_subset_path)
        self.train_subset_artifact = self.registry.register(
            train_subset_path, "fine_tune_subset_manifest", "inputs"
        )
        self.validation_subset_artifact = self.registry.register(
            validation_subset_path, "fine_tune_subset_manifest", "inputs"
        )

        checkpoint_dir = tmp_path / "checkpoint-cache"
        checkpoint_dir.mkdir()
        self.checkpoint_path = checkpoint_dir / "fixture-mace.model"
        payload = b"fixture checkpoint bytes"
        self.checkpoint_path.write_bytes(payload)
        self.checkpoint_manifest_path = checkpoint_dir / "fixture-mace-manifest.json"
        self.checkpoint_manifest_path.write_text(
            json.dumps(
                {
                    "model_id": "fixture-mace",
                    "family": "mace_mp",
                    "checkpoint_filename": "fixture-mace.model",
                    "source_url": "https://example.invalid/fixture-mace.model",
                    "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                    "checkpoint_size_bytes": len(payload),
                    "license_spdx": "MIT",
                    "mace_torch_version": "0.3.16",
                    "supported_species": ["Cu"],
                    "training_data_statement": "synthetic fixture, no real training data",
                    "scientific_status": "candidate_only",
                },
                indent=2,
                sort_keys=True,
            )
        )
        self.n_inputs = len(self.registry.all())

    def context(self, tmp_path: Path, step_id: str = "mace-finetune") -> SkillContext:
        return SkillContext(
            run_dir=tmp_path,
            step_id=step_id,
            seed=7,
            attempt=0,
            registry=self.registry,
        )

    def params(self, **overrides: Any) -> MACEFineTuneInput:
        payload: dict[str, Any] = {
            "dataset_artifact": self.dataset_artifact.artifact_id,
            "dataset_manifest_artifact": self.manifest_artifact.artifact_id,
            "split_manifest_artifact": self.split_artifact.artifact_id,
            "train_subset_manifest_artifact": self.train_subset_artifact.artifact_id,
            "validation_subset_manifest_artifact": (
                self.validation_subset_artifact.artifact_id
            ),
            "training_lineage_artifact": self.lineage_artifact.artifact_id,
            "checkpoint_manifest_path": str(self.checkpoint_manifest_path),
            "checkpoint_path": str(self.checkpoint_path),
            "run_name": "boundary-fixture",
            "seed": 7,
            "max_epochs": 1,
            "max_optimizer_steps": 1,
            "early_stopping_patience": 1,
            "gradient_clip": 10.0,
            "learning_rate": 0.01,
            "batch_size": 8,
            "valid_batch_size": 2,
            "device": "cpu",
            "max_wall_seconds": 600,
        }
        payload.update(overrides)
        return MACEFineTuneInput.model_validate(payload)


def fake_runner_result() -> ControlledTrainingResult:
    return ControlledTrainingResult(
        completed_epochs=1,
        optimizer_steps=1,
        best_validation_force_mae_ev_per_a=0.125,
        patience_count=0,
        stopped_early=False,
        records=[
            {
                "epoch": 0,
                "optimizer_step": 1,
                "training_loss": 1.5,
                "validation_loss": 1.25,
                "validation_force_component_mae_ev_per_a": 0.125,
                "learning_rate": 0.01,
            }
        ],
        changed_parameter_tensors=3,
        max_abs_parameter_change=0.05,
    )


def install_fake_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(**kwargs: Any) -> ControlledTrainingResult:
        Path(kwargs["checkpoint_path"]).write_bytes(b"controlled checkpoint")
        Path(kwargs["model_path"]).write_bytes(b"fine-tuned model")
        return fake_runner_result()

    monkeypatch.setattr(RUNNER_TARGET, fake)


def test_skill_is_registered() -> None:
    assert get_skill("mace_finetune") is MACEFineTuneSkill


def test_schema_blocks_pilot_without_frozen_preregistration(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(ValueError, match="frozen H2 preregistration"):
        fixture.params(execution_mode="pilot")
    with pytest.raises(ValueError, match="two epochs"):
        fixture.params(max_epochs=3, max_optimizer_steps=3)


def test_pilot_mode_is_disabled_before_h2(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    params = fixture.params(execution_mode="pilot", h2_preregistration_frozen=True)
    with pytest.raises(SkillError, match="pilot fine-tuning is disabled"):
        MACEFineTuneSkill().run(params, fixture.context(tmp_path))


def test_success_registers_model_manifest_metrics_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_runner(monkeypatch)
    fixture = Fixture(tmp_path)
    ctx = fixture.context(tmp_path)
    output = MACEFineTuneSkill().run(fixture.params(), ctx)
    assert isinstance(output, MACEFineTuneOutput)
    assert output.n_train == 2
    assert output.n_validation == 2
    assert output.optimizer_steps == 1
    assert output.changed_parameter_tensors == 3
    assert not output.resumed
    assert len(ctx.registry.all()) == fixture.n_inputs + 6
    assert all(ctx.registry.verify(item.artifact_id) for item in ctx.registry.all())

    manifest = json.loads((tmp_path / output.model_manifest_path).read_text())
    assert manifest["scientific_status"] == "training_boundary_only"
    assert manifest["dataset_content_sha256"] == fixture.dataset.content_hash()
    assert manifest["split_semantic_sha256"] == fixture.split.semantic_hash()
    assert manifest["training_lineage_artifact"] == fixture.lineage_artifact.artifact_id
    assert manifest["changed_parameter_tensors"] == 3

    state = FineTuneResumeState.model_validate_json(
        (tmp_path / output.training_state_path).read_text()
    )
    assert state.optimizer_steps == 1
    assert state.checkpoint_artifact == output.checkpoint_artifact
    metrics = json.loads((tmp_path / output.training_metrics_path).read_text())
    assert metrics["monitor"] == "validation_force_component_mae_ev_per_a"
    assert metrics["records"][0]["optimizer_step"] == 1


def test_training_subset_cannot_cross_protected_partitions(tmp_path: Path) -> None:
    plain = Fixture(tmp_path / "plain")
    frozen_id = plain.split.record_ids[PartitionName.FROZEN_TEST.value][0]
    initial = plain.split.record_ids[PartitionName.INITIAL_LABELED.value]
    fixture = Fixture(tmp_path / "leaky", train_ids=sorted([*initial, frozen_id]))
    with pytest.raises(SkillError, match="protected partition"):
        MACEFineTuneSkill().run(fixture.params(), fixture.context(tmp_path / "leaky"))


def test_batch_below_training_subset_is_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(SkillError, match="batch_size >= the training subset"):
        MACEFineTuneSkill().run(fixture.params(batch_size=1), fixture.context(tmp_path))


def test_checkpoint_hash_mismatch_aborts(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.checkpoint_path.write_bytes(b"tampered checkpoint bytes!!")
    with pytest.raises(SkillError, match="checkpoint validation failed"):
        MACEFineTuneSkill().run(fixture.params(), fixture.context(tmp_path))


def test_unregistered_artifact_is_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    params = fixture.params(dataset_artifact="inputs:not-registered.json")
    with pytest.raises(SkillError, match="not registered"):
        MACEFineTuneSkill().run(params, fixture.context(tmp_path))


def test_oom_failure_emits_failure_artifact_and_batch_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(**kwargs: Any) -> ControlledTrainingResult:
        raise RuntimeError("CUDA out of memory while allocating training batch")

    monkeypatch.setattr(RUNNER_TARGET, explode)
    fixture = Fixture(tmp_path)
    ctx = fixture.context(tmp_path)
    with pytest.raises(SkillError) as excinfo:
        MACEFineTuneSkill().run(fixture.params(), ctx)
    error = excinfo.value
    assert error.failure_class is FailureClass.RESOURCE_EXHAUSTED
    assert error.retryable
    assert error.recommended_action is RecoveryDecision.REFINE
    assert error.repair_params == {"batch_size": 4}
    kinds = {item.kind for item in ctx.registry.all()}
    assert "mace_fine_tune_failure" in kinds
    assert "fine_tuned_mace_model" not in kinds
    failure_dir = ctx.step_dir / "failed-attempt-000"
    assert failure_dir.is_dir()
    payload = json.loads(
        (failure_dir / "fine_tune_failure-attempt-000.json").read_text()
    )
    assert payload["successful_model_emitted"] is False
    assert payload["repair_params"] == {"batch_size": 4}


def test_nan_failure_repairs_learning_rate_and_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(**kwargs: Any) -> ControlledTrainingResult:
        raise ValueError("non-finite training or validation metric")

    monkeypatch.setattr(RUNNER_TARGET, explode)
    fixture = Fixture(tmp_path)
    ctx = fixture.context(tmp_path)
    with pytest.raises(SkillError) as excinfo:
        MACEFineTuneSkill().run(fixture.params(), ctx)
    error = excinfo.value
    assert error.failure_class is FailureClass.SIMULATION_INSTABILITY
    assert error.retryable
    assert error.repair_params == {"learning_rate": 0.005, "gradient_clip": 5.0}


def test_resume_contract_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_runner(monkeypatch)
    fixture = Fixture(tmp_path)
    stale = FineTuneResumeState(
        run_name="boundary-fixture",
        seed=7,
        completed_epochs=1,
        optimizer_steps=1,
        best_validation_force_mae_ev_per_a=0.125,
        patience_count=0,
        checkpoint_artifact="inputs:stale-checkpoint.pt",
        model_artifact="inputs:stale-model.model",
        resume_contract_sha256="0" * 64,
        base_checkpoint_sha256="1" * 64,
        dataset_content_sha256=fixture.dataset.content_hash(),
        split_semantic_sha256=fixture.split.semantic_hash(),
    )
    state_path = tmp_path / "inputs" / "stale-state.json"
    state_path.write_text(
        json.dumps(stale.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    state_artifact = fixture.registry.register(state_path, "mace_training_state", "inputs")
    params = fixture.params(
        resume_state_artifact=state_artifact.artifact_id,
        max_epochs=2,
        max_optimizer_steps=2,
    )
    with pytest.raises(SkillError, match="resume contract hash mismatch"):
        MACEFineTuneSkill().run(params, fixture.context(tmp_path))


def _real_checkpoint() -> Path:
    value = os.environ.get("MACE_TEST_CHECKPOINT")
    if value is None:
        pytest.skip("set MACE_TEST_CHECKPOINT for the opt-in real CPU fixture")
    return Path(value).resolve()


def _bulk_records() -> list[LabeledConfiguration]:
    records: list[LabeledConfiguration] = []
    for index in range(5):
        atoms = bulk("Cu", "fcc", a=3.58 + index * 0.02, cubic=True)
        records.append(
            LabeledConfiguration(
                config_id=f"real-fixture-{index:02d}",
                source_id=f"synthetic[{index}]",
                top_group="real-fixture",
                group_id="real-fixture",
                split_unit_id=f"real-unit-{index:02d}",
                symbols=["Cu"] * len(atoms),
                positions=[[float(x) for x in row] for row in atoms.positions],
                cell=[[float(x) for x in row] for row in atoms.cell[:]],
                pbc=[True, True, True],
                energy_ev=-14.5 - index * 0.05,
                forces_ev_per_a=[[0.0, 0.0, 0.0] for _ in range(len(atoms))],
                virial_stress_kbar=None,
                level_of_theory="synthetic-test",
                source_record_sha256=f"{index + 200:064x}",
            )
        )
    return records


def test_real_cpu_one_step_and_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint = _real_checkpoint()
    manifest = MACECheckpointManifest.load(REAL_MANIFEST_PATH)
    resolve_checkpoint(manifest, checkpoint)
    records = _bulk_records()
    train_records, validation_records = records[:3], records[3:]
    common: dict[str, Any] = {
        "foundation_path": checkpoint,
        "train_records": train_records,
        "validation_records": validation_records,
        "resume_contract_sha256": "a" * 64,
        "seed": 3,
        "optimizer_name": "adam",
        "learning_rate": 0.005,
        "gradient_clip": 10.0,
        "batch_size": 4,
        "valid_batch_size": 2,
        "patience": 5,
        "device_name": "cpu",
        "default_dtype": "float64",
        "max_wall_seconds": 900,
    }
    controlled_checkpoint = tmp_path / "controlled-checkpoint.pt"
    first = run_controlled_training(
        checkpoint_path=controlled_checkpoint,
        model_path=tmp_path / "step-one.model",
        resume_checkpoint_path=None,
        max_epochs=1,
        max_optimizer_steps=1,
        **common,
    )
    assert first.completed_epochs == 1
    assert first.optimizer_steps == 1
    assert first.changed_parameter_tensors > 0
    assert first.max_abs_parameter_change > 0.0
    assert controlled_checkpoint.is_file()

    resumed = run_controlled_training(
        checkpoint_path=tmp_path / "controlled-checkpoint-resumed.pt",
        model_path=tmp_path / "step-two.model",
        resume_checkpoint_path=controlled_checkpoint,
        max_epochs=2,
        max_optimizer_steps=2,
        **common,
    )
    assert resumed.completed_epochs == 2
    assert resumed.optimizer_steps == 2
    assert resumed.records[0] == first.records[0]
    assert (tmp_path / "step-two.model").is_file()
