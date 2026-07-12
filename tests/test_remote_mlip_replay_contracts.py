"""Focused contracts for the bounded real-MACE replay infrastructure."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mlip_research_agent.data.bounded_view import (
    EXPECTED_DATASET_CONTENT_SHA256,
    EXPECTED_NORMALIZED_MANIFEST_SHA256,
    EXPECTED_SPLIT_MANIFEST_SHA256,
    BoundedLabelView,
    ReplayLabeledConfiguration,
)
from mlip_research_agent.research.auto_research import (
    AgentReview,
    AutoResearchController,
    ExperimentProposal,
    LocalDemoConfig,
    LoopState,
    MutationPolicy,
    ReviewSynthesis,
    SyntheticAggregateEvaluator,
    SyntheticQuadraticAdapter,
    generate_proposal,
)
from mlip_research_agent.research.auto_research.adapters.mace_phase2 import (
    build_initial_mace_proposal,
)
from mlip_research_agent.research.auto_research.operations import (
    ExactlyOnceOperation,
    OptimizerOperationRequest,
)
from mlip_research_agent.skills.evaluation.mlip_metrics.implementation import (
    evaluate_bounded_validation,
)
from mlip_research_agent.skills.evaluation.mlip_metrics.schema import (
    BoundedValidationRequest,
)
from mlip_research_agent.skills.mlip.mace_finetune.runner import (
    _configure_trainable_layers,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(index: int) -> ReplayLabeledConfiguration:
    return ReplayLabeledConfiguration(
        config_id=f"fixture-{index:03d}",
        source_id=f"fixture[{index}]",
        top_group="fixture",
        group_id="fixture",
        split_unit_id=f"unit-{index:03d}",
        symbols=["Cu"],
        positions=[[float(index), 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=[True, True, True],
        energy_ev=-1.0,
        forces_ev_per_a=[[0.0, 0.0, 0.0]],
        level_of_theory="synthetic",
        source_record_sha256=f"{index + 1:064x}",
    )


def _view() -> BoundedLabelView:
    return BoundedLabelView(
        source_dataset_content_sha256=EXPECTED_DATASET_CONTENT_SHA256,
        normalized_manifest_sha256=EXPECTED_NORMALIZED_MANIFEST_SHA256,
        split_manifest_sha256=EXPECTED_SPLIT_MANIFEST_SHA256,
        split_semantic_sha256="a" * 64,
        initial_labeled=[_record(i) for i in range(32)],
        validation=[_record(i) for i in range(32, 72)],
    )


def test_bounded_view_is_exact_and_has_no_stress_field(tmp_path: Path) -> None:
    view = _view()
    path = tmp_path / "view.json"
    view.save(path)
    assert len(view.initial_labeled) == 32
    assert len(view.validation) == 40
    assert "stress" not in path.read_text().lower()
    with pytest.raises(ValueError, match="exactly 40"):
        BoundedLabelView.model_validate(
            {**view.model_dump(mode="json"), "validation": view.validation[:-1]}
        )


def test_typed_wp5_bounded_boundary_emits_provenance_and_aggregates(tmp_path: Path) -> None:
    view = _view()
    view_path = tmp_path / "view.json"
    view_sha = view.save(view_path)
    predictions = {
        "schema_version": "2.0.0",
        "dataset_id": "cu_phase2",
        "dataset_content_sha256": EXPECTED_DATASET_CONTENT_SHA256,
        "split_semantic_sha256": "a" * 64,
        "model_id": "fixture-model",
        "model_manifest_artifact": "iteration-001:model_manifest.json",
        "checkpoint_sha256": "b" * 64,
        "energy_unit": "eV",
        "force_unit": "eV/angstrom",
        "structure_set_sha256": "c" * 64,
        "predictions": [
            {
                "record_id": record.config_id,
                "energy_ev": record.energy_ev,
                "forces_ev_per_a": record.forces_ev_per_a,
            }
            for record in view.validation
        ],
    }
    predictions_path = tmp_path / "predictions.json"
    rerun_path = tmp_path / "predictions_rerun.json"
    predictions_path.write_text(json.dumps(predictions))
    rerun_path.write_text(json.dumps(predictions))
    model_manifest_path = tmp_path / "model_manifest.json"
    model_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "model_id": "fixture-model",
                "checkpoint": {"sha256": "b" * 64},
                "predictions_artifact": "iteration-001:predictions.json",
                "structure_set_sha256": "c" * 64,
            }
        )
    )
    artifact = evaluate_bounded_validation(
        request=BoundedValidationRequest(
            predictions_artifact="iteration-001:predictions.json",
            predictions_rerun_artifact="iteration-001:predictions_rerun.json",
            model_manifest_artifact="iteration-001:model_manifest.json",
            bounded_view_sha256=view_sha,
            dataset_content_sha256=EXPECTED_DATASET_CONTENT_SHA256,
            split_semantic_sha256="a" * 64,
            high_error_threshold_ev_per_a=1.0,
        ),
        bounded_view_path=view_path,
        predictions_path=predictions_path,
        predictions_rerun_path=rerun_path,
        model_manifest_path=model_manifest_path,
        output_path=tmp_path / "metrics.json",
    )
    assert artifact.boundary == "mlip_metrics/bounded_validation/1.0.0"
    assert artifact.aggregate.n_structures == 40
    assert artifact.aggregate.force_component_mae_ev_per_a == 0.0
    assert artifact.claim_eligible is False


def test_exactly_once_receipt_refuses_ambiguous_restart(tmp_path: Path) -> None:
    request = OptimizerOperationRequest(
        operation_id="iteration-001",
        git_commit="a" * 40,
        proposal_fingerprint="b" * 64,
        config_fingerprint="c" * 64,
        seed=1,
        dataset_content_sha256=EXPECTED_DATASET_CONTENT_SHA256,
        split_manifest_sha256=EXPECTED_SPLIT_MANIFEST_SHA256,
        checkpoint_sha256="d" * 64,
    ).sealed()
    operation = ExactlyOnceOperation(tmp_path, request)
    assert operation.begin() is None
    with pytest.raises(RuntimeError, match="ambiguous"):
        ExactlyOnceOperation(tmp_path, request).begin()
    operation.complete(model_sha256="e" * 64, checkpoint_sha256="f" * 64)
    completed = ExactlyOnceOperation(tmp_path, request).begin()
    assert completed is not None and completed.optimizer_steps == 1


def test_trainable_layer_policy_is_real() -> None:
    torch = pytest.importorskip("torch")

    class Fixture(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.interactions = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
            self.readouts = torch.nn.ModuleList([torch.nn.Linear(2, 1)])

    model = Fixture()
    trainable, frozen = _configure_trainable_layers(model, "readout_only")
    assert trainable and all(name.startswith("readouts.") for name in trainable)
    assert frozen and all(not name.startswith("readouts.") for name in frozen)
    trainable, _ = _configure_trainable_layers(model, "last_interaction_and_readout")
    assert any(name.startswith("interactions.1.") for name in trainable)


def _review(role: str, packet_sha256: str = "a" * 64) -> AgentReview:
    return AgentReview(
        role=role,
        review_status="continue",
        recommended_mutation_class="gradient_clip",
        recommended_direction="decrease within the recorded bound",
        scientific_rationale="This is a bounded infrastructure comparison, not a claim.",
        engineering_rationale="The next configuration remains deterministic and reversible.",
        risks=["validation reuse"],
        evidence_artifact_ids=["iteration_001_evaluation"],
        review_packet_sha256=packet_sha256,
        limitations=["single seed"],
    ).sealed()


def test_controller_pauses_for_sealed_external_reviews(tmp_path: Path) -> None:
    config = LocalDemoConfig.load(REPO_ROOT / "configs/research/ralphthon_local_demo.yaml")
    policy = MutationPolicy.load(REPO_ROOT / config.mutation_policy_path)
    controller = AutoResearchController(
        run_id="external-review-run",
        run_dir=tmp_path / "external-review-run",
        objective=config.objective.sealed(),
        mutation_policy=policy,
        adapter=SyntheticQuadraticAdapter(),
        evaluator=SyntheticAggregateEvaluator(),
        acceptance=config.acceptance,
        base_config=dict(config.base_config),
        fixture_seed=config.fixture_seed,
        git_commit="a" * 40,
        external_review_after_iteration=1,
    )
    assert controller.run() is LoopState.AWAITING_EXTERNAL_REVIEW
    assert not (controller.run_dir / "iteration-002").exists()
    for name in (
        "compute_attestation.json",
        "dataset_verification.json",
        "split_verification.json",
        "checkpoint_verification.json",
    ):
        (controller.run_dir / name).write_text("{}\n")
    from mlip_research_agent.research.auto_research.review import build_review_packet

    packet = build_review_packet(controller.run_dir)
    reviews = [
        _review(role, packet.content_sha256)
        for role in ("mlip_scientist", "active_learning_scientist", "scientific_auditor")
    ]
    evidence = {}
    for name in ("proposal", "evaluation", "decision", "lesson"):
        payload = __import__("json").loads(
            (controller.run_dir / "iteration-001" / f"{name}.json").read_text()
        )
        evidence[name] = payload["content_sha256"]
    synthesis = ReviewSynthesis(
        review_hashes={review.role: review.content_sha256 for review in reviews},
        recommended_mutation_class="gradient_clip",
        recommended_direction="decrease",
        rationale="Three host reviews support one more bounded infrastructure iteration.",
        accepted_recommendations=["remain bounded"],
        rejected_recommendations=["no checkpoint promotion"],
        iteration_evidence_sha256=evidence,
        review_packet_sha256=packet.content_sha256,
    ).sealed()
    proposal = generate_proposal(
        objective=controller.objective,
        policy=policy,
        current_config=controller.store.current_config(),
        iteration_index=controller.state.iteration_index,
        prior=controller._prior_outcome(),
        seed=config.fixture_seed + 1,
    )
    proposal = proposal.model_copy(
        update={
            "required_skills": [
                *proposal.required_skills,
                f"review_synthesis:{synthesis.content_sha256}",
            ]
        }
    ).sealed()
    controller.resume_with_external_review(
        reviews=reviews,
        review_packet=packet,
        synthesis=synthesis,
        proposal=proposal,
    )
    assert controller.run() is LoopState.COMPLETE


def _host_module() -> ModuleType:
    path = REPO_ROOT / "scripts/colab/run_ralphthon_mace_replay.py"
    spec = importlib.util.spec_from_file_location("replay_host", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _remote_module() -> ModuleType:
    path = REPO_ROOT / "scripts/research/run_mace_auto_research.py"
    spec = importlib.util.spec_from_file_location("replay_remote", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_host_contract_rejects_non_l4_and_short_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _host_module()
    monkeypatch.setattr(
        module,
        "_git",
        lambda *args: "a" * 40 if args[:2] == ("rev-parse", "HEAD") else "",
    )
    with pytest.raises(module.ReplayContractError, match="full 40"):
        module.validate_replay_contract(
            commit="abc1234", gpu="L4", n_gpus=1, max_runtime_minutes=60
        )
    with pytest.raises(module.ReplayContractError, match="only NVIDIA L4"):
        module.validate_replay_contract(
            commit="a" * 40, gpu="A100", n_gpus=1, max_runtime_minutes=60
        )


def test_iteration_one_pullback_manifest_rejects_tampering(tmp_path: Path) -> None:
    module = _host_module()
    artifact = tmp_path / "iteration-001" / "proposal.json"
    artifact.parent.mkdir()
    artifact.write_text("original\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "artifact_id": "iteration-001:proposal.json",
                    "relative_path": "iteration-001/proposal.json",
                    "sha256": digest,
                    "size_bytes": artifact.stat().st_size,
                    "kind": "auto_research",
                    "created_by_step": "iteration-001",
                }
            ]
        )
    )
    module.verify_registered_manifest(tmp_path)
    artifact.write_text("tampered\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        module.verify_registered_manifest(tmp_path)


@pytest.mark.parametrize("failure_stage", ["new", "upload"])
def test_host_cleanup_runs_after_allocation_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    module = _host_module()
    label_view = tmp_path / "view.json"
    label_view.write_text("{}")
    monkeypatch.setattr(module, "validate_replay_contract", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "sha256_file",
        lambda path: module.EXPECTED_BOUNDED_VIEW_FILE_SHA256,
    )
    monkeypatch.setattr(module.BoundedLabelView, "load", lambda path: object())
    monkeypatch.setattr(module.SplitManifest, "load", lambda path: object())
    monkeypatch.setattr(module, "verify_bounded_view_membership", lambda view, split: None)
    stopped: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> str:
        if command[:2] == ["colab", "sessions"]:
            return ""
        if failure_stage == "new" and command[:2] == ["colab", "new"]:
            raise TimeoutError("allocation transport timeout")
        if failure_stage == "upload" and command[:2] == ["colab", "upload"]:
            raise RuntimeError("upload failed")
        return ""

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: stopped.append(command)
        or SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(module, "_confirm_stopped", lambda session, deadline: "confirmed_absent")
    args = SimpleNamespace(
        commit="a" * 40,
        gpu="L4",
        n_gpus=1,
        max_runtime_minutes=60,
        label_view=label_view,
        label_view_sha256=module.EXPECTED_BOUNDED_VIEW_FILE_SHA256,
        output=tmp_path / "output",
        review_bundle=tmp_path / "reviews",
    )
    with pytest.raises((RuntimeError, TimeoutError)):
        module.run_replay(args)
    assert any(command[:2] == ["colab", "stop"] for command in stopped)


def test_cleanup_requires_stable_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _host_module()
    observations = iter(["", "ral-aaaaaaaa", "", "", ""])
    polls: list[int] = []

    def sessions(command: list[str], **kwargs: object) -> str:
        polls.append(1)
        return next(observations)

    monkeypatch.setattr(module, "_run", sessions)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    assert (
        module._confirm_stopped("ral-aaaaaaaa", deadline=time.monotonic() + 30)
        == "confirmed_absent_stable"
    )
    assert len(polls) == 5


@pytest.mark.parametrize("torch_count,smi", [(0, ""), (2, "GPU 0\nGPU 1\n"), (1, "")])
def test_remote_rejects_observed_gpu_count_mismatch(torch_count: int, smi: str) -> None:
    module = _remote_module()
    with pytest.raises(RuntimeError, match="exactly one observed"):
        module.validate_observed_gpu_count(torch_count, smi)
    assert module.validate_observed_gpu_count(1, "GPU 0: NVIDIA L4\n") == 1


def test_initial_mace_proposal_is_remote_gpu_and_not_fixture_language() -> None:
    config = LocalDemoConfig.load(REPO_ROOT / "configs/research/ralphthon_mace_replay.yaml")
    policy = MutationPolicy.load(REPO_ROOT / config.mutation_policy_path)
    proposal = build_initial_mace_proposal(
        objective=config.objective,
        policy=policy,
        config=dict(config.base_config),
        seed=config.fixture_seed,
    )
    assert proposal.verify_seal()
    assert proposal.estimated_compute.device == "gpu"
    assert proposal.estimated_compute.remote is True
    assert "synthetic" not in proposal.model_dump_json().lower()
    legality = policy.check_mutations(
        proposal.proposal_id,
        proposal.proposed_mutations,
        dict(config.base_config),
        allowed_classes=config.objective.allowed_mutation_classes,
    )
    assert legality.legal


def test_host_review_sealer_binds_iteration_one_and_makes_legal_proposal(
    tmp_path: Path,
) -> None:
    config_path = REPO_ROOT / "configs/research/ralphthon_local_demo.yaml"
    config = LocalDemoConfig.load(config_path)
    policy = MutationPolicy.load(REPO_ROOT / config.mutation_policy_path)
    controller = AutoResearchController(
        run_id="seal-review-run",
        run_dir=tmp_path / "seal-review-run",
        objective=config.objective.sealed(),
        mutation_policy=policy,
        adapter=SyntheticQuadraticAdapter(),
        evaluator=SyntheticAggregateEvaluator(),
        acceptance=config.acceptance,
        base_config=dict(config.base_config),
        fixture_seed=config.fixture_seed,
        git_commit="a" * 40,
        external_review_after_iteration=1,
    )
    assert controller.run() is LoopState.AWAITING_EXTERNAL_REVIEW
    from mlip_research_agent.research.auto_research.review import (
        build_review_packet,
        verify_review_packet,
    )

    for name in (
        "compute_attestation.json",
        "dataset_verification.json",
        "split_verification.json",
        "checkpoint_verification.json",
    ):
        (controller.run_dir / name).write_text("{}\n")
    packet = build_review_packet(controller.run_dir)
    (controller.run_dir / "review_packet.json").write_text(packet.canonical_text())
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    for role in ("mlip_scientist", "active_learning_scientist", "scientific_auditor"):
        payload = _review(role).model_dump(mode="json", exclude={"content_sha256", "role"})
        (drafts / f"{role}.json").write_text(json.dumps(payload))
    (drafts / "synthesis.json").write_text(
        json.dumps(
            {
                "recommended_mutation_class": "gradient_clip",
                "recommended_direction": "decrease within the recorded bound",
                "rationale": "The three bounded reviews support a reversible stability test.",
                "accepted_recommendations": ["one bounded mutation"],
                "rejected_recommendations": ["no checkpoint selection"],
            }
        )
    )
    sealer_path = REPO_ROOT / "scripts/research/seal_mace_review_bundle.py"
    spec = importlib.util.spec_from_file_location("review_sealer", sealer_path)
    assert spec is not None and spec.loader is not None
    sealer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sealer)
    output = sealer.seal_bundle(
        run_dir=controller.run_dir,
        drafts_dir=drafts,
        output_dir=tmp_path / "sealed",
        config_path=config_path,
    )
    synthesis = ReviewSynthesis.model_validate_json((output / "synthesis.json").read_text())
    proposal = ExperimentProposal.model_validate_json((output / "proposal.json").read_text())
    assert synthesis.verify_seal()
    assert proposal.verify_seal()
    assert proposal.parent_iteration_id == "iteration-001"
    assert f"review_synthesis:{synthesis.content_sha256}" in proposal.required_skills
    assert {"mace_finetune", "mlip_metrics"}.issubset(proposal.required_skills)
    assert (output / "READY").read_text() == "sealed\n"
    (controller.run_dir / "iteration-001/evaluation.json").write_text("{}\n")
    with pytest.raises(ValueError, match="does not match"):
        verify_review_packet(packet, controller.run_dir)
