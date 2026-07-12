#!/usr/bin/env python
"""Execute or exactly resume the remote MACE Auto Research controller."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import urllib.request
from importlib import metadata
from pathlib import Path

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.compute.attestation import perform_attestation, probe_nvidia_smi
from mlip_research_agent.compute.policy import Decision, load_policy
from mlip_research_agent.compute.schemas import (
    AcceleratorId,
    JobSpec,
    ProviderName,
    WorkloadClass,
)
from mlip_research_agent.data.bounded_view import (
    EXPECTED_BOUNDED_VIEW_FILE_SHA256,
    BoundedLabelView,
    verify_bounded_view_membership,
)
from mlip_research_agent.data.split import SplitManifest
from mlip_research_agent.research.auto_research import (
    AgentReview,
    AutoResearchController,
    ExperimentProposal,
    LocalDemoConfig,
    MutationPolicy,
    ReviewPacket,
    ReviewSynthesis,
)
from mlip_research_agent.research.auto_research.adapters.mace_phase2 import (
    MACEPhase2Adapter,
    build_initial_mace_proposal,
)
from mlip_research_agent.research.auto_research.mace_evaluator import MACEValidationEvaluator
from mlip_research_agent.skills.mlip.mace_inference.checkpoint import (
    MACECheckpointManifest,
    resolve_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def validate_observed_gpu_count(torch_count: int, nvidia_smi_output: str) -> int:
    listed = [line for line in nvidia_smi_output.splitlines() if line.strip()]
    if torch_count != 1 or len(listed) != 1:
        raise RuntimeError("replay requires exactly one observed NVIDIA GPU")
    return 1


def _checkpoint() -> tuple[Path, Path]:
    manifest_path = REPO_ROOT / "configs/models/mace_mp_0_small_candidate.json"
    manifest = MACECheckpointManifest.load(manifest_path)
    cache = REPO_ROOT / "data/cache/models"
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / manifest.checkpoint_filename
    if not checkpoint.is_file():
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".download")
        urllib.request.urlretrieve(manifest.source_url, temporary)
        temporary.replace(checkpoint)
    resolve_checkpoint(manifest, checkpoint)
    return manifest_path, checkpoint


def _controller(args: argparse.Namespace) -> AutoResearchController:
    config = LocalDemoConfig.load(REPO_ROOT / args.config)
    policy = MutationPolicy.load(REPO_ROOT / config.mutation_policy_path)
    manifest, checkpoint = _checkpoint()
    adapter = MACEPhase2Adapter(
        label_view_path=args.label_view,
        checkpoint_manifest_path=manifest,
        checkpoint_path=checkpoint,
        label_view_sha256=args.label_view_sha256,
        git_commit=args.commit,
        compute_attestation="run:compute_attestation.json",
    )
    initial_proposal = build_initial_mace_proposal(
        objective=config.objective,
        policy=policy,
        config=dict(config.base_config),
        seed=config.fixture_seed,
    )
    return AutoResearchController(
        run_id=args.run_id,
        run_dir=REPO_ROOT / config.output_root / args.run_id,
        objective=config.objective.sealed(),
        mutation_policy=policy,
        adapter=adapter,
        evaluator=MACEValidationEvaluator(args.label_view),
        acceptance=config.acceptance,
        base_config=dict(config.base_config),
        fixture_seed=config.fixture_seed,
        git_commit=args.commit,
        external_review_after_iteration=1,
        initial_proposal=initial_proposal,
    )


def _write_run_verifications(args: argparse.Namespace) -> None:
    config = LocalDemoConfig.load(REPO_ROOT / args.config)
    run_dir = REPO_ROOT / config.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    view = BoundedLabelView.load(args.label_view)
    if args.label_view_sha256 != EXPECTED_BOUNDED_VIEW_FILE_SHA256:
        raise RuntimeError("uploaded label view is not the authorized projection")
    split_manifest = SplitManifest.load(
        REPO_ROOT / "data_registry/datasets/cu_phase2/split_manifest.json"
    )
    verify_bounded_view_membership(view, split_manifest)
    manifest_path, checkpoint = _checkpoint()
    manifest = MACECheckpointManifest.load(manifest_path)
    resolved = resolve_checkpoint(manifest, checkpoint)
    observation = probe_nvidia_smi()
    if observation is None:
        raise RuntimeError("L4 replay has no visible GPU")
    policy = load_policy(REPO_ROOT / "configs/compute_policy.yaml")
    spec = JobSpec(
        name="ralphthon-mace-replay",
        workload_class=WorkloadClass.GPU_SMOKE_TEST,
        requested_accelerator=AcceleratorId.NVIDIA_L4,
        n_gpus=1,
        command=["python", "scripts/research/run_mace_auto_research.py"],
        max_runtime_minutes=60,
        repo_commit=args.commit,
        mlip_backend="mace",
    )
    attestation = perform_attestation(
        policy,
        ProviderName.COLAB,
        spec,
        observation,
        run_id=args.run_id,
    )
    if attestation.policy_decision is not Decision.ALLOW:
        raise RuntimeError(f"L4 attestation denied: {attestation.policy_reasons}")
    import torch

    observed_gpu_count = torch.cuda.device_count()
    nvidia_smi_output = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
    ).stdout
    validate_observed_gpu_count(observed_gpu_count, nvidia_smi_output)
    attestation_payload = attestation.model_dump(mode="json")
    attestation_payload["observed_gpu_count"] = observed_gpu_count
    files = {
        "compute_attestation.json": attestation_payload,
        "dataset_verification.json": {
            "dataset_content_sha256": view.source_dataset_content_sha256,
            "normalized_manifest_sha256": view.normalized_manifest_sha256,
            "bounded_label_view_sha256": args.label_view_sha256,
            "n_initial_labeled": len(view.initial_labeled),
            "n_validation": len(view.validation),
            "contains_only_d0_and_validation": True,
            "scientific_status": "infrastructure_only",
            "claim_eligible": False,
        },
        "split_verification.json": {
            "split_manifest_sha256": view.split_manifest_sha256,
            "split_semantic_sha256": view.split_semantic_sha256,
            "protected_partitions_accessed": False,
            "acquisition_pool_labels_accessed": False,
            "scientific_status": "infrastructure_only",
            "claim_eligible": False,
        },
        "checkpoint_verification.json": {
            "checkpoint_sha256": resolved.sha256,
            "checkpoint_size_bytes": resolved.size_bytes,
            "model_id": resolved.model_id,
            "mace_torch_version": metadata.version("mace-torch"),
            "scientific_status": "candidate_only",
            "claim_eligible": False,
        },
    }
    freeze_path = run_dir / "pip_freeze.txt"
    freeze_text = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    if freeze_path.exists() and freeze_path.read_text() != freeze_text:
        raise RuntimeError("immutable pip freeze changed")
    freeze_path.write_text(freeze_text)
    lock_path = REPO_ROOT / "scripts/colab/ralphthon_mace_replay_requirements.txt"
    files["environment.json"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": args.commit,
        "adapter": "mace_phase2_infrastructure_replay/1.0.0",
        "evaluator": "mlip_validation_evaluator/1.0.0",
        "dependency_lock_sha256": sha256_file(lock_path),
        "pip_freeze_sha256": sha256_file(freeze_path),
        "torch_version": metadata.version("torch"),
        "mace_torch_version": metadata.version("mace-torch"),
        "e3nn_version": metadata.version("e3nn"),
        "numpy_version": metadata.version("numpy"),
        "ase_version": metadata.version("ase"),
        "scientific_status": "infrastructure_only",
        "claim_eligible": False,
    }
    for name, payload in files.items():
        path = run_dir / name
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise RuntimeError(f"immutable run verification changed: {name}")
            continue
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["iteration1", "iteration2"])
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--label-view", type=Path, required=True)
    parser.add_argument("--label-view-sha256", required=True)
    parser.add_argument("--config", default="configs/research/ralphthon_mace_replay.yaml")
    parser.add_argument("--review-bundle", type=Path)
    args = parser.parse_args()
    if len(args.commit) != 40:
        parser.error("--commit must be a full 40-character SHA")
    if sha256_file(args.label_view) != args.label_view_sha256:
        parser.error("--label-view-sha256 does not match the uploaded bounded view")
    if args.phase == "iteration1":
        _write_run_verifications(args)
    controller = _controller(args)
    if args.phase == "iteration1":
        state = controller.run()
        if state.value != "awaiting_external_review":
            raise RuntimeError(f"iteration 1 did not reach the review pause: {state.value}")
        print(json.dumps({"state": state.value, "run_id": args.run_id}))
        return 0
    if args.review_bundle is None:
        parser.error("iteration2 requires --review-bundle")
    reviews = [
        AgentReview.model_validate_json((args.review_bundle / f"{role}.json").read_text())
        for role in (
            "mlip_scientist",
            "active_learning_scientist",
            "scientific_auditor",
        )
    ]
    synthesis = ReviewSynthesis.model_validate_json(
        (args.review_bundle / "synthesis.json").read_text()
    )
    review_packet = ReviewPacket.model_validate_json(
        (args.review_bundle / "review_packet.json").read_text()
    )
    proposal = ExperimentProposal.model_validate_json(
        (args.review_bundle / "proposal.json").read_text()
    )
    controller.resume_with_external_review(
        reviews=reviews,
        review_packet=review_packet,
        synthesis=synthesis,
        proposal=proposal,
    )
    final = controller.run()
    if final.value != "complete":
        raise RuntimeError(f"iteration 2 ended in {final.value}")
    print(json.dumps({"state": final.value, "run_id": args.run_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
