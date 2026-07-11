#!/usr/bin/env python
"""Bounded Colab staging for the real-MACE GPU boundary (P2-Q4).

Authorized scope (docs/PHASE2_OPEN_QUESTIONS.md P2-Q4): one GPU, L4 first,
at most 60 minutes, no full pilot. A single A100-40GB retry is allowed only
after this script exits with the dedicated OOM code, and that retry is a
human decision — the script never re-submits anything itself.

Stages, all mandatory and ordered:
  1. GPU attestation against ``configs/compute_policy.yaml`` (unknown devices
     denied; there is deliberately no CPU fallback);
  2. pinned checkpoint resolution — download from the manifest's source URL
     only when absent, then exact filename/size/SHA-256 verification;
  3. one real MACE inference batch through ``mace_inference`` on CUDA plus the
     same batch on CPU, with parity asserted within declared tolerances;
  4. one real optimizer step through the controlled fine-tune runner on CUDA
     (parameters must change);
  5. checkpoint save + resume round-trip (a second optimizer step continuing
     from the saved state);
  6. an artifact manifest (SHA-256 of every produced file) so the whole run
     directory can be pulled back and verified locally.

Exit codes: 0 pass, 1 fail, 3 policy denial, 4 GPU absent, 5 verified CUDA
OOM (the only condition under which the A100-40GB retry is authorized).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from ase.build import bulk

from mlip_research_agent.artifacts.registry import ArtifactRegistry, sha256_file
from mlip_research_agent.compute.attestation import (
    perform_attestation,
    probe_nvidia_smi,
    write_attestation,
)
from mlip_research_agent.compute.policy import Decision, load_policy
from mlip_research_agent.compute.schemas import (
    AcceleratorId,
    JobSpec,
    ProviderName,
    WorkloadClass,
)
from mlip_research_agent.data.manifests import LabeledConfiguration
from mlip_research_agent.skills.atomistics.structures_io import (
    StructureSet,
    atoms_to_record,
)
from mlip_research_agent.skills.base import SkillContext
from mlip_research_agent.skills.mlip.mace_finetune.runner import run_controlled_training
from mlip_research_agent.skills.mlip.mace_inference.checkpoint import (
    MACECheckpointManifest,
    resolve_checkpoint,
)
from mlip_research_agent.skills.mlip.mace_inference.implementation import MACEInferenceSkill
from mlip_research_agent.skills.mlip.mace_inference.schema import (
    MACEInferenceInput,
    MACEInferenceOutput,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "1.0.0"
STAGING_RECORD = "staging_record.json"

# Declared CPU/GPU parity tolerances for float64 inference on the small
# fixture batch. Bitwise equality across devices is not promised; anything
# beyond these bounds is a staging failure, not a tolerance to widen.
ENERGY_PARITY_ATOL_EV = 5e-5
FORCE_PARITY_ATOL_EV_PER_A = 5e-5

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_POLICY_DENIED = 3
EXIT_NO_GPU = 4
EXIT_CUDA_OOM = 5

TRACKED_PACKAGES = (
    "mlip-research-agent",
    "numpy",
    "ase",
    "pydantic",
    "torch",
    "mace-torch",
    "e3nn",
)


class StagingOom(RuntimeError):
    """Verified CUDA out-of-memory, the only A100-40GB retry trigger."""


@dataclass
class StageResult:
    name: str
    status: str  # "pass" | "fail"
    seconds: float
    detail: str = ""


class WallBudget:
    def __init__(self, max_wall_seconds: int) -> None:
        self.started = time.monotonic()
        self.max_wall_seconds = max_wall_seconds

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def check(self, stage: str) -> None:
        if self.elapsed() > self.max_wall_seconds:
            raise RuntimeError(
                f"staging wall budget exceeded before stage {stage!r}: "
                f"{self.elapsed():.0f}s > {self.max_wall_seconds}s"
            )


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    commit = proc.stdout.strip()
    if proc.returncode != 0 or not commit:
        raise RuntimeError("staging requires an exact git commit; rev-parse failed")
    return commit


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _cuda_compat_class() -> str:
    import torch

    if torch.cuda.is_available() and torch.version.cuda:
        return f"cuda-{torch.version.cuda.split('.')[0]}"
    return "cpu"


def stage_attestation(
    run_dir: Path,
    run_id: str,
    commit: str,
    requested: AcceleratorId,
) -> tuple[str, JobSpec]:
    observation = probe_nvidia_smi()
    if observation is None:
        raise SystemExit(EXIT_NO_GPU)
    policy = load_policy(REPO_ROOT / "configs" / "compute_policy.yaml")
    spec = JobSpec(
        name="colab-staging-mace-boundary",
        workload_class=WorkloadClass.GPU_SMOKE_TEST,
        requested_accelerator=requested,
        command=["python", "scripts/colab/staging_mace_boundary.py"],
        max_runtime_minutes=60,
        repo_commit=commit,
        dependency_hash=sha256_file(REPO_ROOT / "pyproject.toml"),
        mlip_backend="mace",
        cuda_compat_class=_cuda_compat_class(),
    )
    record = perform_attestation(
        policy,
        ProviderName.COLAB,
        spec,
        observation,
        run_id=run_id,
        environment_hash=sha256_file(REPO_ROOT / "pyproject.toml"),
        config_hash=sha256_file(REPO_ROOT / "configs" / "compute_policy.yaml"),
    )
    write_attestation(record, run_dir)
    if record.policy_decision.decision != Decision.ALLOW:
        raise SystemExit(EXIT_POLICY_DENIED)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(EXIT_NO_GPU)
    return f"{observation.name} (torch: {torch.cuda.get_device_name(0)})", spec


def stage_checkpoint(manifest_path: Path, cache_dir: Path) -> Path:
    manifest = MACECheckpointManifest.load(manifest_path)
    checkpoint_path = cache_dir / manifest.checkpoint_filename
    if not checkpoint_path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".download")
        urllib.request.urlretrieve(manifest.source_url, temporary)
        os.replace(temporary, checkpoint_path)
    resolve_checkpoint(manifest, checkpoint_path)
    return checkpoint_path


def _inference(
    run_dir: Path,
    device: str,
    manifest_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    inference_root = run_dir / f"inference-{device}"
    inference_root.mkdir(parents=True)
    base = bulk("Cu", "fcc", a=3.6, cubic=True)
    rattled = base.copy()
    rng = np.random.default_rng(11)
    rattled.positions = rattled.positions + rng.normal(0.0, 0.02, rattled.positions.shape)
    structures = StructureSet(
        systems=[atoms_to_record(base, 0, 0.0), atoms_to_record(rattled, 1, 0.02)]
    )
    inputs_dir = inference_root / "inputs"
    inputs_dir.mkdir()
    structures.save(inputs_dir / "structures.json")
    ctx = SkillContext(
        run_dir=inference_root,
        step_id="staging-inference",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(inference_root),
    )
    output = MACEInferenceSkill().run(
        MACEInferenceInput(
            structures_path="inputs/structures.json",
            checkpoint_manifest_path=str(manifest_path),
            checkpoint_path=str(checkpoint_path),
            dataset_id="colab-staging-fixture",
            dataset_content_sha256="d" * 64,
            split_semantic_sha256="e" * 64,
            record_ids=["base", "rattled"],
            device="cuda" if device == "cuda" else "cpu",
            default_dtype="float64",
        ),
        ctx,
    )
    assert isinstance(output, MACEInferenceOutput)
    payload: dict[str, Any] = json.loads(
        (inference_root / output.predictions_path).read_text()
    )
    return payload


def stage_inference_parity(
    run_dir: Path, manifest_path: Path, checkpoint_path: Path
) -> str:
    gpu = _inference(run_dir, "cuda", manifest_path, checkpoint_path)
    cpu = _inference(run_dir, "cpu", manifest_path, checkpoint_path)
    max_energy_delta = 0.0
    max_force_delta = 0.0
    for gpu_record, cpu_record in zip(
        gpu["predictions"], cpu["predictions"], strict=True
    ):
        max_energy_delta = max(
            max_energy_delta,
            abs(float(gpu_record["energy_ev"]) - float(cpu_record["energy_ev"])),
        )
        gpu_forces = np.asarray(gpu_record["forces_ev_per_a"], dtype=np.float64)
        cpu_forces = np.asarray(cpu_record["forces_ev_per_a"], dtype=np.float64)
        max_force_delta = max(max_force_delta, float(np.abs(gpu_forces - cpu_forces).max()))
    detail = (
        f"max |E_gpu - E_cpu| = {max_energy_delta:.3e} eV, "
        f"max |F_gpu - F_cpu| = {max_force_delta:.3e} eV/A"
    )
    if max_energy_delta > ENERGY_PARITY_ATOL_EV or max_force_delta > FORCE_PARITY_ATOL_EV_PER_A:
        raise RuntimeError(f"CPU/GPU parity outside declared tolerance: {detail}")
    return detail


def _training_records() -> list[LabeledConfiguration]:
    records: list[LabeledConfiguration] = []
    for index in range(5):
        atoms = bulk("Cu", "fcc", a=3.58 + index * 0.02, cubic=True)
        records.append(
            LabeledConfiguration(
                config_id=f"staging-fixture-{index:02d}",
                source_id=f"synthetic[{index}]",
                top_group="staging-fixture",
                group_id="staging-fixture",
                split_unit_id=f"staging-unit-{index:02d}",
                symbols=["Cu"] * len(atoms),
                positions=[[float(x) for x in row] for row in atoms.positions],
                cell=[[float(x) for x in row] for row in atoms.cell[:]],
                pbc=[True, True, True],
                energy_ev=-14.5 - index * 0.05,
                forces_ev_per_a=[[0.0, 0.0, 0.0] for _ in range(len(atoms))],
                virial_stress_kbar=None,
                level_of_theory="synthetic-staging",
                source_record_sha256=f"{index + 300:064x}",
            )
        )
    return records


def stage_optimizer_step_and_round_trip(run_dir: Path, checkpoint_path: Path) -> str:
    records = _training_records()
    train_records, validation_records = records[:3], records[3:]
    training_dir = run_dir / "training"
    training_dir.mkdir()
    common: dict[str, Any] = {
        "foundation_path": checkpoint_path,
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
        "device_name": "cuda",
        "default_dtype": "float64",
        "max_wall_seconds": 900,
    }
    controlled_checkpoint = training_dir / "controlled-checkpoint.pt"
    first = run_controlled_training(
        checkpoint_path=controlled_checkpoint,
        model_path=training_dir / "step-one.model",
        resume_checkpoint_path=None,
        max_epochs=1,
        max_optimizer_steps=1,
        **common,
    )
    if first.optimizer_steps != 1 or first.changed_parameter_tensors <= 0:
        raise RuntimeError("GPU optimizer step did not change trainable parameters")
    resumed = run_controlled_training(
        checkpoint_path=training_dir / "controlled-checkpoint-resumed.pt",
        model_path=training_dir / "step-two.model",
        resume_checkpoint_path=controlled_checkpoint,
        max_epochs=2,
        max_optimizer_steps=2,
        **common,
    )
    if resumed.completed_epochs != 2 or resumed.optimizer_steps != 2:
        raise RuntimeError("checkpoint round-trip did not resume at the epoch boundary")
    if resumed.records[0] != first.records[0]:
        raise RuntimeError("resumed run does not preserve the first epoch's metric record")
    return (
        f"step-1 validation force MAE {first.best_validation_force_mae_ev_per_a:.6f} eV/A, "
        f"changed tensors {first.changed_parameter_tensors}; resume reached epoch "
        f"{resumed.completed_epochs} with preserved epoch-0 record"
    )


def stage_artifact_manifest(run_dir: Path) -> str:
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        entries.append(
            {
                "relative_path": path.relative_to(run_dir).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest_path = run_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "files": entries}, indent=2, sort_keys=True)
        + "\n"
    )
    return f"{len(entries)} files hashed for pull-back"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts/colab_staging")
    parser.add_argument(
        "--requested-accelerator",
        choices=[AcceleratorId.NVIDIA_L4.value, AcceleratorId.NVIDIA_A100_40GB.value],
        default=AcceleratorId.NVIDIA_L4.value,
        help="A100-40GB is valid only as the single human-approved OOM retry",
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=REPO_ROOT / "configs/models/mace_mp_0_small_candidate.json",
    )
    parser.add_argument(
        "--checkpoint-cache", type=Path, default=REPO_ROOT / "data/cache/models"
    )
    parser.add_argument("--max-wall-seconds", type=int, default=3300)
    args = parser.parse_args()

    run_id = f"staging-{uuid.uuid4().hex[:8]}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True)
    budget = WallBudget(args.max_wall_seconds)
    commit = _git_commit()
    stages: list[StageResult] = []
    exit_code = EXIT_PASS
    gpu_name = None
    failure: dict[str, str] | None = None

    _write_text(run_dir / "pip_freeze.txt", _freeze())
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        _write_text(run_dir / "nvidia_smi.txt", smi.stdout + smi.stderr)
    except OSError as exc:
        _write_text(run_dir / "nvidia_smi.txt", f"nvidia-smi unavailable: {exc}\n")

    checkpoint_holder: dict[str, Path] = {}
    stage_plan: list[tuple[str, Any]] = [
        (
            "gpu_attestation_policy",
            lambda: stage_attestation(
                run_dir, run_id, commit, AcceleratorId(args.requested_accelerator)
            ),
        ),
        (
            "pinned_checkpoint_resolution",
            lambda: stage_checkpoint(args.checkpoint_manifest, args.checkpoint_cache),
        ),
        (
            "real_inference_cpu_gpu_parity",
            lambda: stage_inference_parity(
                run_dir, args.checkpoint_manifest, checkpoint_holder["path"]
            ),
        ),
        (
            "one_optimizer_step_and_checkpoint_round_trip",
            lambda: stage_optimizer_step_and_round_trip(
                run_dir, checkpoint_holder["path"]
            ),
        ),
        ("artifact_manifest", lambda: stage_artifact_manifest(run_dir)),
    ]
    for name, runner in stage_plan:
        budget.check(name)
        started = time.monotonic()
        try:
            outcome = runner()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else EXIT_FAIL
            stages.append(
                StageResult(name, "fail", time.monotonic() - started, f"exit {code}")
            )
            exit_code = code
            break
        except BaseException as exc:
            oom = _is_oom(exc)
            stages.append(
                StageResult(
                    name,
                    "fail",
                    time.monotonic() - started,
                    f"{type(exc).__name__}: {exc}"[:2000],
                )
            )
            failure = {"stage": name, "error_type": type(exc).__name__}
            exit_code = EXIT_CUDA_OOM if oom else EXIT_FAIL
            break
        detail = ""
        if name == "gpu_attestation_policy":
            gpu_name, _ = outcome
            detail = str(gpu_name)
        elif name == "pinned_checkpoint_resolution":
            checkpoint_holder["path"] = outcome
            detail = str(outcome)
        elif isinstance(outcome, str):
            detail = outcome
        stages.append(StageResult(name, "pass", time.monotonic() - started, detail))

    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "colab_staging_mace_boundary",
        "scientific_status": "staging_only",
        "authorization": "P2-Q4 bounded staging (one GPU, <=60 min, no full pilot)",
        "run_id": run_id,
        "repo_commit": commit,
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "gpu_name": gpu_name,
        "requested_accelerator": args.requested_accelerator,
        "parity_tolerances": {
            "energy_atol_ev": ENERGY_PARITY_ATOL_EV,
            "force_atol_ev_per_a": FORCE_PARITY_ATOL_EV_PER_A,
        },
        "wall_seconds": budget.elapsed(),
        "stages": [asdict(stage) for stage in stages],
        "failure": failure,
        "exit_code": exit_code,
        "passed": exit_code == EXIT_PASS,
    }
    (run_dir / STAGING_RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return exit_code


def _freeze() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


if __name__ == "__main__":
    raise SystemExit(main())
