#!/usr/bin/env python
"""Standalone Colab GPU preflight battery.

Runs 11 bounded checks (GPU attestation/policy, imports, CUDA tensor op,
determinism, structure round trip, mock-gated MLIP/training stubs, checkpoint
round trip, artifact hashing, provenance generation, bounded failure
recovery) and writes their evidence under ``<output-root>/<run-id>/``. No
workflow executor is involved; this is a thin, dependency-light script meant
to run on a bare Colab VM right after ``pip install -e .``.

Exit codes: 0 pass, 1 fail, 3 policy denial (attestation deny), 4 GPU
required but absent (no --allow-cpu).
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.compute.attestation import (
    ATTESTATION_FILENAME,
    perform_attestation,
    probe_nvidia_smi,
    write_attestation,
)
from mlip_research_agent.compute.policy import Decision, load_policy
from mlip_research_agent.compute.preflight import PreflightRecord, PreflightStore
from mlip_research_agent.compute.schemas import (
    AcceleratorId,
    JobSpec,
    ProviderName,
    WorkloadClass,
)
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError

REPO_ROOT = Path(__file__).resolve().parents[2]
DIRECT_DEPENDENCIES = ("numpy", "ase", "pydantic", "PyYAML", "mlip-research-agent")


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail" | "skip"
    detail: str = ""


def _git_commit(repo_dir: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "0000000unknown"
    commit = proc.stdout.strip()
    if proc.returncode != 0 or not commit:
        return "0000000unknown"
    return commit


def _direct_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in DIRECT_DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _cuda_compat_class() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available() and torch.version.cuda:
        major = torch.version.cuda.split(".")[0]
        return f"cuda-{major}"
    return "cpu"


def _write_attestation_stub(
    run_dir: Path,
    *,
    run_id: str,
    requested_accelerator: AcceleratorId,
    repo_commit: str,
    reason: str,
) -> None:
    """Environment-style stub for the no-GPU paths; never a policy denial."""
    stub = {
        "schema_version": "0.1.0",
        "kind": "colab_preflight_no_gpu_stub",
        "run_id": run_id,
        "provider": ProviderName.COLAB.value,
        "workload_class": WorkloadClass.PREFLIGHT.value,
        "requested_accelerator": requested_accelerator.value,
        "observed_gpu_name": None,
        "policy_decision": None,
        "reason": reason,
        "repo_commit": repo_commit,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (run_dir / ATTESTATION_FILENAME).write_text(
        json.dumps(stub, indent=2, sort_keys=True) + "\n"
    )


def check_gpu_attestation_and_policy(
    *,
    run_dir: Path,
    run_id: str,
    config: dict[str, Any],
    allow_cpu: bool,
    repo_commit: str,
    policy_path: Path,
    config_path: Path,
    pyproject_path: Path,
) -> tuple[CheckResult, dict[str, bool]]:
    flags = {"policy_denied": False, "gpu_required_absent": False}
    requested_raw = str(config.get("requested_accelerator", "nvidia-l4"))
    try:
        requested_accelerator = AcceleratorId(requested_raw)
    except ValueError:
        requested_accelerator = AcceleratorId.NVIDIA_L4

    observation = probe_nvidia_smi()

    if observation is None:
        if allow_cpu:
            _write_attestation_stub(
                run_dir,
                run_id=run_id,
                requested_accelerator=requested_accelerator,
                repo_commit=repo_commit,
                reason="no GPU visible (nvidia-smi missing or returned no device); "
                "skipped under --allow-cpu, no deny attestation written",
            )
            return (
                CheckResult(
                    name="gpu_attestation_and_policy",
                    status="skip",
                    detail="no GPU visible; skipped under --allow-cpu",
                ),
                flags,
            )
        flags["gpu_required_absent"] = True
        _write_attestation_stub(
            run_dir,
            run_id=run_id,
            requested_accelerator=requested_accelerator,
            repo_commit=repo_commit,
            reason="no GPU visible (nvidia-smi missing or returned no device) and "
            "--allow-cpu was not set",
        )
        return (
            CheckResult(
                name="gpu_attestation_and_policy",
                status="fail",
                detail="GPU required but absent; rerun with --allow-cpu for a CPU-only run",
            ),
            flags,
        )

    policy = load_policy(policy_path)
    spec = JobSpec(
        name="colab-preflight",
        workload_class=WorkloadClass.PREFLIGHT,
        requested_accelerator=requested_accelerator,
        command=["true"],
        max_runtime_minutes=30,
        repo_commit=repo_commit,
    )
    environment_hash = sha256_file(pyproject_path) if pyproject_path.is_file() else ""
    config_hash = sha256_file(config_path) if config_path.is_file() else ""
    record = perform_attestation(
        policy,
        ProviderName.COLAB,
        spec,
        observation,
        run_id=run_id,
        environment_hash=environment_hash,
        config_hash=config_hash,
    )
    write_attestation(record, run_dir)
    (run_dir / ATTESTATION_FILENAME).write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )

    if record.policy_decision is Decision.DENY:
        flags["policy_denied"] = True
        return (
            CheckResult(
                name="gpu_attestation_and_policy",
                status="fail",
                detail=f"policy denied: {'; '.join(record.policy_reasons)}",
            ),
            flags,
        )
    return (
        CheckResult(
            name="gpu_attestation_and_policy",
            status="pass",
            detail=(
                f"observed={record.observed_gpu_name!r} "
                f"canonical={record.canonical_accelerator}"
            ),
        ),
        flags,
    )


def check_import_checks() -> CheckResult:
    modules = ["numpy", "ase", "pydantic", "yaml", "mlip_research_agent"]
    missing = [mod for mod in modules if not _can_import(mod)]
    if missing:
        return CheckResult(
            name="import_checks", status="fail", detail=f"missing modules: {', '.join(missing)}"
        )
    return CheckResult(
        name="import_checks", status="pass", detail=f"imported: {', '.join(modules)}"
    )


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def check_cuda_tensor_op() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult(name="cuda_tensor_op", status="skip", detail="torch is not installed")
    if not torch.cuda.is_available():
        return CheckResult(
            name="cuda_tensor_op",
            status="skip",
            detail="torch installed but no CUDA device visible",
        )
    a = torch.rand(64, 64, device="cuda")
    b = torch.rand(64, 64, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    return CheckResult(
        name="cuda_tensor_op",
        status="pass",
        detail=f"64x64 matmul on {torch.cuda.get_device_name(0)}, result shape {tuple(c.shape)}",
    )


def check_deterministic_seed() -> CheckResult:
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    numpy_match = bool(np.array_equal(rng_a.random(8), rng_b.random(8)))

    torch_note = "torch not installed"
    torch_match = True
    try:
        import torch

        gen_a = torch.Generator().manual_seed(42)
        gen_b = torch.Generator().manual_seed(42)
        torch_match = bool(
            torch.equal(torch.rand(8, generator=gen_a), torch.rand(8, generator=gen_b))
        )
        torch_note = "torch generators matched" if torch_match else "torch generators diverged"
    except ImportError:
        pass

    if numpy_match and torch_match:
        return CheckResult(
            name="deterministic_seed", status="pass", detail=f"numpy rngs matched; {torch_note}"
        )
    return CheckResult(
        name="deterministic_seed",
        status="fail",
        detail=f"numpy_match={numpy_match} torch_match={torch_match}",
    )


def check_tiny_structure_load(run_dir: Path) -> CheckResult:
    from ase.build import bulk

    from mlip_research_agent.skills.atomistics.structures_io import (
        StructureSet,
        atoms_to_record,
        record_to_atoms,
    )

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    record = atoms_to_record(atoms, index=0, perturbation_scale=0.0)
    structure_set = StructureSet(systems=[record])
    path = run_dir / "checks" / "tiny_structure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    structure_set.save(path)

    reloaded = StructureSet.load(path)
    reloaded_atoms = record_to_atoms(reloaded.systems[0])

    symbols_match = list(atoms.get_chemical_symbols()) == list(
        reloaded_atoms.get_chemical_symbols()
    )
    positions_match = bool(np.allclose(atoms.get_positions(), reloaded_atoms.get_positions()))
    n_atoms = len(atoms)

    if symbols_match and positions_match and n_atoms == 32:
        return CheckResult(
            name="tiny_structure_load",
            status="pass",
            detail=f"{n_atoms}-atom 2x2x2 fcc Cu round-tripped through StructureSet",
        )
    return CheckResult(
        name="tiny_structure_load",
        status="fail",
        detail=(
            f"round trip mismatch: symbols_match={symbols_match} "
            f"positions_match={positions_match}"
        ),
    )


def check_mlip_inference() -> CheckResult:
    return CheckResult(
        name="mlip_inference",
        status="skip",
        detail="no real MLIP backend installed (MACE gated on approval)",
    )


def check_train_step() -> CheckResult:
    return CheckResult(
        name="train_step",
        status="skip",
        detail="torch training path not yet integrated",
    )


def check_checkpoint_roundtrip(run_dir: Path) -> CheckResult:
    checks_dir = run_dir / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    payload = {"step": 7, "loss": 0.1234, "seed": 42}

    json_path = checks_dir / "checkpoint.json"
    json_path.write_text(json.dumps(payload, sort_keys=True))
    json_match = json.loads(json_path.read_text()) == payload

    torch_note = "torch not installed"
    torch_match = True
    try:
        import torch

        tensor_path = checks_dir / "checkpoint.pt"
        state = {"weight": torch.arange(8, dtype=torch.float32)}
        torch.save(state, tensor_path)
        loaded = torch.load(tensor_path)
        torch_match = bool(torch.equal(state["weight"], loaded["weight"]))
        torch_note = "torch state dict round-tripped"
    except ImportError:
        pass

    if json_match and torch_match:
        return CheckResult(
            name="checkpoint_roundtrip", status="pass", detail=f"json round-tripped; {torch_note}"
        )
    return CheckResult(
        name="checkpoint_roundtrip",
        status="fail",
        detail=f"json_match={json_match} torch_match={torch_match}",
    )


def check_artifact_hashing(run_dir: Path) -> CheckResult:
    checks_dir = run_dir / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    fixture = checks_dir / "hash_fixture.txt"
    fixture.write_text("mlip-research-agent colab preflight fixture\n")
    first = sha256_file(fixture)
    second = sha256_file(fixture)
    if first == second:
        return CheckResult(name="artifact_hashing", status="pass", detail=f"sha256={first}")
    return CheckResult(
        name="artifact_hashing",
        status="fail",
        detail=f"hash mismatch across repeated calls: {first} != {second}",
    )


def check_provenance_generation(run_dir: Path, repo_commit: str) -> CheckResult:
    checks_dir = run_dir / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    path = checks_dir / "provenance_roundtrip.json"
    payload = {
        "commit": repo_commit,
        "python_version": sys.version.split()[0],
        "package_versions": _direct_package_versions(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    reloaded = json.loads(path.read_text())
    if reloaded == payload:
        return CheckResult(
            name="provenance_generation",
            status="pass",
            detail="provenance stub round-tripped through JSON",
        )
    return CheckResult(
        name="provenance_generation", status="fail", detail="provenance stub round trip mismatch"
    )


def check_failure_recovery() -> CheckResult:
    repair_params = {"retry_count": 1, "note": "preflight-synthetic"}
    survived = False
    try:
        raise SkillError(
            "synthetic bounded failure for preflight verification",
            failure_class=FailureClass.TOOL_ERROR,
            severity=Severity.LOW,
            retryable=True,
            repair_params=repair_params,
        )
    except SkillError as exc:
        survived = (
            exc.repair_params == repair_params
            and exc.retryable
            and exc.failure_class is FailureClass.TOOL_ERROR
        )
    if survived:
        return CheckResult(
            name="failure_recovery",
            status="pass",
            detail="SkillError repair_params survived raise/catch, bounded",
        )
    return CheckResult(
        name="failure_recovery", status="fail", detail="repair_params did not survive raise/catch"
    )


def _conda_env_diff(repo_root: Path, versions: dict[str, str]) -> str:
    env_path = repo_root / "environment.yml"
    if not env_path.is_file():
        return "environment.yml not found; cannot diff"
    declared: list[str] = []
    in_dependencies = False
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "dependencies:":
            in_dependencies = True
            continue
        if not in_dependencies:
            continue
        if stripped and not stripped.startswith("- ") and not line.startswith((" ", "-")):
            # Left column of a new top-level YAML key ends the dependencies list.
            break
        if not stripped.startswith("- "):
            continue
        name = stripped[2:].split(">=")[0].split("<")[0].split("=")[0].strip()
        if name and name not in {"python", "pip"}:
            declared.append(name)
    tracked = {name.lower() for name, version in versions.items() if version != "not-installed"}
    known_dev_only = {"pytest", "pytest-cov", "ruff", "mypy", "types-pyyaml"}
    unresolved = sorted(
        name
        for name in declared
        if name.lower() not in tracked and name.lower() not in known_dev_only
    )
    if unresolved:
        return (
            f"environment.yml declares {len(declared)} packages; not resolvable via "
            f"importlib.metadata in this runtime: {', '.join(unresolved)}"
        )
    return (
        f"environment.yml declares {len(declared)} packages "
        f"({', '.join(declared)}); all tracked direct dependencies present in this runtime"
    )


def _build_environment_report(repo_root: Path, repo_commit: str) -> dict[str, Any]:
    pyproject_path = repo_root / "pyproject.toml"
    versions = _direct_package_versions()

    cuda_version: str | None = None
    try:
        import torch

        if torch.cuda.is_available():
            cuda_version = str(torch.version.cuda)
    except ImportError:
        pass

    return {
        "schema_version": "0.1.0",
        "repo_commit": repo_commit,
        "python_version": sys.version.split()[0],
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "installed_direct_dependencies": versions,
        "cuda_version": cuda_version,
        "dependency_spec_hash": sha256_file(pyproject_path) if pyproject_path.is_file() else "",
        "conda_env_diff": _conda_env_diff(repo_root, versions),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _write_artifact_manifest(run_dir: Path) -> Path:
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir).as_posix()
        if rel == "artifact-manifest.json":
            continue
        entries.append(
            {"path": rel, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    manifest_path = run_dir / "artifact-manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Colab GPU preflight: attestation, imports, determinism, "
        "and round-trip checks, no workflow executor required.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/colab"),
        help="Root directory; a fresh <run-id> subdirectory is created under it.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Treat an absent GPU as a skip instead of a failure.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Preflight config YAML (default: configs/colab/preflight.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    config_path = args.config or (REPO_ROOT / "configs" / "colab" / "preflight.yaml")
    config: dict[str, Any] = {}
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text())
        if isinstance(loaded, dict):
            config = loaded
    allow_cpu = bool(args.allow_cpu or config.get("allow_cpu", False))

    run_id = f"preflight-{uuid.uuid4().hex[:8]}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    repo_commit = _git_commit(REPO_ROOT)
    policy_path = REPO_ROOT / "configs" / "compute_policy.yaml"
    pyproject_path = REPO_ROOT / "pyproject.toml"

    results: list[CheckResult] = []

    result1, flags = check_gpu_attestation_and_policy(
        run_dir=run_dir,
        run_id=run_id,
        config=config,
        allow_cpu=allow_cpu,
        repo_commit=repo_commit,
        policy_path=policy_path,
        config_path=config_path,
        pyproject_path=pyproject_path,
    )
    results.append(result1)

    checks_cfg = config.get("checks", {})
    if not isinstance(checks_cfg, dict):
        checks_cfg = {}

    remaining_checks: list[tuple[str, Any]] = [
        ("import_checks", check_import_checks),
        ("cuda_tensor_op", check_cuda_tensor_op),
        ("deterministic_seed", check_deterministic_seed),
        ("tiny_structure_load", lambda: check_tiny_structure_load(run_dir)),
        ("mlip_inference", check_mlip_inference),
        ("train_step", check_train_step),
        ("checkpoint_roundtrip", lambda: check_checkpoint_roundtrip(run_dir)),
        ("artifact_hashing", lambda: check_artifact_hashing(run_dir)),
        ("provenance_generation", lambda: check_provenance_generation(run_dir, repo_commit)),
        ("failure_recovery", check_failure_recovery),
    ]
    for name, func in remaining_checks:
        cfg_value = checks_cfg.get(name, True)
        if cfg_value is False or cfg_value == "skip":
            results.append(
                CheckResult(name=name, status="skip", detail=f"disabled via config ({cfg_value!r})")
            )
            continue
        results.append(func())

    overall_passed = not any(r.status == "fail" for r in results)
    if flags["policy_denied"]:
        exit_code = 3
    elif flags["gpu_required_absent"]:
        exit_code = 4
    elif not overall_passed:
        exit_code = 1
    else:
        exit_code = 0

    finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    wall_time_seconds = time.monotonic() - started

    log_lines = [f"{r.name}: {r.status} - {r.detail}" for r in results]
    (run_dir / "logs" / "preflight.log").write_text("\n".join(log_lines) + "\n")

    environment_report = _build_environment_report(REPO_ROOT, repo_commit)
    (run_dir / "environment.json").write_text(
        json.dumps(environment_report, indent=2, sort_keys=True) + "\n"
    )

    test_report = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "checks": [asdict(r) for r in results],
        "passed": overall_passed,
        "generated_at": finished_at,
    }
    (run_dir / "test-report.json").write_text(
        json.dumps(test_report, indent=2, sort_keys=True) + "\n"
    )

    counts = {
        "pass": sum(1 for r in results if r.status == "pass"),
        "fail": sum(1 for r in results if r.status == "fail"),
        "skip": sum(1 for r in results if r.status == "skip"),
        "total": len(results),
    }
    metrics = {
        "run_id": run_id,
        "counts": counts,
        "wall_time_seconds": wall_time_seconds,
        "generated_at": finished_at,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    provenance = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "repo_commit": repo_commit,
        "command": [sys.executable, *sys.argv],
        "started_at": started_at,
        "finished_at": finished_at,
        "passed": overall_passed,
        "exit_code": exit_code,
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    dependency_hash = sha256_file(pyproject_path) if pyproject_path.is_file() else ""
    store = PreflightStore(directory=args.output_root / "records")
    preflight_record = PreflightRecord(
        run_id=run_id,
        passed=overall_passed,
        repo_commit=repo_commit,
        dependency_hash=dependency_hash,
        mlip_backend="none",
        cuda_compat_class=_cuda_compat_class(),
        workflow_schema_version="0.1.0",
        created_at=finished_at,
    )
    store.save(preflight_record)

    _write_artifact_manifest(run_dir)

    print(f"preflight run {run_id}: {'PASS' if overall_passed else 'FAIL'} (exit={exit_code})")
    for result in results:
        print(f"  {result.name}: {result.status} - {result.detail}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
