#!/usr/bin/env python3
"""Bounded Colab replay of the local Auto Research (Ralphthon) loop.

PREPARED, NOT AUTHORIZED: the previous P2-Q4 staging authorization was
consumed by the completed 2026-07-12 run. Launching this task requires a new
explicit owner authorization for the exact commit being shipped
(docs/PHASE2_OPEN_QUESTIONS.md P2-Q4, D-P2-MERGE-RECONCILIATION).

Runs on the VM (driven by ``scripts/colab/colab_cli_run.sh
ralphthon_boundary_replay <COMMIT_SHA>``):

1. GPU attestation against ``configs/compute_policy.yaml`` (fail-closed;
   the loop itself is CPU-deterministic — the GPU attestation proves the
   remote environment path, not a scientific result).
2. The two-iteration synthetic demo loop, executed as one fresh controller
   instance per ``step()`` — every state boundary is an interruption +
   deterministic resume, checkpointed per round.
3. Independent trace grading on the VM.
4. A replay record plus a SHA-256 artifact manifest for hash-verified
   pull-back of the per-iteration artifacts.

Scientific status: staging_only. Synthetic fixtures only — no Cu
partitions, frozen test data, hidden labels, or claims. Exit codes: 0 pass,
1 fail, 3 policy denial, 4 no GPU (mirrors the staging task; no OOM path
because the loop allocates no GPU memory).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlip_research_agent.artifacts.registry import sha256_file  # noqa: E402
from mlip_research_agent.compute.attestation import (  # noqa: E402
    perform_attestation,
    probe_nvidia_smi,
    write_attestation,
)
from mlip_research_agent.compute.policy import Decision, load_policy  # noqa: E402
from mlip_research_agent.compute.types import (  # noqa: E402
    AcceleratorId,
    JobSpec,
    ProviderName,
    WorkloadClass,
)
from mlip_research_agent.research.auto_research import (  # noqa: E402
    TERMINAL_STATES,
    AutoResearchController,
    LocalDemoConfig,
    LoopState,
    MutationPolicy,
    SyntheticAggregateEvaluator,
    SyntheticQuadraticAdapter,
)
from mlip_research_agent.verification.trace_grader import grade_trace  # noqa: E402

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_POLICY_DENIED = 3
EXIT_NO_GPU = 4

SCHEMA_VERSION = "1.0.0"
TRACKED_PACKAGES = ("torch", "numpy", "pydantic", "mlip-research-agent")
MAX_STEPS = 200


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
        raise RuntimeError("replay requires an exact git commit; rev-parse failed")
    return commit


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def stage_attestation(run_dir: Path, run_id: str, commit: str) -> str:
    observation = probe_nvidia_smi()
    if observation is None:
        raise SystemExit(EXIT_NO_GPU)
    policy = load_policy(REPO_ROOT / "configs" / "compute_policy.yaml")
    spec = JobSpec(
        name="colab-ralphthon-boundary-replay",
        workload_class=WorkloadClass.GPU_SMOKE_TEST,
        requested_accelerator=AcceleratorId.NVIDIA_L4,
        command=["python", "scripts/colab/ralphthon_boundary_replay.py"],
        max_runtime_minutes=60,
        repo_commit=commit,
        dependency_hash=sha256_file(REPO_ROOT / "pyproject.toml"),
        mlip_backend="none",
        cuda_compat_class="cpu-loop",
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
    if record.policy_decision != Decision.ALLOW:
        raise SystemExit(EXIT_POLICY_DENIED)
    return observation.name


def stage_interrupted_loop(run_dir: Path, commit: str) -> dict[str, object]:
    demo = LocalDemoConfig.load(REPO_ROOT / "configs/research/ralphthon_local_demo.yaml")
    policy = MutationPolicy.load(REPO_ROOT / demo.mutation_policy_path)
    loop_dir = run_dir / "auto_research"

    def controller() -> AutoResearchController:
        return AutoResearchController(
            run_id="ralphthon-replay",
            run_dir=loop_dir,
            objective=demo.objective.sealed(),
            mutation_policy=policy,
            adapter=SyntheticQuadraticAdapter(),
            evaluator=SyntheticAggregateEvaluator(),
            acceptance=demo.acceptance,
            base_config=demo.base_config,
            fixture_seed=demo.fixture_seed,
            git_commit=commit,
        )

    # One fresh controller instance per step: every state boundary is an
    # interruption followed by a deterministic on-disk resume.
    states: list[str] = []
    for _ in range(MAX_STEPS):
        instance = controller()
        if instance.state.state in TERMINAL_STATES:
            break
        states.append(instance.state.state.value)
        instance.step()
    final = controller().state.state
    if final is not LoopState.COMPLETE:
        raise SystemExit(EXIT_FAIL)

    report = grade_trace(loop_dir, demo.acceptance)
    (loop_dir / "trace_grade.json").write_text(report.model_dump_json(indent=2) + "\n")
    if not report.passed:
        raise SystemExit(EXIT_FAIL)
    return {
        "terminal_state": final.value,
        "interruptions_survived": len(states),
        "iterations_graded": report.n_iterations_graded,
        "trace_grade": "pass",
    }


def stage_artifact_manifest(run_dir: Path) -> int:
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
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "files": entries}, indent=2, sort_keys=True)
        + "\n"
    )
    return len(entries)


def main() -> int:
    commit = _git_commit()
    run_id = f"ralphthon-replay-{commit[:8]}"
    run_dir = REPO_ROOT / "artifacts" / "colab_ralphthon" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "colab_ralphthon_boundary_replay",
        "run_id": run_id,
        "repo_commit": commit,
        "scientific_status": "staging_only",
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "authorization": "REQUIRED: new explicit owner authorization (P2-Q4 consumed)",
    }
    exit_code = EXIT_PASS
    try:
        record["gpu_name"] = stage_attestation(run_dir, run_id, commit)
        record["loop"] = stage_interrupted_loop(run_dir, commit)
        record["n_manifest_files"] = stage_artifact_manifest(run_dir)
        record["passed"] = True
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else EXIT_FAIL
        record["passed"] = False
        record["failure_exit_code"] = exit_code
    finally:
        record["wall_seconds"] = time.monotonic() - started
        (run_dir / "replay_record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        if exit_code == EXIT_PASS:
            stage_artifact_manifest(run_dir)
    print(f"TASK_EXIT:{exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
