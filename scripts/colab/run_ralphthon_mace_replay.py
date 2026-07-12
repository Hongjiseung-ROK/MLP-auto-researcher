#!/usr/bin/env python
"""Host owner for one exact-commit, L4-only, pause/review/resume replay."""

# Remote snippets are intentionally kept as literal, reviewable command blocks.
# ruff: noqa: E501, UP031

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.data.bounded_view import (
    EXPECTED_BOUNDED_VIEW_FILE_SHA256,
    BoundedLabelView,
    verify_bounded_view_membership,
)
from mlip_research_agent.data.split import SplitManifest
from mlip_research_agent.research.auto_research.review import build_review_packet
from mlip_research_agent.verification.trace_grader import (
    grade_remote_infrastructure_trace,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class ReplayContractError(ValueError):
    pass


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def validate_replay_contract(
    *, commit: str, gpu: str, n_gpus: int, max_runtime_minutes: int
) -> None:
    if not FULL_SHA.fullmatch(commit):
        raise ReplayContractError("commit must be a full 40-character lowercase SHA")
    if _git("rev-parse", "HEAD") != commit:
        raise ReplayContractError("commit must equal the clean local HEAD")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise ReplayContractError("replay requires a completely clean worktree")
    if gpu != "L4":
        raise ReplayContractError("ralphthon_mace_replay permits only NVIDIA L4")
    if n_gpus != 1:
        raise ReplayContractError("ralphthon_mace_replay requires exactly one GPU")
    if not 1 <= max_runtime_minutes <= 60:
        raise ReplayContractError("total session runtime must be between 1 and 60 minutes")


def _remaining(deadline: float) -> int:
    seconds = int(deadline - time.monotonic())
    if seconds <= 0:
        raise TimeoutError("total Colab replay deadline exhausted")
    return seconds


def _run(command: list[str], *, timeout: int, input_text: str | None = None) -> str:
    proc = subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(command)}\n{proc.stderr}"
        )
    return proc.stdout


def _remote_python(session: str, code: str, *, deadline: float) -> str:
    """Execute once. Scientific commands are deliberately never resent."""
    output = _run(
        ["colab", "exec", "-s", session, "--timeout", str(_remaining(deadline))],
        timeout=_remaining(deadline),
        input_text=code,
    )
    if "REMOTE_STEP_OK" not in output:
        raise RuntimeError(f"remote command returned without success sentinel:\n{output}")
    return output


def _strict_manifest(root: Path) -> Path:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "trace_grade.json"}:
            entries.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest = root / "artifact_manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": "1.0.0", "files": entries}, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def verify_strict_manifest(root: Path) -> None:
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    expected = {entry["relative_path"]: entry for entry in manifest["files"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "trace_grade.json"}
    }
    if actual != set(expected):
        raise RuntimeError("pull-back contains missing or unmanifested files")
    for relative, entry in expected.items():
        if sha256_file(root / relative) != entry["sha256"]:
            raise RuntimeError(f"pull-back hash mismatch: {relative}")


def verify_registered_manifest(root: Path) -> None:
    """Verify every controller-registered artifact in an intermediate pull-back."""
    entries = json.loads((root / "manifest.json").read_text())
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("iteration-1 controller manifest is missing or empty")
    for entry in entries:
        path = root / entry["relative_path"]
        if not path.is_file():
            raise RuntimeError(f"registered iteration-1 artifact is missing: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"iteration-1 artifact hash mismatch: {path}")


def _confirm_stopped(session: str, *, deadline: float) -> str:
    absent_polls = 0
    while True:
        sessions = _run(["colab", "sessions"], timeout=min(30, _remaining(deadline)))
        if session not in sessions:
            absent_polls += 1
            if absent_polls >= 3:
                return "confirmed_absent_stable"
        else:
            absent_polls = 0
        if _remaining(deadline) <= 5:
            raise RuntimeError(f"Colab session {session} remains active")
        time.sleep(2)


def run_replay(args: argparse.Namespace) -> Path:
    validate_replay_contract(
        commit=args.commit,
        gpu=args.gpu,
        n_gpus=args.n_gpus,
        max_runtime_minutes=args.max_runtime_minutes,
    )
    if not args.label_view.is_file():
        raise ReplayContractError("bounded label view is missing")
    if not SHA256_HEX.fullmatch(args.label_view_sha256):
        raise ReplayContractError("bounded label-view SHA-256 must be 64 lowercase hex characters")
    if args.label_view_sha256 != EXPECTED_BOUNDED_VIEW_FILE_SHA256:
        raise ReplayContractError("bounded label-view SHA-256 is not the authorized projection")
    if sha256_file(args.label_view) != args.label_view_sha256:
        raise ReplayContractError("bounded label-view SHA-256 mismatch")
    view = BoundedLabelView.load(args.label_view)
    split = SplitManifest.load(REPO_ROOT / "data_registry/datasets/cu_phase2/split_manifest.json")
    verify_bounded_view_membership(view, split)
    if args.output.exists():
        raise ReplayContractError("pull-back destination must not already exist")
    if args.review_bundle.exists():
        raise ReplayContractError("review bundle must not preexist the iteration-1 pause")
    session_started = time.monotonic()
    hard_deadline = session_started + args.max_runtime_minutes * 60
    cleanup_reserve_seconds = min(120, max(15, args.max_runtime_minutes * 15))
    deadline = hard_deadline - cleanup_reserve_seconds
    session = f"ral-{args.commit[:8]}"
    run_id = f"ralphthon-mace-{args.commit[:8]}"
    cleanup_attempt_required = False
    cleanup_state = "not_allocated"
    temporary_output = args.output.with_name(f".{args.output.name}.staging")
    if temporary_output.exists():
        raise ReplayContractError("stale pull-back staging directory exists")
    with tempfile.TemporaryDirectory(prefix="ralphthon-mace-") as temporary:
        tmp = Path(temporary)
        try:
            _run(
                ["git", "-C", str(REPO_ROOT), "bundle", "create", str(tmp / "repo.bundle"), "HEAD"],
                timeout=60,
            )
            sessions_before = _run(["colab", "sessions"], timeout=30)
            if session in sessions_before:
                raise ReplayContractError(f"deterministic session already exists: {session}")
            cleanup_attempt_required = True
            _run(["colab", "new", "-s", session, "--gpu", "L4"], timeout=_remaining(deadline))
            _run(
                [
                    "colab",
                    "upload",
                    "-s",
                    session,
                    str(tmp / "repo.bundle"),
                    "/content/repo.bundle",
                ],
                timeout=_remaining(deadline),
            )
            _run(
                [
                    "colab",
                    "upload",
                    "-s",
                    session,
                    str(args.label_view),
                    "/content/replay_label_view.json",
                ],
                timeout=_remaining(deadline),
            )
            _remote_python(
                session,
                """import subprocess, sys
subprocess.run(['git','clone','/content/repo.bundle','/content/MLP-auto-researcher'], check=True)
subprocess.run(['git','-C','/content/MLP-auto-researcher','checkout','--detach','%s'], check=True)
head=subprocess.check_output(['git','-C','/content/MLP-auto-researcher','rev-parse','HEAD'], text=True).strip()
assert head == '%s'
subprocess.run([sys.executable,'-m','pip','install','-q','-e','/content/MLP-auto-researcher'], check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r','/content/MLP-auto-researcher/scripts/colab/ralphthon_mace_replay_requirements.txt'], check=True)
print('REMOTE_STEP_OK')
"""
                % (args.commit, args.commit),
                deadline=deadline,
            )
            # No resend: a timeout here is an ambiguous scientific execution and fails closed.
            _remote_python(
                session,
                """import subprocess, sys
proc=subprocess.run([sys.executable,'scripts/research/run_mace_auto_research.py','iteration1','--commit','%s','--run-id','%s','--label-view','/content/replay_label_view.json','--label-view-sha256','%s'], cwd='/content/MLP-auto-researcher')
assert proc.returncode == 0
print('REMOTE_STEP_OK')
"""
                % (args.commit, run_id, args.label_view_sha256),
                deadline=deadline,
            )
            _remote_python(
                session,
                """import shutil
shutil.make_archive('/content/iteration1','zip','/content/MLP-auto-researcher/artifacts/auto_research/%s')
print('REMOTE_STEP_OK')
"""
                % run_id,
                deadline=deadline,
            )
            _run(
                [
                    "colab",
                    "download",
                    "-s",
                    session,
                    "/content/iteration1.zip",
                    str(tmp / "iteration1.zip"),
                ],
                timeout=_remaining(deadline),
            )
            iteration1_dir = args.output.with_name(f"{args.output.name}-iteration1")
            if iteration1_dir.exists():
                raise ReplayContractError("iteration-1 pull-back destination already exists")
            with zipfile.ZipFile(tmp / "iteration1.zip") as archive:
                archive.extractall(iteration1_dir)
            verify_registered_manifest(iteration1_dir)
            packet = build_review_packet(iteration1_dir)
            packet_path = iteration1_dir / "review_packet.json"
            packet_path.write_text(packet.canonical_text())
            print(f"ITERATION1_READY={iteration1_dir}", flush=True)
            print(f"REVIEW_PACKET={packet_path}", flush=True)
            ready = args.review_bundle / "READY"
            while not ready.is_file():
                if _remaining(deadline) <= 60:
                    raise TimeoutError("review bundle was not sealed before the session deadline")
                time.sleep(2)
            shutil.make_archive(str(tmp / "reviews"), "zip", args.review_bundle)
            _run(
                [
                    "colab",
                    "upload",
                    "-s",
                    session,
                    str(tmp / "reviews.zip"),
                    "/content/reviews.zip",
                ],
                timeout=_remaining(deadline),
            )
            _remote_python(
                session,
                """import shutil, subprocess, sys
shutil.unpack_archive('/content/reviews.zip','/content/reviews')
proc=subprocess.run([sys.executable,'scripts/research/run_mace_auto_research.py','iteration2','--commit','%s','--run-id','%s','--label-view','/content/replay_label_view.json','--label-view-sha256','%s','--review-bundle','/content/reviews'], cwd='/content/MLP-auto-researcher')
assert proc.returncode == 0
shutil.make_archive('/content/final','zip','/content/MLP-auto-researcher/artifacts/auto_research/%s')
print('REMOTE_STEP_OK')
"""
                % (args.commit, run_id, args.label_view_sha256, run_id),
                deadline=deadline,
            )
            _run(
                ["colab", "download", "-s", session, "/content/final.zip", str(tmp / "final.zip")],
                timeout=_remaining(deadline),
            )
            temporary_output.mkdir(parents=True)
            with zipfile.ZipFile(tmp / "final.zip") as archive:
                archive.extractall(temporary_output)
        finally:
            if cleanup_attempt_required:
                stop_error = ""
                try:
                    stop = subprocess.run(
                        ["colab", "stop", "-s", session],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    stop_error = stop.stderr if stop.returncode != 0 else ""
                except subprocess.TimeoutExpired as exc:
                    stop_error = f"stop command timed out: {exc}"
                cleanup_state = _confirm_stopped(session, deadline=hard_deadline)
                if stop_error and cleanup_state != "confirmed_absent_stable":
                    raise RuntimeError(f"failed to stop Colab session: {stop_error}")
                print(f"COLAB_CLEANUP={cleanup_state}:{session}", flush=True)
        (temporary_output / "cleanup_confirmation.json").write_text(
            json.dumps(
                {
                    "session": session,
                    "status": cleanup_state,
                    "session_elapsed_seconds": time.monotonic() - session_started,
                    "maximum_session_seconds": args.max_runtime_minutes * 60,
                    "scientific_status": "infrastructure_only",
                    "claim_eligible": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _strict_manifest(temporary_output)
        verify_strict_manifest(temporary_output)
        temporary_output.replace(args.output)
        grade = grade_remote_infrastructure_trace(
            args.output,
            exact_commit=args.commit,
            label_view_sha256=args.label_view_sha256,
        )
        (args.output / "trace_grade.json").write_text(
            grade.model_dump_json(indent=2) + "\n"
        )
        if not grade.passed:
            raise RuntimeError(
                "pulled-back remote trace failed independent host grading; "
                f"evidence retained at {args.output}"
            )
    return args.output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--max-runtime-minutes", type=int, default=60)
    parser.add_argument("--label-view", type=Path, required=True)
    parser.add_argument("--label-view-sha256", required=True)
    parser.add_argument("--review-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validate_replay_contract(
        commit=args.commit,
        gpu=args.gpu,
        n_gpus=args.n_gpus,
        max_runtime_minutes=args.max_runtime_minutes,
    )
    if args.validate_only:
        return 0
    print(run_replay(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
