#!/usr/bin/env bash
# Colab CLI launcher (owner decision 2026-07-12: the research agent drives all
# Colab execution through the security-reviewed google-colab-cli; notebooks
# are not an execution path).
#
# Ships the EXACT local commit to a fresh Colab GPU VM as a git bundle (no
# GitHub dependency), installs pinned dependencies, runs the requested task,
# pulls the artifact bundle back, verifies its SHA-256 manifest locally, and
# always releases the VM.
#
# Usage: colab_cli_run.sh <preflight|staging> <COMMIT_SHA> [GPU]
#   COMMIT_SHA  Exact commit reachable from HEAD (never a branch name).
#   GPU         Colab accelerator (default L4). For staging, A100 is valid
#               only as the single human-approved retry after a verified OOM
#               (task exit code 5); the policy attestation on the VM denies
#               every non-allowlisted device either way.
#
# Exit codes mirror the remote task's exit code (staging: 0 pass, 1 fail,
# 3 policy denial, 4 no GPU, 5 verified CUDA OOM); driver failures exit 2.
set -euo pipefail

TASK=${1:-}
COMMIT_SHA=${2:-}
GPU=${3:-L4}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_REPO=/content/MLP-auto-researcher

case "$TASK" in
  preflight)
    TASK_CMD="scripts/colab/preflight.py --output-root artifacts/colab"
    ARTIFACT_ROOT=artifacts/colab
    ;;
  staging)
    TASK_CMD="scripts/colab/staging_mace_boundary.py"
    ARTIFACT_ROOT=artifacts/colab_staging
    ;;
  *)
    echo "usage: colab_cli_run.sh <preflight|staging> <COMMIT_SHA> [GPU]" >&2
    exit 2
    ;;
esac

if ! [[ "$COMMIT_SHA" =~ ^[0-9a-f]{7,40}$ ]]; then
  echo "error: COMMIT_SHA must be an exact git commit SHA: '$COMMIT_SHA'" >&2
  exit 2
fi
FULL_SHA=$(git -C "$REPO_ROOT" rev-parse --verify "${COMMIT_SHA}^{commit}")
if ! git -C "$REPO_ROOT" merge-base --is-ancestor "$FULL_SHA" HEAD; then
  echo "error: $FULL_SHA is not reachable from HEAD; bundle would not contain it" >&2
  exit 2
fi

SESSION="${TASK:0:3}-${FULL_SHA:0:8}"
WORKDIR=$(mktemp -d)
PULLBACK_DIR="$REPO_ROOT/$ARTIFACT_ROOT/pullback-${FULL_SHA:0:8}"

cleanup() {
  colab stop -s "$SESSION" >/dev/null 2>&1 || true
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

# remote_py <timeout-seconds>: run stdin as Python in the session's kernel and
# require the STEP_OK sentinel, because `colab exec` does not propagate remote
# exceptions as exit codes. The CLI's kernel client occasionally hits a
# transient 10s HTTP read timeout while attaching, so one bounded retry is
# allowed for that transport failure only (kernel state persists; every
# snippet here is safe to resend).
remote_py() {
  local timeout=$1
  local code output status
  code=$(cat)
  for attempt in 1 2; do
    output=$(colab exec -s "$SESSION" --timeout "$timeout" <<<"$code" 2>&1)
    status=$?
    if [ $status -eq 0 ] && grep -q "STEP_OK" <<<"$output"; then
      echo "$output"
      return 0
    fi
    if grep -qE "ReadTimeout|ConnectionError|Read timed out" <<<"$output" && [ "$attempt" -eq 1 ]; then
      echo "warning: transient colab exec transport failure; retrying once" >&2
      sleep 15
      continue
    fi
    echo "$output" >&2
    echo "error: remote step failed (colab exec exit $status)" >&2
    return 2
  done
  return 2
}

echo "== bundling $FULL_SHA =="
git -C "$REPO_ROOT" bundle create "$WORKDIR/repo.bundle" HEAD >/dev/null

echo "== creating session $SESSION (GPU: $GPU) =="
colab new -s "$SESSION" --gpu "$GPU"

echo "== uploading commit bundle =="
colab upload -s "$SESSION" "$WORKDIR/repo.bundle" /content/repo.bundle

echo "== cloning exact commit on the VM =="
remote_py 300 <<PY
import subprocess
subprocess.run(["git", "clone", "/content/repo.bundle", "$REMOTE_REPO"], check=True)
subprocess.run(["git", "-C", "$REMOTE_REPO", "checkout", "--detach", "$FULL_SHA"], check=True)
head = subprocess.run(["git", "-C", "$REMOTE_REPO", "rev-parse", "HEAD"],
                      capture_output=True, text=True, check=True).stdout.strip()
assert head == "$FULL_SHA", f"commit mismatch: {head}"
print("STEP_OK", head)
PY

echo "== installing package + pinned dependencies =="
remote_py 1800 <<PY
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "$REMOTE_REPO"], check=True)
if "$TASK" == "staging":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    "$REMOTE_REPO/scripts/colab/staging_requirements.txt"], check=True)
print("STEP_OK installed")
PY

echo "== running $TASK on the VM =="
TASK_OUTPUT=$(remote_py 3600 <<PY
import subprocess, sys
proc = subprocess.run([sys.executable, *"$TASK_CMD".split()], cwd="$REMOTE_REPO")
print(f"TASK_EXIT={proc.returncode}")
print("STEP_OK ran")
PY
)
echo "$TASK_OUTPUT"
TASK_EXIT=$(grep -o "TASK_EXIT=[0-9]*" <<<"$TASK_OUTPUT" | head -1 | cut -d= -f2)
if [ -z "$TASK_EXIT" ]; then
  echo "error: could not determine remote task exit code" >&2
  exit 2
fi

echo "== zipping artifacts on the VM =="
remote_py 600 <<PY
import shutil
shutil.make_archive("/content/pullback", "zip", "$REMOTE_REPO/$ARTIFACT_ROOT")
print("STEP_OK zipped")
PY

echo "== pulling artifacts back =="
mkdir -p "$PULLBACK_DIR"
colab download -s "$SESSION" /content/pullback.zip "$WORKDIR/pullback.zip"
unzip -q -o "$WORKDIR/pullback.zip" -d "$PULLBACK_DIR"

echo "== releasing the VM =="
colab stop -s "$SESSION" || true

echo "== verifying pulled-back artifact hashes =="
python3 "$REPO_ROOT/scripts/colab/verify_pullback.py" "$PULLBACK_DIR"

echo "== done: task exit code $TASK_EXIT; artifacts under $PULLBACK_DIR =="
if [ "$TASK" = "staging" ] && [ "$TASK_EXIT" = "5" ]; then
  echo "verified CUDA OOM: the single authorized retry is" >&2
  echo "  colab_cli_run.sh staging $FULL_SHA A100   (human decision, once)" >&2
fi
exit "$TASK_EXIT"
