#!/usr/bin/env bash
# Colab bootstrap: clone/checkout an exact commit, install the package, run
# the preflight battery.
#
# Usage: bootstrap.sh <COMMIT_SHA> [TASK]
#   COMMIT_SHA  Full or short git commit SHA to test. Never a branch name
#               ("main"/"HEAD"/etc) — the whole point of preflight is testing
#               an exact, reproducible commit, not a moving ref.
#   TASK        Optional label recorded in the summary line (default: preflight).
#
# environment.yml remains the authoritative dependency spec for local dev
# machines (`conda env create -f environment.yml`); this script does the
# minimal `pip install -e .` needed to exercise the package on a bare Colab
# VM, where conda is not set up. No conda commands run here.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/Hongjiseung-ROK/MLP-auto-researcher.git}
REPO_DIR=/content/MLP-auto-researcher

if [ "$#" -lt 1 ]; then
  echo "usage: bootstrap.sh <COMMIT_SHA> [TASK]" >&2
  exit 2
fi

COMMIT_SHA=$1
TASK=${2:-preflight}

case "$COMMIT_SHA" in
  main | master | HEAD | origin/* | refs/*)
    echo "error: COMMIT_SHA must be an exact commit SHA, not a branch/ref name: '$COMMIT_SHA'" >&2
    exit 2
    ;;
esac
if ! [[ "$COMMIT_SHA" =~ ^[0-9a-f]{7,40}$ ]]; then
  echo "error: COMMIT_SHA does not look like a git commit SHA: '$COMMIT_SHA'" >&2
  exit 2
fi

if [ -d "$REPO_DIR/.git" ]; then
  echo "== repo present at $REPO_DIR, fetching =="
  git -C "$REPO_DIR" fetch --all --tags
else
  echo "== cloning $REPO_URL into $REPO_DIR =="
  git clone "$REPO_URL" "$REPO_DIR"
fi

echo "== checking out $COMMIT_SHA (detached) =="
git -C "$REPO_DIR" checkout --detach "$COMMIT_SHA"

echo "== installing package (pip install -e .) =="
pip install -e "$REPO_DIR"

echo "== nvidia-smi =="
nvidia-smi || echo "nvidia-smi unavailable"

echo "== running preflight =="
set +e
python "$REPO_DIR/scripts/colab/preflight.py" --output-root "$REPO_DIR/artifacts/colab"
STATUS=$?
set -e

case "$STATUS" in
  0) echo "SUMMARY: PASS (task=$TASK, commit=$COMMIT_SHA)" ;;
  3) echo "SUMMARY: POLICY-DENIED (task=$TASK, commit=$COMMIT_SHA)" ;;
  *) echo "SUMMARY: FAIL (task=$TASK, commit=$COMMIT_SHA, exit=$STATUS)" ;;
esac

exit "$STATUS"
