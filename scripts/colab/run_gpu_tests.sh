#!/usr/bin/env bash
# Run on a Colab VM right after scripts/colab/bootstrap.sh has installed the
# package. Runs the repo's own test suite (excluding tests that need real
# remote credentials) plus a CPU-tolerant preflight pass.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

python -m pytest tests -q -m "not colab_remote and not vessl_remote"
python scripts/colab/preflight.py --output-root artifacts/colab --allow-cpu
