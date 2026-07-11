"""Opt-in tests against real Colab/VESSL credentials; skipped when absent.

These never submit real jobs -- they only check that the provider reports an
authenticated session via validate_auth(). The compute layer itself is
mock-first (docs/OPEN_QUESTIONS.md gates any real backend wiring in src/), so
this test file supplies its own minimal, test-local, read-only transports
that shell out to the already-installed CLI tools purely to confirm they are
authenticated -- e.g. `colab sessions` (lists sessions, starts nothing).
Nothing here submits a job or touches src/.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mlip_research_agent.compute.colab import ColabProvider
from mlip_research_agent.compute.policy import load_policy
from mlip_research_agent.compute.schemas import JobSpec
from mlip_research_agent.compute.vessl import VesslProvider

REPO_ROOT = Path(__file__).parent.parent
POLICY_PATH = REPO_ROOT / "configs" / "compute_policy.yaml"

_COLAB_TOKEN_PATH = Path.home() / ".config" / "colab-cli" / "token.json"


def _colab_credentials_missing() -> bool:
    return shutil.which("colab") is None or not _COLAB_TOKEN_PATH.exists()


def _vessl_credentials_missing() -> bool:
    return "VESSL_ACCESS_TOKEN" not in os.environ


class _AuthOnlyColabTransport:
    """Read-only real transport: checks `colab` CLI auth, never runs a job."""

    def auth_status(self) -> tuple[bool, str, str]:
        try:
            proc = subprocess.run(
                ["colab", "sessions"], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, "colab-cli", f"colab CLI auth check failed: {exc!r}"
        authenticated = proc.returncode == 0
        detail = "colab CLI reachable" if authenticated else proc.stderr.strip()
        return authenticated, "colab-cli", detail

    def start_session(self, spec: JobSpec) -> Any:
        raise NotImplementedError("opt-in auth check never submits jobs")

    def execute(self, spec: JobSpec, run_dir: Path) -> Any:
        raise NotImplementedError("opt-in auth check never submits jobs")

    def stop_session(self) -> None:
        raise NotImplementedError("opt-in auth check never submits jobs")


class _AuthOnlyVesslTransport:
    """Read-only real transport: checks VESSL CLI auth, never runs a job."""

    def auth_status(self) -> tuple[bool, str, str]:
        try:
            proc = subprocess.run(
                ["vessl", "whoami"], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, "vessl-cli", f"vessl CLI auth check failed: {exc!r}"
        authenticated = proc.returncode == 0
        detail = "vessl CLI reachable" if authenticated else proc.stderr.strip()
        return authenticated, "vessl-cli", detail

    def start_session(self, spec: JobSpec) -> Any:
        raise NotImplementedError("opt-in auth check never submits jobs")

    def execute(self, spec: JobSpec, run_dir: Path) -> Any:
        raise NotImplementedError("opt-in auth check never submits jobs")

    def stop_session(self) -> None:
        raise NotImplementedError("opt-in auth check never submits jobs")


@pytest.mark.colab_remote
@pytest.mark.skipif(_colab_credentials_missing(), reason="no local Colab CLI credentials")
def test_colab_remote_validate_auth_reports_authenticated(tmp_path: Path) -> None:
    policy = load_policy(POLICY_PATH)
    provider = ColabProvider(policy, tmp_path, transport=_AuthOnlyColabTransport())
    status = provider.validate_auth()
    assert status.authenticated is True


@pytest.mark.vessl_remote
@pytest.mark.skipif(_vessl_credentials_missing(), reason="no VESSL_ACCESS_TOKEN in environment")
def test_vessl_remote_validate_auth_reports_authenticated(tmp_path: Path) -> None:
    policy = load_policy(POLICY_PATH)
    provider = VesslProvider(policy, tmp_path, transport=_AuthOnlyVesslTransport())
    status = provider.validate_auth()
    assert status.authenticated is True
