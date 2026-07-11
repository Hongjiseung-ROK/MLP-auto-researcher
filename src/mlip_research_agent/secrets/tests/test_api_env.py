"""Tests for the Materials Project api.env loader.

Every fixture here is synthetic and written to tmp_path. These tests must
never read, reference, or depend on the real repo-root api.env file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from mlip_research_agent.secrets.api_env import (
    EXPORTED_KEY_NAME,
    SUPPORTED_KEY_NAMES,
    ApiEnvStatus,
    load_mp_api_key,
    mp_api_key_env,
)

SYNTHETIC_VALUE = "synthetic-not-a-real-key-PLACEHOLDER"


def _write_env(tmp_path: Path, content: str) -> Path:
    env_file = tmp_path / "synthetic.env"
    env_file.write_text(content, encoding="utf-8")
    return env_file


def _assert_no_key_material(status: ApiEnvStatus, value: str = SYNTHETIC_VALUE) -> None:
    assert value not in status.model_dump_json()
    assert value not in repr(status)


def test_missing_file_returns_honest_status(tmp_path: Path) -> None:
    environ: dict[str, str] = {}
    status = load_mp_api_key(tmp_path / "does-not-exist.env", environ=environ)
    assert status.file_present is False
    assert status.key_present is False
    assert status.key_name is None
    assert status.exported is False
    assert environ == {}


def test_empty_file_returns_honest_status(tmp_path: Path) -> None:
    pytest.importorskip("dotenv")
    env_file = _write_env(tmp_path, "")
    environ: dict[str, str] = {}
    status = load_mp_api_key(env_file, environ=environ)
    assert status.file_present is True
    assert status.key_present is False
    assert status.key_name is None
    assert status.exported is False
    assert environ == {}


def test_unsupported_keys_only(tmp_path: Path) -> None:
    pytest.importorskip("dotenv")
    env_file = _write_env(tmp_path, f"SOME_OTHER_KEY={SYNTHETIC_VALUE}\nUNRELATED=1\n")
    environ: dict[str, str] = {}
    status = load_mp_api_key(env_file, environ=environ)
    assert status.file_present is True
    assert status.key_present is False
    assert status.exported is False
    assert environ == {}
    _assert_no_key_material(status)


@pytest.mark.parametrize("name", SUPPORTED_KEY_NAMES)
def test_each_supported_name_is_recognized(tmp_path: Path, name: str) -> None:
    pytest.importorskip("dotenv")
    env_file = _write_env(tmp_path, f"{name}={SYNTHETIC_VALUE}\n")
    environ: dict[str, str] = {}
    status = load_mp_api_key(env_file, environ=environ)
    assert status.file_present is True
    assert status.key_present is True
    assert status.key_name == name
    assert status.exported is True
    assert environ == {EXPORTED_KEY_NAME: SYNTHETIC_VALUE}
    _assert_no_key_material(status)


def test_precedence_order_mp_api_key_wins(tmp_path: Path) -> None:
    pytest.importorskip("dotenv")
    winner = f"{SYNTHETIC_VALUE}-winner"
    lines = [
        f"MATERIALS_PROJECT_KEY={SYNTHETIC_VALUE}-third",
        f"MATERIALS_PROJECT_API_KEY={SYNTHETIC_VALUE}-second",
        f"MP_API_KEY={winner}",
    ]
    env_file = _write_env(tmp_path, "\n".join(lines) + "\n")
    environ: dict[str, str] = {}
    status = load_mp_api_key(env_file, environ=environ)
    assert status.key_name == "MP_API_KEY"
    assert environ[EXPORTED_KEY_NAME] == winner


def test_empty_valued_name_falls_through_to_next(tmp_path: Path) -> None:
    pytest.importorskip("dotenv")
    env_file = _write_env(
        tmp_path,
        f"MP_API_KEY=\nMATERIALS_PROJECT_API_KEY={SYNTHETIC_VALUE}\n",
    )
    environ: dict[str, str] = {}
    status = load_mp_api_key(env_file, environ=environ)
    assert status.key_name == "MATERIALS_PROJECT_API_KEY"
    assert environ[EXPORTED_KEY_NAME] == SYNTHETIC_VALUE


def test_fake_environ_leaves_os_environ_untouched(tmp_path: Path) -> None:
    pytest.importorskip("dotenv")
    env_file = _write_env(tmp_path, f"MP_API_KEY={SYNTHETIC_VALUE}\n")
    before = os.environ.get(EXPORTED_KEY_NAME)
    environ: dict[str, str] = {}
    load_mp_api_key(env_file, environ=environ)
    assert environ == {EXPORTED_KEY_NAME: SYNTHETIC_VALUE}
    assert os.environ.get(EXPORTED_KEY_NAME) == before


def test_context_manager_sets_then_removes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("dotenv")
    monkeypatch.delenv(EXPORTED_KEY_NAME, raising=False)
    env_file = _write_env(tmp_path, f"MP_API_KEY={SYNTHETIC_VALUE}\n")
    with mp_api_key_env(env_file) as status:
        assert status.exported is True
        assert os.environ[EXPORTED_KEY_NAME] == SYNTHETIC_VALUE
    assert EXPORTED_KEY_NAME not in os.environ


def test_context_manager_removes_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("dotenv")
    monkeypatch.delenv(EXPORTED_KEY_NAME, raising=False)
    env_file = _write_env(tmp_path, f"MP_API_KEY={SYNTHETIC_VALUE}\n")
    with pytest.raises(ValueError, match="boom"), mp_api_key_env(env_file):
        raise ValueError("boom")
    assert EXPORTED_KEY_NAME not in os.environ


def test_context_manager_restores_preexisting_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("dotenv")
    preexisting = "synthetic-preexisting-PLACEHOLDER"
    monkeypatch.setenv(EXPORTED_KEY_NAME, preexisting)
    env_file = _write_env(tmp_path, f"MP_API_KEY={SYNTHETIC_VALUE}\n")
    with mp_api_key_env(env_file):
        assert os.environ[EXPORTED_KEY_NAME] == SYNTHETIC_VALUE
    assert os.environ[EXPORTED_KEY_NAME] == preexisting


def test_status_never_contains_key_material(tmp_path: Path) -> None:
    pytest.importorskip("dotenv")
    env_file = _write_env(tmp_path, f"MP_API_KEY={SYNTHETIC_VALUE}\n")
    environ: dict[str, str] = {}
    status = load_mp_api_key(env_file, environ=environ)
    _assert_no_key_material(status)
    # The model has no field that could carry a value even by accident.
    assert set(ApiEnvStatus.model_fields) == {
        "file_present",
        "key_present",
        "key_name",
        "exported",
        "detail",
    }


def test_runtime_error_when_dotenv_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = _write_env(tmp_path, f"MP_API_KEY={SYNTHETIC_VALUE}\n")
    # A None entry in sys.modules makes `import dotenv` raise ImportError.
    monkeypatch.setitem(sys.modules, "dotenv", None)
    environ: dict[str, str] = {}
    with pytest.raises(RuntimeError, match=r"materials-project") as excinfo:
        load_mp_api_key(env_file, environ=environ)
    assert SYNTHETIC_VALUE not in str(excinfo.value)
    assert environ == {}
