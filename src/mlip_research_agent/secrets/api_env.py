"""Secure loader for the Materials Project API key from the repo-root ``api.env``.

Security invariants:

- The key value is copied into the process environment (``MP_API_KEY``) and
  nowhere else. No value, value length, hash, or fragment ever appears in a
  returned model, exception message, log line, or repr.
- ``ApiEnvStatus`` structurally cannot carry key material: it has only
  booleans, the matched key *name* (which is not secret), and human guidance
  text built exclusively from static strings and file paths.
- Prefer :func:`mp_api_key_env` so the exported key does not outlive its use.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict

#: Supported key names in the env file, in precedence order.
SUPPORTED_KEY_NAMES = ("MP_API_KEY", "MATERIALS_PROJECT_API_KEY", "MATERIALS_PROJECT_KEY")

#: Environment variable the selected value is exported under.
EXPORTED_KEY_NAME = "MP_API_KEY"

_DOTENV_INSTALL_HINT = (
    "python-dotenv is required to parse the api.env file. "
    "Install it with: pip install -e '.[materials-project]'"
)


def _default_env_file() -> Path:
    """Repo-root ``api.env`` (this file lives at src/mlip_research_agent/secrets/)."""
    return Path(__file__).resolve().parents[3] / "api.env"


class ApiEnvStatus(BaseModel):
    """Outcome of a load attempt. Carries names and booleans only, never values."""

    model_config = ConfigDict(extra="forbid")

    file_present: bool
    key_present: bool
    key_name: str | None  # which SUPPORTED_KEY_NAMES entry matched; the name is not secret
    exported: bool
    detail: str  # human guidance built from static strings/paths, never any part of a value


def load_mp_api_key(
    env_file: Path | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> ApiEnvStatus:
    """Load the Materials Project API key from ``env_file`` into ``environ``.

    The first name in :data:`SUPPORTED_KEY_NAMES` with a non-empty value wins;
    its value is copied into ``environ["MP_API_KEY"]``. A missing file or
    missing key returns an honest status (``exported=False``) without raising.

    Raises:
        RuntimeError: if python-dotenv is not installed (guidance only, no
            file contents).
    """
    if env_file is None:
        env_file = _default_env_file()
    if environ is None:
        environ = os.environ

    if not env_file.is_file():
        return ApiEnvStatus(
            file_present=False,
            key_present=False,
            key_name=None,
            exported=False,
            detail=(
                f"No env file found at {env_file}. Create it with a line like "
                f"'{EXPORTED_KEY_NAME}=<your Materials Project API key>'."
            ),
        )

    try:
        from dotenv import dotenv_values  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:
        raise RuntimeError(_DOTENV_INSTALL_HINT) from exc

    values = dotenv_values(env_file)
    for name in SUPPORTED_KEY_NAMES:
        value = values.get(name)
        if value:
            environ[EXPORTED_KEY_NAME] = value
            return ApiEnvStatus(
                file_present=True,
                key_present=True,
                key_name=name,
                exported=True,
                detail=(
                    f"Found '{name}' in {env_file} and exported it as "
                    f"'{EXPORTED_KEY_NAME}'."
                ),
            )

    supported = ", ".join(SUPPORTED_KEY_NAMES)
    return ApiEnvStatus(
        file_present=True,
        key_present=False,
        key_name=None,
        exported=False,
        detail=(
            f"Env file {env_file} exists but defines none of the supported key "
            f"names with a non-empty value: {supported}."
        ),
    )


@contextmanager
def mp_api_key_env(env_file: Path | None = None) -> Iterator[ApiEnvStatus]:
    """Export the key into ``os.environ`` for the duration of the block.

    On exit, ``MP_API_KEY`` is removed — or restored to its pre-existing value —
    even if the block raises. This is the preferred consumption path.
    """
    previous = os.environ.get(EXPORTED_KEY_NAME)
    status = load_mp_api_key(env_file, environ=os.environ)
    try:
        yield status
    finally:
        if previous is None:
            os.environ.pop(EXPORTED_KEY_NAME, None)
        else:
            os.environ[EXPORTED_KEY_NAME] = previous
