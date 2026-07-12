"""Strict current-Cloud ``vesslctl`` subprocess transport."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mlip_research_agent.compute.router import redact_secrets

JsonValue = dict[str, Any] | list[Any]
Runner = Callable[..., subprocess.CompletedProcess[bytes]]

_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_READ_ONLY_EXACT = {
    ("auth", "status"),
    ("config", "show"),
    ("billing", "show"),
    ("org", "list"),
    ("team", "list"),
    ("cluster", "list"),
    ("resource-spec", "list", "--usable-only"),
    ("storage", "list"),
    ("volume", "list"),
    ("job", "list"),
}


class VesslCliError(RuntimeError):
    pass


class VesslCliNotInstalled(VesslCliError):
    pass


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            folded = key_text.casefold().replace("-", "_")
            if any(
                marker in folded
                for marker in ("token", "secret", "password", "credential", "api_key")
            ):
                sanitized[key_text] = "[REDACTED]"
            else:
                sanitized[key_text] = _sanitize_json(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    return value


class VesslCliTransport:
    """Allowlisted subprocess boundary; credentials remain owned by vesslctl."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        timeout_seconds: int = 30,
        max_output_bytes: int = 2_000_000,
        runner: Runner = subprocess.run,
    ) -> None:
        self.executable = executable or shutil.which("vesslctl")
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self._runner = runner

    @property
    def installed(self) -> bool:
        return self.executable is not None

    def version(self) -> str:
        result = self._invoke(["--version"])
        value = result.strip()
        if not value:
            raise VesslCliError("vesslctl returned an empty version")
        return value

    def read_json(self, args: Sequence[str]) -> JsonValue:
        normalized = tuple(args)
        if not self._read_only_allowed(normalized):
            raise VesslCliError(f"vesslctl command is not read-only allowlisted: {normalized!r}")
        text = self._invoke([*normalized, "-o", "json"])
        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VesslCliError("vesslctl returned malformed JSON") from exc
        if not isinstance(payload, dict | list):
            raise VesslCliError("vesslctl JSON root must be an object or array")
        sanitized = _sanitize_json(payload)
        assert isinstance(sanitized, dict | list)
        return sanitized

    def create_job(self, config_path: Path, *, approval_validated: bool) -> JsonValue:
        if not approval_validated:
            raise VesslCliError("job create requires a validated sealed cost approval")
        if not config_path.is_file():
            raise VesslCliError("generated VESSL Job config is missing")
        text = self._invoke(["job", "create", "--file", str(config_path), "-o", "json"])
        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VesslCliError("vesslctl job create returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise VesslCliError("vesslctl job create JSON root must be an object")
        sanitized = _sanitize_json(payload)
        assert isinstance(sanitized, dict)
        return sanitized

    def _read_only_allowed(self, args: tuple[str, ...]) -> bool:
        if args in _READ_ONLY_EXACT:
            return True
        return (
            len(args) == 3
            and args[:2] in {("job", "show"), ("job", "logs")}
            and bool(_SLUG.fullmatch(args[2]))
        )

    def _invoke(self, args: list[str]) -> str:
        if self.executable is None:
            raise VesslCliNotInstalled(
                "vesslctl is not installed; installation requires explicit owner confirmation"
            )
        command = [self.executable, *args]
        try:
            result = self._runner(
                command,
                shell=False,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VesslCliError(f"vesslctl execution failed: {type(exc).__name__}") from exc
        stdout = bytes(result.stdout)
        stderr = bytes(result.stderr)
        if len(stdout) + len(stderr) > self.max_output_bytes:
            raise VesslCliError("vesslctl output exceeded the configured size limit")
        stdout_text = redact_secrets(stdout.decode("utf-8", errors="replace"))
        stderr_text = redact_secrets(stderr.decode("utf-8", errors="replace"))
        if result.returncode != 0:
            detail = stderr_text.strip() or stdout_text.strip() or "no detail"
            raise VesslCliError(f"vesslctl exited {result.returncode}: {detail}")
        return stdout_text
