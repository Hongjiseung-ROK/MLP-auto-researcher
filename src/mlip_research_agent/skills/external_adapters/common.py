"""Shared fail-closed plumbing for adapters over the locked external skill pack.

The upstream registration lives in
``external-skills/jinzhezenggroup/computational-chemistry-agent-skills/upstream.yaml``;
a test asserts the constants here match it. Adapters never execute upstream
code — they implement each capability natively against the underlying tool and
fail closed when the optional dependency is not installed.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

from mlip_research_agent.artifacts.registry import Artifact
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillContext, SkillError

UPSTREAM_REPOSITORY = (
    "https://github.com/jinzhezenggroup/computational-chemistry-agent-skills"
)
UPSTREAM_COMMIT = "93ea0c4c716ad116869fba2ade26cccfd5cd05fc"

EXTERNAL_SCIENTIFIC_STATUS = "external_adapter_infrastructure_only"


def adapter_error(message: str) -> SkillError:
    return SkillError(
        message,
        failure_class=FailureClass.VALIDATION_ERROR,
        severity=Severity.HIGH,
        retryable=False,
    )


def missing_dependency(package: str, extra: str | None) -> SkillError:
    hint = (
        f"install the optional {extra!r} extra in a compatible environment"
        if extra
        else "install the external tool"
    )
    return SkillError(
        f"external adapter dependency {package!r} is not installed; {hint}. "
        "External adapters fail closed by design (upstream.yaml policy).",
        failure_class=FailureClass.TOOL_ERROR,
        severity=Severity.HIGH,
        retryable=False,
    )


def require_module(module: str, *, package: str, extra: str | None) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise missing_dependency(package, extra) from exc


def upstream_provenance(upstream_path: str) -> dict[str, str]:
    """Provenance block every adapter output carries."""
    return {
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_skill_path": upstream_path,
        "scientific_status": EXTERNAL_SCIENTIFIC_STATUS,
    }


def write_registered_json(
    ctx: SkillContext,
    filename: str,
    payload: dict[str, Any],
    kind: str,
) -> Artifact:
    path = ctx.step_dir / filename
    if path.exists():
        raise adapter_error(f"external adapter output already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return ctx.registry.register(path, kind, ctx.step_id)


def write_registered_text(
    ctx: SkillContext,
    filename: str,
    text: str,
    kind: str,
) -> Artifact:
    path = ctx.step_dir / filename
    if path.exists():
        raise adapter_error(f"external adapter output already exists: {path}")
    path.write_text(text)
    return ctx.registry.register(path, kind, ctx.step_id)


def load_structure_set(ctx: SkillContext, relative: str) -> Any:
    from mlip_research_agent.skills.atomistics.structures_io import StructureSet

    path = (ctx.run_dir / relative).resolve()
    try:
        path.relative_to(ctx.run_dir.resolve())
    except ValueError as exc:
        raise adapter_error(f"structures path escapes the run directory: {relative}") from exc
    if not path.is_file():
        raise adapter_error(f"structures file not found: {relative}")
    try:
        return StructureSet.load(path)
    except (OSError, ValueError) as exc:
        raise adapter_error(f"structures file is invalid: {exc}") from exc
