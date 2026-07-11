"""Materials Project reconnaissance skill.

Reconnaissance/metadata only: outputs are NOT claim-eligible dataset sources
(see SKILL.md). Security invariants:

- The API key is only ever exported into the process environment by
  :func:`mlip_research_agent.secrets.api_env.mp_api_key_env` for the duration
  of the network call; ``MPRester()`` reads ``MP_API_KEY`` from the
  environment. This module never opens ``api.env``, never receives the key as
  a value, and never writes key material, raw exception text, or HTTP headers
  into artifacts, cache files, logs, or error messages.
- Every stored byte is sanitized first: documents are reduced to exactly the
  explicitly requested fields before anything is written or cached.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from mlip_research_agent.artifacts.registry import Artifact, sha256_file
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.secrets.api_env import ApiEnvStatus, mp_api_key_env
from mlip_research_agent.skills.atomistics.structures_io import StructureSet, atoms_to_record
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.data.materials_project.schema import (
    MaterialsProjectInput,
    MaterialsProjectOutput,
)
from mlip_research_agent.skills.data.materials_project.validators import validate_inputs

RESPONSE_FILENAME = "response.json"
STRUCTURES_FILENAME = "structures.json"
PROVENANCE_FILENAME = "provenance.json"
#: Cheapest possible authenticated probe: one known material, one field.
AUTH_PROBE_MATERIAL_ID = "mp-30"
MAX_NETWORK_ATTEMPTS = 3
#: Fixed short backoff (seconds) between retryable attempts; deterministic, no jitter.
RETRY_BACKOFF_SECONDS = 0.05

_INSTALL_HINT = (
    "mp-api is not installed in this environment. Install the optional extra with: "
    "pip install -e '.[materials-project]'"
)

T = TypeVar("T")


def _default_cache_root() -> Path:
    """Repo-root ``data/cache/materials_project`` (gitignored; sanitized payloads only)."""
    return Path(__file__).resolve().parents[5] / "data" / "cache" / "materials_project"


def _load_mprester() -> Any:
    """Lazy import so the skill registers without the materials-project extra."""
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise SkillError(
            _INSTALL_HINT,
            failure_class=FailureClass.TOOL_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        ) from exc
    return MPRester


def _require_exported_key(status: ApiEnvStatus) -> None:
    if not status.exported:
        # status.detail is built from static strings and file paths only.
        raise SkillError(
            f"Materials Project API key unavailable: {status.detail}",
            failure_class=FailureClass.TOOL_ERROR,
            severity=Severity.HIGH,
            retryable=False,
            likely_causes=[
                "api.env is missing or defines none of the supported key names "
                "(MP_API_KEY, MATERIALS_PROJECT_API_KEY, MATERIALS_PROJECT_KEY)"
            ],
        )


def _classify_exception(exc: Exception) -> tuple[FailureClass, bool, str]:
    """Map a provider exception to (failure_class, retryable, label).

    Classification uses the exception class name and message as best mp_api
    exposes them; the raw message is never propagated into SkillError text.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(t in text for t in ("401", "403", "unauthorized", "forbidden", "invalid api key")):
        return FailureClass.TOOL_ERROR, False, "auth"
    if any(t in text for t in ("429", "rate limit", "too many requests")):
        return FailureClass.RESOURCE_EXHAUSTED, True, "rate_limit"
    return FailureClass.TOOL_ERROR, True, "transient"


def _call_with_retries(fn: Callable[[], T], what: str) -> T:
    """Bounded deterministic retry loop: max 3 attempts, retryable classes only."""
    for attempt in range(1, MAX_NETWORK_ATTEMPTS + 1):
        try:
            return fn()
        except SkillError:
            raise
        except Exception as exc:  # provider exception surface is untyped
            failure_class, retryable, label = _classify_exception(exc)
            if not retryable or attempt == MAX_NETWORK_ATTEMPTS:
                # Only the exception *type name* is included: raw provider
                # messages may echo request details and are never stored.
                raise SkillError(
                    f"{what} failed ({label}) on attempt {attempt}/{MAX_NETWORK_ATTEMPTS}: "
                    f"{type(exc).__name__}",
                    failure_class=failure_class,
                    severity=Severity.MEDIUM if retryable else Severity.HIGH,
                    retryable=retryable,
                    likely_causes={
                        "auth": ["invalid or expired Materials Project API key"],
                        "rate_limit": ["Materials Project rate limit hit"],
                        "transient": ["transient network or provider error"],
                    }[label],
                ) from exc
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise AssertionError("unreachable")


def _jsonable(value: Any) -> Any:
    """Deterministic plain-JSON conversion for sanitized document values."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _jsonable(dump())
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _jsonable(as_dict())
    return str(value)


def _doc_to_dict(doc: Any, fields: list[str]) -> dict[str, Any]:
    """Reduce a provider document to exactly the requested fields, plain JSON."""
    if isinstance(doc, dict):
        raw = {name: doc.get(name) for name in fields}
    else:
        raw = {name: getattr(doc, name, None) for name in fields}
    return {name: _jsonable(value) for name, value in raw.items()}


def _canonical_query_key(query: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(query, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cache_path(cache_root: Path, query: dict[str, Any]) -> Path:
    return cache_root / f"{_canonical_query_key(query)}.json"


def _cache_load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cache_store(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _client_version() -> str | None:
    try:
        return importlib.metadata.version("mp-api")
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_payload_artifact(
    ctx: SkillContext, filename: str, kind: str, payload: dict[str, Any]
) -> Artifact:
    path = ctx.step_dir / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return ctx.registry.register(path, kind=kind, step_id=ctx.step_id)


def _write_provenance(
    ctx: SkillContext,
    *,
    operation: str,
    query: dict[str, Any],
    material_ids: list[str],
    from_cache: bool,
    response_path: Path,
) -> Artifact:
    payload = {
        "skill": MaterialsProjectReconSkill.name,
        "operation": operation,
        "query": query,
        "mp_api_client_version": _client_version(),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "from_cache": from_cache,
        "material_ids": material_ids,
        "response_file": response_path.name,
        "response_sha256": sha256_file(response_path),
    }
    return _write_payload_artifact(ctx, PROVENANCE_FILENAME, "mp_provenance", payload)


@register_skill
class MaterialsProjectReconSkill(Skill):
    name = "materials_project_recon"
    input_model = MaterialsProjectInput
    output_model = MaterialsProjectOutput
    cost_class = "cheap"  # bounded metadata queries; cached repeats are free
    permission_level = "auto"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, MaterialsProjectInput)
        validate_inputs(params)
        cache_root = (
            Path(params.cache_root) if params.cache_root is not None else _default_cache_root()
        )
        env_file = Path(params.api_env_file) if params.api_env_file is not None else None
        if params.operation == "auth_check":
            return self._auth_check(env_file)
        if params.operation == "summary_search":
            return self._summary_search(params, ctx, cache_root, env_file)
        return self._get_structures(params, ctx, cache_root, env_file)

    def _auth_check(self, env_file: Path | None) -> MaterialsProjectOutput:
        mprester_cls = _load_mprester()
        with mp_api_key_env(env_file) as status:
            _require_exported_key(status)

            def _probe() -> list[Any]:
                with mprester_cls() as mpr:
                    return list(
                        mpr.materials.summary.search(
                            material_ids=[AUTH_PROBE_MATERIAL_ID], fields=["material_id"]
                        )
                    )

            try:
                docs = _call_with_retries(_probe, "auth_check probe")
            except SkillError as err:
                retry_note = "retryable" if err.retryable else "non-retryable"
                return MaterialsProjectOutput(
                    operation="auth_check",
                    authenticated=False,
                    n_results=0,
                    material_ids=[],
                    response_artifact=None,
                    provenance_artifact=None,
                    from_cache=False,
                    detail=(
                        f"not authenticated: {err.failure_class.value} ({retry_note}) — {err}"
                    ),
                )
        return MaterialsProjectOutput(
            operation="auth_check",
            authenticated=True,
            n_results=len(docs),
            material_ids=[],
            response_artifact=None,
            provenance_artifact=None,
            from_cache=False,
            detail="authenticated: probe query succeeded",
        )

    def _summary_search(
        self,
        params: MaterialsProjectInput,
        ctx: SkillContext,
        cache_root: Path,
        env_file: Path | None,
    ) -> MaterialsProjectOutput:
        query = {
            "operation": "summary_search",
            "chemsys": params.chemsys,
            "fields": sorted(set(params.fields)),
            "max_results": params.max_results,
        }
        cache_file = _cache_path(cache_root, query)
        payload = _cache_load(cache_file) if params.use_cache else None
        from_cache = payload is not None

        if payload is None:
            mprester_cls = _load_mprester()
            with mp_api_key_env(env_file) as status:
                _require_exported_key(status)

                def _fetch() -> list[Any]:
                    with mprester_cls() as mpr:
                        return list(
                            mpr.materials.summary.search(
                                chemsys=params.chemsys, fields=list(params.fields)
                            )
                        )

                docs = _call_with_retries(_fetch, "summary_search")
            documents = [_doc_to_dict(doc, params.fields) for doc in docs]
            documents.sort(key=lambda d: str(d.get("material_id")))
            documents = documents[: params.max_results]
            payload = {"operation": "summary_search", "query": query, "documents": documents}
            if params.use_cache:
                _cache_store(cache_file, payload)

        documents = list(payload.get("documents", []))
        material_ids = [str(doc["material_id"]) for doc in documents if "material_id" in doc]
        response_artifact = _write_payload_artifact(ctx, RESPONSE_FILENAME, "mp_response", payload)
        provenance_artifact = _write_provenance(
            ctx,
            operation="summary_search",
            query=query,
            material_ids=material_ids,
            from_cache=from_cache,
            response_path=ctx.step_dir / RESPONSE_FILENAME,
        )
        source = "cache" if from_cache else "network"
        return MaterialsProjectOutput(
            operation="summary_search",
            authenticated=not from_cache,
            n_results=len(documents),
            material_ids=material_ids,
            response_artifact=response_artifact.artifact_id,
            provenance_artifact=provenance_artifact.artifact_id,
            from_cache=from_cache,
            detail=(
                f"summary_search for chemsys '{params.chemsys}' returned "
                f"{len(documents)} document(s) from {source}"
            ),
        )

    def _get_structures(
        self,
        params: MaterialsProjectInput,
        ctx: SkillContext,
        cache_root: Path,
        env_file: Path | None,
    ) -> MaterialsProjectOutput:
        material_ids = sorted(params.material_ids)
        query = {"operation": "get_structures", "material_ids": material_ids}
        cache_file = _cache_path(cache_root, query)
        payload = _cache_load(cache_file) if params.use_cache else None
        from_cache = payload is not None

        if payload is not None:
            structure_set = StructureSet.model_validate(payload["structure_set"])
        else:
            mprester_cls = _load_mprester()
            try:
                from pymatgen.io.ase import AseAtomsAdaptor
            except ImportError as exc:
                raise SkillError(
                    "pymatgen is required for get_structures; install the atomistics "
                    "extra: pip install -e '.[atomistics]'",
                    failure_class=FailureClass.TOOL_ERROR,
                    severity=Severity.HIGH,
                    retryable=False,
                ) from exc
            with mp_api_key_env(env_file) as status:
                _require_exported_key(status)
                structures = [
                    _fetch_structure(mprester_cls, material_id) for material_id in material_ids
                ]
            records = [
                atoms_to_record(AseAtomsAdaptor.get_atoms(structure), index, 0.0)
                for index, structure in enumerate(structures)
            ]
            structure_set = StructureSet(systems=records)
            payload = {
                "operation": "get_structures",
                "query": query,
                "structure_set": structure_set.model_dump(),
            }
            if params.use_cache:
                _cache_store(cache_file, payload)

        structures_path = ctx.step_dir / STRUCTURES_FILENAME
        structure_set.save(structures_path)
        response_artifact = ctx.registry.register(
            structures_path, kind="mp_structures", step_id=ctx.step_id
        )
        provenance_artifact = _write_provenance(
            ctx,
            operation="get_structures",
            query=query,
            material_ids=material_ids,
            from_cache=from_cache,
            response_path=structures_path,
        )
        source = "cache" if from_cache else "network"
        return MaterialsProjectOutput(
            operation="get_structures",
            authenticated=not from_cache,
            n_results=len(structure_set.systems),
            material_ids=material_ids,
            response_artifact=response_artifact.artifact_id,
            provenance_artifact=provenance_artifact.artifact_id,
            from_cache=from_cache,
            detail=(
                f"get_structures fetched {len(structure_set.systems)} structure(s) "
                f"from {source}"
            ),
        )


def _fetch_structure(mprester_cls: Any, material_id: str) -> Any:
    def _call() -> Any:
        with mprester_cls() as mpr:
            return mpr.get_structure_by_material_id(material_id)

    return _call_with_retries(_call, f"get_structures[{material_id}]")
