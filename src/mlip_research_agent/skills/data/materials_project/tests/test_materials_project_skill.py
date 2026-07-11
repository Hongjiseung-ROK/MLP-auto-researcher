"""Offline correctness tests for materials_project_recon.

No network, no real key, no reads of the repo-root api.env: a fake
``mp_api.client`` module is injected into sys.modules and the key comes from a
synthetic tmp env file passed via the ``api_env_file`` dev knob.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.schemas.failure import FailureClass
from mlip_research_agent.skills.atomistics.structures_io import StructureSet
from mlip_research_agent.skills.base import SkillContext, SkillError
from mlip_research_agent.skills.data.materials_project import implementation
from mlip_research_agent.skills.data.materials_project.implementation import (
    MaterialsProjectReconSkill,
)
from mlip_research_agent.skills.data.materials_project.schema import (
    MaterialsProjectInput,
    MaterialsProjectOutput,
)

SYNTHETIC_KEY = "synthetic-PLACEHOLDER-key-0000"


@dataclass
class FakeMPApi:
    """Canned provider state shared with the injected FakeMPRester."""

    summary_docs: list[Any] = field(default_factory=list)
    structures: dict[str, Any] = field(default_factory=dict)
    errors: list[Exception] = field(default_factory=list)  # popped once per call
    calls: list[str] = field(default_factory=list)

    def _maybe_raise(self) -> None:
        if self.errors:
            raise self.errors.pop(0)


@pytest.fixture()
def fake_mp(monkeypatch: pytest.MonkeyPatch) -> FakeMPApi:
    controller = FakeMPApi()

    class FakeSummary:
        def search(self, **kwargs: Any) -> list[Any]:
            controller.calls.append("summary.search")
            controller._maybe_raise()
            return list(controller.summary_docs)

    class FakeMPRester:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.materials = SimpleNamespace(summary=FakeSummary())

        def __enter__(self) -> FakeMPRester:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def get_structure_by_material_id(self, material_id: str) -> Any:
            controller.calls.append(f"get_structure:{material_id}")
            controller._maybe_raise()
            return controller.structures[material_id]

    client_mod = types.ModuleType("mp_api.client")
    client_mod.MPRester = FakeMPRester  # type: ignore[attr-defined]
    pkg_mod = types.ModuleType("mp_api")
    pkg_mod.client = client_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mp_api", pkg_mod)
    monkeypatch.setitem(sys.modules, "mp_api.client", client_mod)
    monkeypatch.setattr(implementation, "RETRY_BACKOFF_SECONDS", 0.0)
    return controller


@pytest.fixture()
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic-api.env"
    path.write_text(f"MP_API_KEY={SYNTHETIC_KEY}\n")
    return path


def make_ctx(tmp_path: Path, step_id: str = "recon") -> SkillContext:
    run_dir = tmp_path / f"run-{step_id}"
    return SkillContext(
        run_dir=run_dir,
        step_id=step_id,
        seed=7,
        attempt=0,
        registry=ArtifactRegistry(run_dir),
    )


def run_skill(
    tmp_path: Path, env_file: Path, step_id: str = "recon", **overrides: object
) -> tuple[MaterialsProjectOutput, SkillContext]:
    ctx = make_ctx(tmp_path, step_id)
    inputs = MaterialsProjectInput.model_validate(
        {
            "api_env_file": str(env_file),
            "cache_root": str(tmp_path / "cache"),
            **overrides,
        }
    )
    out = MaterialsProjectReconSkill().run(inputs, ctx)
    assert isinstance(out, MaterialsProjectOutput)
    return out, ctx


def assert_no_key_material(root: Path) -> None:
    """No skill-written byte may contain the key (only the fixture env file holds it)."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "synthetic-api.env":
            assert SYNTHETIC_KEY.encode() not in path.read_bytes(), path


def test_missing_key_raises_guided_tool_error(fake_mp: FakeMPApi, tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.env"
    with pytest.raises(SkillError) as excinfo:
        run_skill(tmp_path, missing, operation="summary_search")
    err = excinfo.value
    assert err.failure_class is FailureClass.TOOL_ERROR
    assert not err.retryable
    assert "MP_API_KEY" in str(err)  # names supported keys, never values
    assert fake_mp.calls == []


def test_auth_check_success(fake_mp: FakeMPApi, tmp_path: Path, env_file: Path) -> None:
    fake_mp.summary_docs = [SimpleNamespace(material_id="mp-30")]
    out, _ = run_skill(tmp_path, env_file, operation="auth_check")
    assert out.authenticated
    assert out.n_results == 1
    assert not out.from_cache
    assert fake_mp.calls == ["summary.search"]


def test_auth_check_maps_auth_failure_without_retry(
    fake_mp: FakeMPApi, tmp_path: Path, env_file: Path
) -> None:
    fake_mp.errors = [Exception("401 Unauthorized")] * 3
    out, _ = run_skill(tmp_path, env_file, operation="auth_check")
    assert not out.authenticated
    assert "auth" in out.detail
    assert "non-retryable" in out.detail
    assert fake_mp.calls == ["summary.search"]  # auth failures do not retry
    assert SYNTHETIC_KEY not in out.detail


def test_auth_check_maps_rate_limit_as_retryable(
    fake_mp: FakeMPApi, tmp_path: Path, env_file: Path
) -> None:
    fake_mp.errors = [Exception("429 Too Many Requests")] * 5
    out, _ = run_skill(tmp_path, env_file, operation="auth_check")
    assert not out.authenticated
    assert "rate_limit" in out.detail
    assert "resource_exhausted" in out.detail
    assert fake_mp.calls == ["summary.search"] * 3  # bounded at 3 attempts


def test_summary_search_sanitizes_orders_and_registers(
    fake_mp: FakeMPApi, tmp_path: Path, env_file: Path
) -> None:
    fake_mp.summary_docs = [
        SimpleNamespace(
            material_id="mp-30",
            formula_pretty="Cu",
            energy_above_hull=0.0,
            builder_meta={"private": "unrequested"},
            api_request_header="never-store-me",
        ),
        SimpleNamespace(
            material_id="mp-1000",
            formula_pretty="Cu2O",
            energy_above_hull=0.01,
        ),
    ]
    out, ctx = run_skill(
        tmp_path,
        env_file,
        operation="summary_search",
        chemsys="Cu",
        fields=["material_id", "formula_pretty", "energy_above_hull"],
    )
    assert out.n_results == 2
    assert not out.from_cache
    assert out.authenticated
    # Deterministic lexicographic ordering by material_id.
    assert out.material_ids == ["mp-1000", "mp-30"]

    kinds = {a.kind for a in ctx.registry.all()}
    assert {"mp_response", "mp_provenance"} <= kinds
    response = json.loads((ctx.step_dir / "response.json").read_text())
    docs = response["documents"]
    assert [d["material_id"] for d in docs] == ["mp-1000", "mp-30"]
    # Exactly the requested fields survive sanitization.
    assert all(
        set(d) == {"material_id", "formula_pretty", "energy_above_hull"} for d in docs
    )
    provenance = json.loads((ctx.step_dir / "provenance.json").read_text())
    assert provenance["query"]["chemsys"] == "Cu"
    assert provenance["material_ids"] == ["mp-1000", "mp-30"]
    assert len(provenance["response_sha256"]) == 64
    assert "mp_api_client_version" in provenance
    # No key material in any written byte (run dir and cache).
    assert_no_key_material(tmp_path)


def test_get_structures_converts_fcc_cu_to_structure_set(
    fake_mp: FakeMPApi, tmp_path: Path, env_file: Path
) -> None:
    pytest.importorskip("pymatgen")
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure

    fractional_coords: list[list[float]] = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]
    fcc_cu = Structure(
        Lattice.cubic(3.6),
        ["Cu", "Cu", "Cu", "Cu"],
        fractional_coords,
    )
    fake_mp.structures = {"mp-30": fcc_cu}
    out, ctx = run_skill(
        tmp_path, env_file, operation="get_structures", material_ids=["mp-30"]
    )
    assert out.n_results == 1
    assert out.material_ids == ["mp-30"]
    kinds = {a.kind for a in ctx.registry.all()}
    assert {"mp_structures", "mp_provenance"} <= kinds

    structure_set = StructureSet.load(ctx.step_dir / "structures.json")
    record = structure_set.systems[0]
    assert record.symbols == ["Cu"] * 4
    assert record.perturbation_scale == 0.0
    assert record.pbc == [True, True, True]
    assert record.cell[0][0] == pytest.approx(3.6)
    assert_no_key_material(tmp_path)


def test_cache_hit_skips_network_and_still_registers(
    fake_mp: FakeMPApi, tmp_path: Path, env_file: Path
) -> None:
    fake_mp.summary_docs = [SimpleNamespace(material_id="mp-30", formula_pretty="Cu")]
    common = {
        "operation": "summary_search",
        "chemsys": "Cu",
        "fields": ["material_id", "formula_pretty"],
    }
    out1, _ = run_skill(tmp_path, env_file, step_id="first", **common)
    assert not out1.from_cache
    assert fake_mp.calls == ["summary.search"]

    out2, ctx2 = run_skill(tmp_path, env_file, step_id="second", **common)
    assert out2.from_cache
    assert not out2.authenticated  # no live call was made
    assert fake_mp.calls == ["summary.search"]  # unchanged: network skipped
    assert out2.material_ids == out1.material_ids
    kinds = {a.kind for a in ctx2.registry.all()}
    assert {"mp_response", "mp_provenance"} <= kinds

    out3, _ = run_skill(tmp_path, env_file, step_id="third", use_cache=False, **common)
    assert not out3.from_cache
    assert fake_mp.calls == ["summary.search", "summary.search"]
    assert_no_key_material(tmp_path)


def test_retry_loop_recovers_from_transient_errors(
    fake_mp: FakeMPApi, tmp_path: Path, env_file: Path
) -> None:
    fake_mp.errors = [ConnectionError("boom"), TimeoutError("slow")]
    fake_mp.summary_docs = [SimpleNamespace(material_id="mp-30", formula_pretty="Cu")]
    out, _ = run_skill(
        tmp_path,
        env_file,
        operation="summary_search",
        fields=["material_id", "formula_pretty"],
    )
    assert out.n_results == 1
    assert fake_mp.calls == ["summary.search"] * 3


def test_retry_loop_stops_at_three_attempts(
    fake_mp: FakeMPApi, tmp_path: Path, env_file: Path
) -> None:
    fake_mp.errors = [ConnectionError("boom")] * 5
    with pytest.raises(SkillError) as excinfo:
        run_skill(
            tmp_path,
            env_file,
            operation="summary_search",
            fields=["material_id", "formula_pretty"],
        )
    err = excinfo.value
    assert err.failure_class is FailureClass.TOOL_ERROR
    assert err.retryable
    assert fake_mp.calls == ["summary.search"] * 3
    assert SYNTHETIC_KEY not in str(err)


@pytest.mark.parametrize(
    "overrides",
    [
        {"operation": "summary_search", "chemsys": "cu"},
        {"operation": "summary_search", "fields": []},
        {"operation": "summary_search", "fields": ["formula_pretty"]},  # no material_id
        {"operation": "get_structures", "material_ids": []},
        {"operation": "get_structures", "material_ids": ["mvc-30"]},
    ],
)
def test_invalid_inputs_rejected(
    fake_mp: FakeMPApi, tmp_path: Path, env_file: Path, overrides: dict[str, object]
) -> None:
    with pytest.raises(SkillError) as excinfo:
        run_skill(tmp_path, env_file, **overrides)  # type: ignore[arg-type]
    assert excinfo.value.failure_class is FailureClass.VALIDATION_ERROR
    assert not excinfo.value.retryable
    assert fake_mp.calls == []
