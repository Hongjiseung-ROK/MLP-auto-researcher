"""Deterministic dataset qualification: every check is machine-readable and
fail-closed. A dataset becomes claim-eligible only after this report passes
AND the H1 human approval is recorded against the normalized manifest hash.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mlip_research_agent.data.manifests import LabeledConfiguration, NormalizedDataset

QUALIFICATION_REPORT_NAME = "qualification_report.json"


class QualificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str = ""


class QualificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    n_configurations: int = Field(ge=0)
    group_histogram: dict[str, int]
    split_unit_count: int = Field(ge=0)
    energy_per_atom_min: float
    energy_per_atom_max: float
    force_magnitude_max: float
    exact_duplicate_count: int = Field(ge=0)
    near_duplicate_count: int = Field(ge=0)
    checks: list[QualificationCheck]
    passed: bool

    def save(self, path: Path) -> str:
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> QualificationReport:
        return cls.model_validate(json.loads(path.read_text()))


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def _det3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _near_duplicate_key(cfg: LabeledConfiguration) -> str:
    """Rounded-content key: catches re-serialized copies of the same frame."""
    payload = {
        "symbols": cfg.symbols,
        "positions": [[round(x, 3) for x in row] for row in cfg.positions],
        "cell": [[round(x, 3) for x in row] for row in cfg.cell],
        "energy": round(cfg.energy_ev, 4),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def qualify_dataset(
    dataset: NormalizedDataset,
    *,
    allowed_species: frozenset[str] = frozenset({"Cu"}),
    energy_per_atom_window: tuple[float, float] = (-4.5, -3.0),
    max_force_magnitude: float = 50.0,
    expected_total: int | None = None,
    expected_group_histogram: dict[str, int] | None = None,
) -> QualificationReport:
    configs = dataset.configurations
    checks: list[QualificationCheck] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append(QualificationCheck(name=name, passed=passed, detail=detail))

    bad_species = sorted(
        {s for c in configs for s in c.symbols if s not in allowed_species}
    )
    check("species_allowlisted", not bad_species, f"unexpected species: {bad_species}")

    nonfinite = [
        c.config_id
        for c in configs
        if not (
            math.isfinite(c.energy_ev)
            and all(_finite(row) for row in c.forces_ev_per_a)
            and (c.virial_stress_kbar is None or _finite(c.virial_stress_kbar))
        )
    ]
    check("labels_finite", not nonfinite, f"non-finite labels in: {nonfinite[:5]}")

    singular = [c.config_id for c in configs if abs(_det3(c.cell)) < 1.0]
    check("cells_nonsingular", not singular, f"near-singular cells: {singular[:5]}")

    non_periodic = [c.config_id for c in configs if not all(c.pbc)]
    check("fully_periodic", not non_periodic, f"non-periodic configs: {non_periodic[:5]}")

    epa = [c.energy_ev / c.n_atoms for c in configs]
    epa_min, epa_max = min(epa), max(epa)
    lo, hi = energy_per_atom_window
    check(
        "energy_per_atom_in_window",
        lo <= epa_min and epa_max <= hi,
        f"observed [{epa_min:.4f}, {epa_max:.4f}] eV/atom vs window [{lo}, {hi}]",
    )

    fmax = max(
        math.sqrt(fx * fx + fy * fy + fz * fz)
        for c in configs
        for fx, fy, fz in c.forces_ev_per_a
    )
    check(
        "force_magnitude_bounded",
        fmax <= max_force_magnitude,
        f"max |F| = {fmax:.4f} eV/A vs bound {max_force_magnitude}",
    )

    content_counts = Counter(c.content_hash() for c in configs)
    exact_dupes = sum(n - 1 for n in content_counts.values() if n > 1)
    check("no_exact_duplicates", exact_dupes == 0, f"{exact_dupes} exact duplicate records")

    near_counts = Counter(_near_duplicate_key(c) for c in configs)
    near_dupes = sum(n - 1 for n in near_counts.values() if n > 1) - exact_dupes
    check(
        "near_duplicates_reported",
        True,
        f"{near_dupes} near-duplicates beyond exact duplicates (informational)",
    )

    if expected_total is not None:
        check(
            "expected_total_count",
            len(configs) == expected_total,
            f"parsed {len(configs)} vs due-diligence count {expected_total}",
        )

    group_histogram = dict(sorted(Counter(c.top_group for c in configs).items()))
    if expected_group_histogram is not None:
        check(
            "expected_group_histogram",
            group_histogram == dict(sorted(expected_group_histogram.items())),
            f"parsed {group_histogram} vs expected {expected_group_histogram}",
        )

    units_declared = bool(
        dataset.units.energy and dataset.units.forces and dataset.level_of_theory
    )
    check("units_and_level_of_theory_declared", units_declared)

    return QualificationReport(
        dataset_id=dataset.dataset_id,
        n_configurations=len(configs),
        group_histogram=group_histogram,
        split_unit_count=len({c.split_unit_id for c in configs}),
        energy_per_atom_min=epa_min,
        energy_per_atom_max=epa_max,
        force_magnitude_max=fmax,
        exact_duplicate_count=exact_dupes,
        near_duplicate_count=near_dupes,
        checks=checks,
        passed=all(c.passed for c in checks),
    )
