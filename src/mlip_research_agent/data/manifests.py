"""Labeled-configuration schema and deterministic dataset serialization.

First-party JSON is the provenance format (byte-stable across library
versions); extended XYZ is generated only as a MACE interchange *export*,
with its hash recorded (docs/PHASE2_DECISIONS.md D-P2-3). Units are explicit
fields, never inferred.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_FORMAT_VERSION = "2.0"

ENERGY_UNIT = "eV"
FORCE_UNIT = "eV/angstrom"
STRESS_UNIT = "kbar_voigt6"


class LabeledConfiguration(BaseModel):
    """One atomic configuration with trusted DFT labels and full lineage."""

    model_config = ConfigDict(extra="forbid")

    config_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    source_id: str = Field(min_length=1, description="e.g. 'mlearn_cu@<commit>/training.json[3]'")
    top_group: str = Field(min_length=1, description="Generation family, e.g. 'Elastic'")
    group_id: str = Field(min_length=1, description="Fine-grained subgroup, e.g. 'aimd_nvt_300K'")
    split_unit_id: str = Field(
        min_length=1,
        description="Atomic unit of splitting; no unit may cross a partition boundary",
    )
    symbols: list[str] = Field(min_length=1)
    positions: list[list[float]]
    cell: list[list[float]]
    pbc: list[bool]
    energy_ev: float
    forces_ev_per_a: list[list[float]]
    virial_stress_kbar: list[float] | None = None
    level_of_theory: str = Field(min_length=1)
    source_record_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _shapes(self) -> LabeledConfiguration:
        n = len(self.symbols)
        if len(self.positions) != n or any(len(p) != 3 for p in self.positions):
            raise ValueError(f"{self.config_id}: positions must be {n}x3")
        if len(self.forces_ev_per_a) != n or any(len(f) != 3 for f in self.forces_ev_per_a):
            raise ValueError(f"{self.config_id}: forces must be {n}x3")
        if len(self.cell) != 3 or any(len(row) != 3 for row in self.cell):
            raise ValueError(f"{self.config_id}: cell must be 3x3")
        if len(self.pbc) != 3:
            raise ValueError(f"{self.config_id}: pbc must have 3 entries")
        if self.virial_stress_kbar is not None and len(self.virial_stress_kbar) != 6:
            raise ValueError(f"{self.config_id}: virial stress must be 6-component Voigt")
        return self

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    def content_hash(self) -> str:
        """Hash of the physical content + labels (id and lineage excluded),
        used both as identity and for duplicate detection."""
        payload = {
            "symbols": self.symbols,
            "positions": self.positions,
            "cell": self.cell,
            "pbc": self.pbc,
            "energy_ev": self.energy_ev,
            "forces_ev_per_a": self.forces_ev_per_a,
            "virial_stress_kbar": self.virial_stress_kbar,
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


class DatasetUnits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    energy: str = ENERGY_UNIT
    forces: str = FORCE_UNIT
    virial_stress: str = STRESS_UNIT
    virial_stress_caveat: str = (
        "kbar inferred from magnitude comparison with Cu elastic constants; "
        "not documented by the source (H1 open item)"
    )


class NormalizedDataset(BaseModel):
    """The qualified, unit-explicit dataset registry format."""

    model_config = ConfigDict(extra="forbid")

    format_version: str = DATASET_FORMAT_VERSION
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    level_of_theory: str
    units: DatasetUnits = DatasetUnits()
    configurations: list[LabeledConfiguration] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> NormalizedDataset:
        ids = [c.config_id for c in self.configurations]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate config_ids: {dupes[:5]}")
        return self

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def save(self, path: Path) -> str:
        """Write deterministic JSON; returns the file's SHA-256."""
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=1) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> NormalizedDataset:
        return cls.model_validate(json.loads(path.read_text()))

    def by_id(self) -> dict[str, LabeledConfiguration]:
        return {c.config_id: c for c in self.configurations}


def to_extxyz(configs: list[LabeledConfiguration]) -> str:
    """Deterministic extended-XYZ export for mace-torch consumption.

    Fixed-precision formatting (not repr) so the emitted bytes are stable;
    the caller records this string's SHA-256 as the training-input hash.
    Labels use REF_energy / REF_forces keys: the ASE reader leaves those in
    info/arrays (a plain `energy` key would be captured by a calculator
    object), and MACE training consumes them via explicit
    --energy_key/--forces_key. Stress is intentionally omitted until its
    unit is H1-confirmed.
    """
    lines: list[str] = []
    for cfg in configs:
        lines.append(str(cfg.n_atoms))
        cell9 = " ".join(f"{x:.10f}" for row in cfg.cell for x in row)
        pbc_str = " ".join("T" if p else "F" for p in cfg.pbc)
        lines.append(
            f'Lattice="{cell9}" '
            f"Properties=species:S:1:pos:R:3:REF_forces:R:3 "
            f"REF_energy={cfg.energy_ev:.10f} "
            f'pbc="{pbc_str}" '
            f"config_id={cfg.config_id}"
        )
        for sym, pos, force in zip(cfg.symbols, cfg.positions, cfg.forces_ev_per_a, strict=True):
            coords = " ".join(f"{x:.10f}" for x in pos)
            forces = " ".join(f"{x:.10f}" for x in force)
            lines.append(f"{sym} {coords} {forces}")
    return "\n".join(lines) + "\n"


def write_extxyz(configs: list[LabeledConfiguration], path: Path) -> str:
    text = to_extxyz(configs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
