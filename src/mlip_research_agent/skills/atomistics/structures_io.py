"""Deterministic JSON serialization for small structure sets.

A first-party format (rather than extxyz) so that artifact bytes are stable
across library versions and reruns; ASE Atoms conversion is provided both ways.
"""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field

FORMAT_VERSION = "0.1"


class StructureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    symbols: list[str]
    positions: list[list[float]]
    cell: list[list[float]]
    pbc: list[bool]
    perturbation_scale: float = Field(ge=0.0)


class StructureSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_version: str = FORMAT_VERSION
    systems: list[StructureRecord]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.model_dump(), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> StructureSet:
        return cls.model_validate(json.loads(path.read_text()))


def record_to_atoms(record: StructureRecord) -> Atoms:
    return Atoms(
        symbols=record.symbols,
        positions=record.positions,
        cell=record.cell,
        pbc=record.pbc,
    )


def atoms_to_record(atoms: Atoms, index: int, perturbation_scale: float) -> StructureRecord:
    return StructureRecord(
        index=index,
        symbols=list(atoms.get_chemical_symbols()),
        positions=[[float(x) for x in row] for row in atoms.get_positions()],
        cell=[[float(x) for x in row] for row in atoms.get_cell()],
        pbc=[bool(p) for p in atoms.get_pbc()],
        perturbation_scale=perturbation_scale,
    )
