"""Typed I/O for the ASE structure-building adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ASEBuildInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element: str = Field(pattern=r"^[A-Z][a-z]?$")
    crystalstructure: Literal["fcc", "bcc", "sc", "hcp", "diamond"]
    lattice_a: float = Field(gt=0, le=20.0, description="Lattice constant in angstrom")
    cubic: bool = True
    repeat: int = Field(default=1, ge=1, le=4)
    n_rattled: int = Field(default=0, ge=0, le=64)
    rattle_stdev_a: float = Field(default=0.0, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def _rattle_consistency(self) -> ASEBuildInput:
        if self.n_rattled > 0 and self.rattle_stdev_a <= 0.0:
            raise ValueError("rattled copies require a positive rattle_stdev_a")
        if self.n_rattled == 0 and self.rattle_stdev_a > 0.0:
            raise ValueError("rattle_stdev_a without n_rattled has no effect")
        if self.crystalstructure == "hcp" and self.cubic:
            raise ValueError("hcp has no cubic conventional cell")
        return self


class ASEBuildOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_artifact: str
    structures_path: str
    provenance_artifact: str
    provenance_path: str
    n_structures: int = Field(gt=0)
    n_atoms_per_structure: int = Field(gt=0)
