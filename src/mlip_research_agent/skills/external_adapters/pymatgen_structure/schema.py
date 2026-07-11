"""Typed I/O for the pymatgen symmetry-analysis adapter."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PymatgenSymmetryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_path: str = Field(description="Run-dir-relative StructureSet JSON")
    symprec: float = Field(default=1e-3, gt=0, le=1.0)


class PymatgenSymmetryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_artifact: str
    report_path: str
    n_structures: int = Field(gt=0)
