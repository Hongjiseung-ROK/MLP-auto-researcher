"""Typed inputs/outputs for the structure-generation skill."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SUPPORTED_LATTICES = ("fcc", "bcc", "sc", "hcp", "diamond")


class StructureGenerationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formula: str = Field(min_length=1, max_length=8, description="Single element, e.g. 'Cu'")
    crystal_structure: str = Field(description=f"One of {SUPPORTED_LATTICES}")
    lattice_constant: float = Field(gt=1.0, lt=20.0, description="Angstrom")
    supercell: tuple[int, int, int] = (2, 2, 2)
    n_candidates: int = Field(gt=1, le=10_000)
    max_perturbation: float = Field(default=0.15, gt=0.0, le=1.0, description="Angstrom stddev")


class StructureGenerationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_artifact: str
    structures_path: str = Field(description="Path relative to the run directory")
    n_structures: int = Field(gt=0)
    n_atoms_per_structure: int = Field(gt=0)
    elements: list[str]
