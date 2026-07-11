"""Typed I/O for the phonopy displacement-generation adapter."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PhonopyDisplacementsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_path: str = Field(description="Run-dir-relative StructureSet JSON")
    structure_index: int = Field(default=0, ge=0)
    supercell: list[int] = Field(min_length=3, max_length=3)
    displacement_distance_a: float = Field(default=0.01, gt=0, le=0.2)

    @model_validator(mode="after")
    def _supercell_bounds(self) -> PhonopyDisplacementsInput:
        if any(n < 1 or n > 4 for n in self.supercell):
            raise ValueError("supercell entries must be within [1, 4]")
        return self


class PhonopyDisplacementsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_artifact: str
    structures_path: str
    provenance_artifact: str
    provenance_path: str
    n_displacements: int = Field(gt=0)
    n_atoms_per_supercell: int = Field(gt=0)
