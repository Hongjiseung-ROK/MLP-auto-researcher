"""Typed I/O for the prepare-only LAMMPS (DeepMD pair style) adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LAMMPSPrepareInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_path: str = Field(description="Run-dir-relative StructureSet JSON")
    structure_index: int = Field(default=0, ge=0)
    pair_style: Literal["deepmd"] = "deepmd"
    model_filename: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    ensemble: Literal["nve"] = "nve"
    timestep_ps: float = Field(default=0.0005, gt=0, le=0.01)
    n_steps: int = Field(default=0, ge=0, le=1_000_000)
    mode: Literal["prepare"] = "prepare"


class LAMMPSPrepareOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_deck_artifact: str
    input_deck_path: str
    data_file_artifact: str
    data_file_path: str
    provenance_artifact: str
    provenance_path: str
    n_atoms: int = Field(gt=0)
    type_map: dict[str, int]
