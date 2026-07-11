"""Typed I/O for the RDKit conformer-generation adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RDKitConformerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smiles: str = Field(min_length=1, max_length=500)
    n_conformers: int = Field(default=1, ge=1, le=10)
    optimize: Literal["none", "mmff94"] = "none"


class RDKitConformerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    molecule_artifact: str
    molecule_path: str
    canonical_smiles: str
    n_atoms: int = Field(gt=0)
    n_conformers: int = Field(gt=0)
