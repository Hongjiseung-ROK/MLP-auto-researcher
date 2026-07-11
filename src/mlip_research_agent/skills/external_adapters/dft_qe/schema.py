"""Typed I/O for the prepare-only Quantum ESPRESSO adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PSEUDO_FILENAME_PATTERN = r"^[A-Za-z0-9._-]+$"


class QEPrepareInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_path: str = Field(description="Run-dir-relative StructureSet JSON")
    structure_index: int = Field(default=0, ge=0)
    calculation: Literal["scf"] = "scf"
    ecutwfc_ry: float = Field(gt=0, le=400)
    ecutrho_ry: float | None = Field(default=None, gt=0, le=4000)
    kpoints: list[int] = Field(min_length=3, max_length=3)
    pseudopotentials: dict[str, str] = Field(
        min_length=1, description="Species symbol to UPF filename"
    )
    mode: Literal["prepare"] = "prepare"

    @model_validator(mode="after")
    def _consistency(self) -> QEPrepareInput:
        import re

        if any(k < 1 or k > 32 for k in self.kpoints):
            raise ValueError("kpoints entries must be within [1, 32]")
        if self.ecutrho_ry is not None and self.ecutrho_ry < 4 * self.ecutwfc_ry:
            raise ValueError("ecutrho_ry must be at least 4 * ecutwfc_ry")
        for symbol, filename in self.pseudopotentials.items():
            if not re.fullmatch(r"[A-Z][a-z]?", symbol):
                raise ValueError(f"invalid species symbol: {symbol!r}")
            if not re.fullmatch(PSEUDO_FILENAME_PATTERN, filename):
                raise ValueError(f"unsafe pseudopotential filename: {filename!r}")
        return self


class QEPrepareOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_deck_artifact: str
    input_deck_path: str
    provenance_artifact: str
    provenance_path: str
    n_atoms: int = Field(gt=0)
    n_species: int = Field(gt=0)
