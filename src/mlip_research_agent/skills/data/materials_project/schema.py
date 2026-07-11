"""Typed inputs/outputs for the materials_project_recon skill."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Safe default field set: identity + stability + symmetry metadata only.
DEFAULT_FIELDS = ("material_id", "formula_pretty", "energy_above_hull", "symmetry")


class MaterialsProjectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["auth_check", "summary_search", "get_structures"]
    chemsys: str = "Cu"
    material_ids: list[str] = Field(default_factory=list)
    fields: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FIELDS),
        description="Only these fields are fetched and stored; everything else is dropped.",
    )
    max_results: int = Field(default=100, ge=1, le=500)
    use_cache: bool = True
    # Dev knobs (tests): override the on-disk cache root / the api.env location.
    # Neither changes what data is stored; cache files hold sanitized fields only.
    cache_root: str | None = Field(default=None, description="dev knob: sanitized cache dir")
    api_env_file: str | None = Field(default=None, description="dev knob: alternate api.env path")


class MaterialsProjectOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str
    #: True iff an authenticated API call succeeded in this run (False on cache hits).
    authenticated: bool
    n_results: int = Field(ge=0)
    material_ids: list[str]
    response_artifact: str | None
    provenance_artifact: str | None
    from_cache: bool
    detail: str  # static/derived guidance text only; never key material or raw headers
