"""Typed inputs/outputs for the mock literature skill."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    title: str
    venue: str
    year: int = Field(ge=1900, le=2100)
    relevance: str


class LiteratureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)


class LiteratureOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    references_artifact: str
    references_path: str
    n_references: int = Field(ge=0)
