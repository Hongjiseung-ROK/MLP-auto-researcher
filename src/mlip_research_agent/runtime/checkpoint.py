"""Step-level checkpointing so interrupted runs resume without recomputation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

CHECKPOINT_NAME = "checkpoint.json"


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    workflow_hash: str
    seed: int
    completed_steps: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="step_id -> validated output payload"
    )

    def save(self, run_dir: Path) -> Path:
        path = run_dir / CHECKPOINT_NAME
        path.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, run_dir: Path) -> Checkpoint | None:
        path = run_dir / CHECKPOINT_NAME
        if not path.is_file():
            return None
        return cls.model_validate_json(path.read_text())
