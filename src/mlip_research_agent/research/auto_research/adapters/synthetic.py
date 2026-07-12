"""Deterministic, non-claim-bearing synthetic benchmark adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from mlip_research_agent.research.auto_research.adapters.base import AdapterRunResult
from mlip_research_agent.research.auto_research.validators import ConfigValue

FIXTURE_NAME = "synthetic_quadratic_finetune_fixture"
FIXTURE_VERSION = "1.0.0"
N_SYNTHETIC_RECORDS = 32
DIVERGENCE_LEARNING_RATE = 0.05


def _as_float(config: dict[str, ConfigValue], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"fixture config key {key!r} must be numeric, got {value!r}")
    return float(value)


def synthetic_validation_residuals(config: dict[str, ConfigValue], seed: int) -> list[float]:
    lr = _as_float(config, "learning_rate")
    clip = _as_float(config, "gradient_clip")
    force_w = _as_float(config, "force_loss_weight")
    energy_w = _as_float(config, "energy_loss_weight")
    patience = _as_float(config, "scheduler_patience")
    base = 0.05 + 0.4 * (math.log10(lr) - math.log10(2e-3)) ** 2
    base += 0.01 * (math.log10(clip) - math.log10(0.5)) ** 2
    base += 0.005 * (math.log10(force_w) - math.log10(20.0)) ** 2
    base += 0.002 * (math.log10(energy_w) - math.log10(1.0)) ** 2
    base += 0.001 * (patience - 5.0) ** 2 / 25.0
    rng = np.random.default_rng([seed, N_SYNTHETIC_RECORDS])
    noise = rng.normal(loc=1.0, scale=0.05, size=N_SYNTHETIC_RECORDS)
    return [float(base * max(n, 0.5)) for n in noise]


class SyntheticQuadraticAdapter:
    @property
    def name(self) -> str:
        return f"{FIXTURE_NAME}/{FIXTURE_VERSION}"

    @property
    def remote(self) -> bool:
        return False

    def execute(self, config: dict[str, ConfigValue], seed: int, workdir: Path) -> AdapterRunResult:
        workdir.mkdir(parents=True, exist_ok=True)
        batch_size = _as_float(config, "batch_size")
        simulated_wall = 5.0 + 0.5 * batch_size
        simulated_memory = 100.0 + 20.0 * batch_size
        definition = {
            "fixture": FIXTURE_NAME,
            "version": FIXTURE_VERSION,
            "n_records": N_SYNTHETIC_RECORDS,
            "scientific_status": "non_scientific",
            "claim_eligible": False,
            "uses_frozen_test_data": False,
            "uses_hidden_labels": False,
            "uses_protected_partitions": False,
            "uses_acquisition_pool_labels": False,
            "config": dict(sorted(config.items())),
            "seed": seed,
        }
        definition_path = workdir / "fixture_definition.json"
        definition_path.write_text(json.dumps(definition, indent=2, sort_keys=True) + "\n")
        lr = _as_float(config, "learning_rate")
        if lr > DIVERGENCE_LEARNING_RATE:
            failure = {
                "failure_category": "training_divergence",
                "detail": (
                    f"learning_rate {lr} exceeds the fixture stability region "
                    f"(> {DIVERGENCE_LEARNING_RATE}); simulated NaN loss"
                ),
                "repair_available": True,
                "suggested_repair": {"learning_rate": lr * 0.5},
            }
            failure_path = workdir / "failure.json"
            failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
            return AdapterRunResult(
                succeeded=False,
                fixture_definition_path=str(definition_path),
                predictions_path=None,
                predictions_rerun_path=None,
                failure_path=str(failure_path),
                failure_category="training_divergence",
                repair_available=True,
                simulated_wall_seconds=simulated_wall,
                simulated_peak_memory_mb=simulated_memory,
            )
        residuals = synthetic_validation_residuals(config, seed)
        predictions = {
            "schema_version": "1.0.0",
            "scientific_status": "non_scientific",
            "record_residuals": {f"synthetic-{i:04d}": r for i, r in enumerate(residuals)},
        }
        predictions_path = workdir / "predictions.json"
        rerun_path = workdir / "predictions_rerun.json"
        text = json.dumps(predictions, indent=2, sort_keys=True) + "\n"
        predictions_path.write_text(text)
        rerun_path.write_text(text)
        return AdapterRunResult(
            succeeded=True,
            fixture_definition_path=str(definition_path),
            predictions_path=str(predictions_path),
            predictions_rerun_path=str(rerun_path),
            failure_path=None,
            failure_category=None,
            repair_available=False,
            simulated_wall_seconds=simulated_wall,
            simulated_peak_memory_mb=simulated_memory,
        )
