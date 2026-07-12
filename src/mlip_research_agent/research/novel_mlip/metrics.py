"""Independent aggregate and tail metrics for the frozen Cu campaign."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mlip_research_agent.data.manifests import LabeledConfiguration


def mean_predictions(
    predictions: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    if len(predictions) < 1:
        raise ValueError("cannot average an empty prediction collection")
    ids = set(predictions[0])
    if any(set(member) != ids for member in predictions):
        raise ValueError("ensemble prediction coverage differs across members")
    output: dict[str, dict[str, Any]] = {}
    for record_id in sorted(ids):
        energies = np.asarray(
            [member[record_id]["energy_ev"] for member in predictions], dtype=float
        )
        forces = np.asarray(
            [member[record_id]["forces_ev_per_a"] for member in predictions],
            dtype=float,
        )
        output[record_id] = {
            "energy_ev": float(energies.mean()),
            "forces_ev_per_a": forces.mean(axis=0).tolist(),
        }
    return output


def evaluate_predictions(
    records: list[LabeledConfiguration],
    predictions: dict[str, dict[str, Any]],
    *,
    high_error_threshold_ev_per_a: float = 0.5,
) -> dict[str, Any]:
    expected = {record.config_id for record in records}
    if set(predictions) != expected:
        raise ValueError("prediction coverage does not match evaluation records")
    energy_errors: list[float] = []
    component_errors: list[float] = []
    vector_errors: list[float] = []
    structure_mean_vector_errors: list[float] = []
    magnitude_biases: list[float] = []
    slope_numerator = 0.0
    slope_denominator = 0.0
    per_record_vector_errors: dict[str, list[float]] = {}
    for record in records:
        prediction = predictions[record.config_id]
        predicted_forces = np.asarray(prediction["forces_ev_per_a"], dtype=float)
        reference_forces = np.asarray(record.forces_ev_per_a, dtype=float)
        if predicted_forces.shape != reference_forces.shape:
            raise ValueError(f"force shape mismatch for {record.config_id}")
        difference = predicted_forces - reference_forces
        local_vectors = np.linalg.norm(difference, axis=1)
        component_errors.extend(np.abs(difference).ravel().tolist())
        vector_errors.extend(local_vectors.tolist())
        per_record_vector_errors[record.config_id] = local_vectors.tolist()
        structure_mean_vector_errors.append(float(local_vectors.mean()))
        energy_errors.append(
            abs(float(prediction["energy_ev"]) - record.energy_ev) / record.n_atoms
        )
        magnitude_biases.extend(
            (
                np.linalg.norm(predicted_forces, axis=1)
                - np.linalg.norm(reference_forces, axis=1)
            ).tolist()
        )
        weight = 1.0 / (3.0 * record.n_atoms)
        slope_numerator += weight * float(np.sum(reference_forces * predicted_forces))
        slope_denominator += weight * float(np.sum(reference_forces * reference_forces))
    components = np.asarray(component_errors)
    vectors = np.asarray(vector_errors)
    beta = slope_numerator / slope_denominator if slope_denominator > 0 else math.nan
    return {
        "n_structures": len(records),
        "n_atoms": sum(record.n_atoms for record in records),
        "energy_mae_ev_per_atom": float(np.mean(energy_errors)),
        "force_component_mae_ev_per_a": float(np.mean(components)),
        "force_component_rmse_ev_per_a": float(np.sqrt(np.mean(components**2))),
        "force_vector_error_p95_ev_per_a": float(
            np.quantile(vectors, 0.95, method="linear")
        ),
        "high_error_structure_fraction": float(
            np.mean(
                np.asarray(structure_mean_vector_errors)
                > high_error_threshold_ev_per_a
            )
        ),
        "force_softening_beta": float(beta),
        "force_softening_one_minus_beta": float(1.0 - beta),
        "median_force_magnitude_bias_ev_per_a": float(np.median(magnitude_biases)),
        "per_record_force_vector_errors_ev_per_a": per_record_vector_errors,
    }


def max_prediction_difference(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError("prediction coverage differs")
    energy_max = 0.0
    force_max = 0.0
    for record_id in left:
        energy_max = max(
            energy_max,
            abs(float(left[record_id]["energy_ev"]) - float(right[record_id]["energy_ev"])),
        )
        force_max = max(
            force_max,
            float(
                np.max(
                    np.abs(
                        np.asarray(left[record_id]["forces_ev_per_a"])
                        - np.asarray(right[record_id]["forces_ev_per_a"])
                    )
                )
            ),
        )
    return {
        "maximum_absolute_energy_difference_ev": energy_max,
        "maximum_absolute_force_component_difference_ev_per_a": force_max,
    }
