"""Outcome-blind acquisition policies for the frozen Cu campaign."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlip_research_agent.skills.active_learning.diversity_select.implementation import (
    farthest_point_selection,
)
from mlip_research_agent.skills.active_learning.mace_descriptors.implementation import (
    standardize_descriptors,
)
from mlip_research_agent.skills.active_learning.tail_risk_score.implementation import (
    force_tail_proxy,
)


def committee_scores(
    member_predictions: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    if len(member_predictions) != 3:
        raise ValueError("frozen committee size is exactly three")
    ids = set(member_predictions[0])
    if any(set(member) != ids for member in member_predictions):
        raise ValueError("committee prediction coverage mismatch")
    scores: dict[str, dict[str, float]] = {}
    for record_id in sorted(ids):
        forces = np.asarray(
            [member[record_id]["forces_ev_per_a"] for member in member_predictions],
            dtype=float,
        )
        mean_forces = forces.mean(axis=0)
        disagreement = np.sqrt(np.mean(np.sum((forces - mean_forces) ** 2, axis=2), axis=0))
        tail_score, tail_count, _, magnitude, _ = force_tail_proxy(
            forces,
            tail_fraction=0.05,
            magnitude_weight=1.0,
            disagreement_weight=1.0,
        )
        scores[record_id] = {
            "mean_population_vector_disagreement_ev_per_a": float(disagreement.mean()),
            "max_population_vector_disagreement_ev_per_a": float(disagreement.max()),
            "force_tail_proxy_ev_per_a": tail_score,
            "tail_atom_count": float(tail_count),
            "mean_predicted_force_magnitude_ev_per_a": float(
                np.linalg.norm(mean_forces, axis=1).mean()
            ),
            "mean_atomic_magnitude_from_proxy_ev_per_a": float(magnitude.mean()),
        }
    return scores


def _fps(
    shortlist: list[str],
    normalization_ids: list[str],
    raw_descriptors: dict[str, list[float]],
    budget: int,
) -> tuple[list[str], dict[str, Any]]:
    normalized, raw_dimension, dimension, retained = standardize_descriptors(
        {
            record_id: np.asarray(raw_descriptors[record_id])
            for record_id in normalization_ids
        },
        min_feature_std=1.0e-12,
        l2_normalize=True,
    )
    ids = sorted(shortlist)
    matrix = np.asarray([normalized[record_id] for record_id in ids], dtype=float)
    picks = farthest_point_selection(ids, matrix, budget)
    return [record_id for record_id, _ in picks], {
        "raw_dimension": raw_dimension,
        "normalized_dimension": dimension,
        "retained_feature_indices": retained,
        "fps_distances": {record_id: distance for record_id, distance in picks},
    }


def select_candidates(
    *,
    policy: str,
    remaining_ids: list[str],
    scores: dict[str, dict[str, float]],
    raw_descriptors: dict[str, list[float]],
    round_index: int,
) -> tuple[list[str], dict[str, Any]]:
    budget = 12
    if len(remaining_ids) < budget:
        raise ValueError("remaining acquisition pool is below the frozen round budget")
    if policy == "random":
        random_seeds = {1: 31001, 2: 31002, 3: 31003}
        generator = np.random.default_rng(random_seeds[round_index])
        selected = sorted(
            str(value)
            for value in generator.choice(sorted(remaining_ids), size=budget, replace=False)
        )
        return selected, {"random_seed": random_seeds[round_index]}
    score_field = (
        "force_tail_proxy_ev_per_a"
        if policy == "tail_risk_fps"
        else "mean_population_vector_disagreement_ev_per_a"
    )
    ordered = sorted(
        remaining_ids,
        key=lambda record_id: (-scores[record_id][score_field], record_id),
    )
    if policy == "disagreement":
        selected = sorted(ordered[:budget])
        return selected, {"score_field": score_field, "shortlist_size": budget}
    if policy not in {"disagreement_fps", "tail_risk_fps"}:
        raise ValueError(f"unknown frozen acquisition policy: {policy}")
    shortlist = ordered[:36]
    selected_in_order, details = _fps(
        shortlist, remaining_ids, raw_descriptors, budget
    )
    return sorted(selected_in_order), {
        "score_field": score_field,
        "shortlist_size": 36,
        "shortlist_record_ids": shortlist,
        "fps_selection_order": selected_in_order,
        **details,
    }


def viability_summary(
    scores: dict[str, dict[str, float]], numerical_floor: float
) -> dict[str, Any]:
    values = np.asarray(
        [
            record["mean_population_vector_disagreement_ev_per_a"]
            for record in scores.values()
        ]
    )
    rounded = np.round(values / 1.0e-12).astype(np.int64)
    counts = np.unique(rounded, return_counts=True)[1]
    tied_fraction = float(counts[counts > 1].sum() / len(values)) if len(values) else 1.0
    threshold = max(10.0 * numerical_floor, 1.0e-6)
    median = float(np.median(values))
    return {
        "median_disagreement_ev_per_a": median,
        "required_minimum_ev_per_a": threshold,
        "fraction_in_score_ties_at_1e-12": tied_fraction,
        "maximum_tied_fraction": 0.2,
        "passed": median > threshold and tied_fraction <= 0.2,
    }
