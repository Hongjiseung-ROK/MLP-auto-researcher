"""Pure contract tests for the frozen real-GPU campaign runtime."""

from __future__ import annotations

from typing import Any

import pytest

from mlip_research_agent.data.manifests import LabeledConfiguration
from mlip_research_agent.research.novel_mlip.metrics import (
    evaluate_predictions,
    max_prediction_difference,
    mean_predictions,
)
from mlip_research_agent.research.novel_mlip.selection import (
    committee_scores,
    select_candidates,
)


def record(record_id: str = "cu-record") -> LabeledConfiguration:
    return LabeledConfiguration(
        config_id=record_id,
        source_id="fixture[0]",
        top_group="AIMD",
        group_id="aimd_nvt_3000k",
        split_unit_id="block-0",
        symbols=["Cu", "Cu"],
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        pbc=[True, True, True],
        energy_ev=-7.0,
        forces_ev_per_a=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        virial_stress_kbar=None,
        level_of_theory="fixture",
        source_record_sha256="a" * 64,
    )


def test_metrics_use_atomic_vector_p95_and_force_softening_slope() -> None:
    target = record()
    predictions = {
        target.config_id: {
            "energy_ev": -6.8,
            "forces_ev_per_a": [[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    }
    metrics = evaluate_predictions([target], predictions)
    assert metrics["energy_mae_ev_per_atom"] == pytest.approx(0.1)
    assert metrics["force_component_mae_ev_per_a"] == pytest.approx(0.25)
    assert metrics["force_vector_error_p95_ev_per_a"] == pytest.approx(0.975)
    assert metrics["force_softening_beta"] == pytest.approx(0.5)
    assert metrics["force_softening_one_minus_beta"] == pytest.approx(0.5)


def test_prediction_mean_and_drift_are_componentwise() -> None:
    left = {"a": {"energy_ev": 1.0, "forces_ev_per_a": [[1.0, 0.0, 0.0]]}}
    right = {"a": {"energy_ev": 3.0, "forces_ev_per_a": [[3.0, 2.0, 0.0]]}}
    mean = mean_predictions([left, right])
    assert mean["a"]["energy_ev"] == pytest.approx(2.0)
    assert mean["a"]["forces_ev_per_a"][0] == pytest.approx([2.0, 1.0, 0.0])
    drift = max_prediction_difference(left, right)
    assert drift["maximum_absolute_energy_difference_ev"] == pytest.approx(2.0)
    assert drift["maximum_absolute_force_component_difference_ev_per_a"] == pytest.approx(
        2.0
    )


def _committee(ids: list[str]) -> list[dict[str, dict[str, Any]]]:
    members: list[dict[str, dict[str, Any]]] = []
    for member_index, scale in enumerate((0.8, 1.0, 1.4)):
        members.append(
            {
                record_id: {
                    "energy_ev": float(index),
                    "forces_ev_per_a": [
                        [scale * (index + 1), 0.1 * member_index, 0.0]
                    ],
                }
                for index, record_id in enumerate(ids)
            }
        )
    return members


def test_committee_scores_and_all_frozen_policies_are_label_free_and_exact_budget() -> None:
    ids = [f"candidate-{index:03d}" for index in range(48)]
    scores = committee_scores(_committee(ids))
    descriptors = {
        record_id: [float(index), float(index % 7), float((index * index) % 11)]
        for index, record_id in enumerate(ids)
    }
    for policy in ("random", "disagreement", "disagreement_fps", "tail_risk_fps"):
        selected, details = select_candidates(
            policy=policy,
            remaining_ids=ids,
            scores=scores,
            raw_descriptors=descriptors,
            round_index=1,
        )
        assert selected == sorted(set(selected))
        assert len(selected) == 12
        assert set(selected) <= set(ids)
        assert "label" not in " ".join(details).lower()


def test_random_policy_is_deterministic_but_round_specific() -> None:
    ids = [f"candidate-{index:03d}" for index in range(48)]
    scores = committee_scores(_committee(ids))
    descriptors = {
        record_id: [float(index), float(index % 5)]
        for index, record_id in enumerate(ids)
    }
    first, _ = select_candidates(
        policy="random",
        remaining_ids=ids,
        scores=scores,
        raw_descriptors=descriptors,
        round_index=1,
    )
    repeated, _ = select_candidates(
        policy="random",
        remaining_ids=list(reversed(ids)),
        scores=scores,
        raw_descriptors=descriptors,
        round_index=1,
    )
    second_round, _ = select_candidates(
        policy="random",
        remaining_ids=ids,
        scores=scores,
        raw_descriptors=descriptors,
        round_index=2,
    )
    assert first == repeated
    assert first != second_round


def test_tail_proxy_ranking_differs_from_disagreement_when_force_magnitude_is_large() -> None:
    members = [
        {
            "low-magnitude": {
                "energy_ev": 0.0,
                "forces_ev_per_a": [[offset, 0.0, 0.0]],
            },
            "high-magnitude": {
                "energy_ev": 0.0,
                "forces_ev_per_a": [[10.0 + 0.1 * offset, 0.0, 0.0]],
            },
        }
        for offset in (-1.0, 0.0, 1.0)
    ]
    scores = committee_scores(members)
    assert (
        scores["low-magnitude"]["mean_population_vector_disagreement_ev_per_a"]
        > scores["high-magnitude"]["mean_population_vector_disagreement_ev_per_a"]
    )
    assert (
        scores["high-magnitude"]["force_tail_proxy_ev_per_a"]
        > scores["low-magnitude"]["force_tail_proxy_ev_per_a"]
    )
