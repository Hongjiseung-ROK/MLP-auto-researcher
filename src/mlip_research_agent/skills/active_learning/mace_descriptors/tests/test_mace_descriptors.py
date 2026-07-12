"""Deterministic, label-free descriptor preprocessing tests."""

from __future__ import annotations

import numpy as np
import pytest

from mlip_research_agent.data.oracle import AcquisitionView
from mlip_research_agent.skills.active_learning.mace_descriptors.implementation import (
    standardize_descriptors,
)
from mlip_research_agent.skills.active_learning.mace_descriptors.validators import (
    validate_num_layers,
    validate_view,
)


def test_pool_standardization_drops_constants_and_l2_normalizes() -> None:
    vectors, raw_dim, dim, retained = standardize_descriptors(
        {
            "b": np.asarray([2.0, 5.0, 6.0]),
            "a": np.asarray([1.0, 5.0, 4.0]),
            "c": np.asarray([4.0, 5.0, 8.0]),
        },
        min_feature_std=1.0e-12,
        l2_normalize=True,
    )
    assert raw_dim == 3
    assert dim == 2
    assert retained == [0, 2]
    assert list(vectors) == ["a", "b", "c"]
    assert all(np.linalg.norm(value) == pytest.approx(1.0) for value in vectors.values())


def test_nonfinite_or_constant_pool_fails() -> None:
    with pytest.raises(ValueError, match="finite"):
        standardize_descriptors(
            {"a": np.asarray([0.0, np.nan]), "b": np.asarray([1.0, 2.0])},
            min_feature_std=1.0e-12,
            l2_normalize=False,
        )


def test_pool_centroid_remains_a_valid_zero_vector() -> None:
    vectors, _, _, _ = standardize_descriptors(
        {
            "a": np.asarray([-1.0, -2.0]),
            "b": np.asarray([0.0, 0.0]),
            "c": np.asarray([1.0, 2.0]),
        },
        min_feature_std=1.0e-12,
        l2_normalize=True,
    )
    assert vectors["b"] == pytest.approx([0.0, 0.0])


def test_view_boundary_rejects_wrong_partition_duplicates_and_species() -> None:
    candidate = {
        "record_id": "candidate-a",
        "symbols": ["Cu"],
        "positions": [[0.0, 0.0, 0.0]],
        "cell": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "pbc": [True, True, True],
        "metadata": {
            "top_group": "pool",
            "group_id": "pool-group",
            "n_atoms": 1,
            "structure_fingerprint_sha256": "a" * 64,
        },
    }
    wrong_partition = AcquisitionView.model_validate(
        {
            "dataset_id": "dataset",
            "split_semantic_sha256": "b" * 64,
            "partition": "frozen_test",
            "candidates": [candidate],
        }
    )
    with pytest.raises(ValueError, match="acquisition_pool"):
        validate_view(wrong_partition, ["Cu"])

    duplicate = AcquisitionView.model_validate(
        {
            "dataset_id": "dataset",
            "split_semantic_sha256": "b" * 64,
            "partition": "acquisition_pool",
            "candidates": [candidate, candidate],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_view(duplicate, ["Cu"])

    unsupported = AcquisitionView.model_validate(
        {
            "dataset_id": "dataset",
            "split_semantic_sha256": "b" * 64,
            "partition": "acquisition_pool",
            "candidates": [candidate],
        }
    )
    with pytest.raises(ValueError, match="unsupported"):
        validate_view(unsupported, ["H"])


def test_num_layers_is_exactly_bounded_by_checkpoint_depth() -> None:
    validate_num_layers(-1, 2)
    validate_num_layers(1, 2)
    validate_num_layers(2, 2)
    with pytest.raises(ValueError, match="between 1 and 2"):
        validate_num_layers(3, 2)
    with pytest.raises(ValueError, match="constant"):
        standardize_descriptors(
            {"a": np.asarray([1.0]), "b": np.asarray([1.0])},
            min_feature_std=1.0e-12,
            l2_normalize=False,
        )
