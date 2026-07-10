"""Metric primitives used by evaluation skills. Pure, deterministic functions."""

from __future__ import annotations

from collections.abc import Sequence


def mean_absolute_error(predictions: Sequence[float], targets: Sequence[float]) -> float:
    if len(predictions) != len(targets):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(targets)} targets"
        )
    if not predictions:
        raise ValueError("cannot compute MAE of an empty set")
    return sum(abs(p - t) for p, t in zip(predictions, targets, strict=True)) / len(predictions)
