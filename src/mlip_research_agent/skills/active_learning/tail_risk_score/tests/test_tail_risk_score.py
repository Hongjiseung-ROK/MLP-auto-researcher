from __future__ import annotations

import numpy as np
import pytest

from mlip_research_agent.skills.active_learning.tail_risk_score.implementation import (
    force_tail_proxy,
)
from mlip_research_agent.skills.active_learning.tail_risk_score.schema import (
    TailRiskScoreInput,
)


def test_force_tail_proxy_uses_population_vector_disagreement_and_upper_tail() -> None:
    forces = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            [[3.0, 0.0, 0.0], [0.0, 6.0, 0.0]],
        ]
    )
    score, count, proxy, magnitude, disagreement = force_tail_proxy(
        forces,
        tail_fraction=0.5,
        magnitude_weight=1.0,
        disagreement_weight=1.0,
    )

    assert count == 1
    assert magnitude.tolist() == pytest.approx([2.0, 3.0])
    expected_disagreement = np.asarray([np.sqrt(2.0 / 3.0), np.sqrt(6.0)])
    assert disagreement.tolist() == pytest.approx(expected_disagreement.tolist())
    assert proxy.tolist() == pytest.approx((magnitude + expected_disagreement).tolist())
    assert score == pytest.approx(float(proxy.max()))


def test_tail_risk_schema_rejects_zero_weights() -> None:
    members = [
        {
            "member_id": f"member-{index}",
            "provenance_sha256": f"{index + 1:064x}",
            "predicted_forces": {"a": [[0.0, 0.0, 0.0]]},
        }
        for index in range(3)
    ]
    with pytest.raises(ValueError, match="at least one force-tail proxy weight"):
        TailRiskScoreInput.model_validate(
            {
                "pool_candidate_ids": ["a"],
                "metadata": {"a": {"source_group": "pool", "n_atoms": 1}},
                "members": members,
                "tail_fraction": 0.05,
                "magnitude_weight": 0.0,
                "disagreement_weight": 0.0,
                "campaign_id": "campaign",
                "round_id": "round-1",
            }
        )
