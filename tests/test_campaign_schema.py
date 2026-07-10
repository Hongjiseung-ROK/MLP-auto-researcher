"""Campaign schema validation tests."""

import pytest
from pydantic import ValidationError

from mlip_research_agent.schemas.campaign import CampaignSpec


def base_payload() -> dict:
    return {
        "name": "test-campaign",
        "mode": "active_learning",
        "target_system": {"formula": "Cu", "crystal_structure": "fcc", "lattice_constant": 3.6},
        "budget": {"max_labels": 8},
    }


def test_valid_campaign_loads(example_campaign) -> None:  # type: ignore[no-untyped-def]
    assert example_campaign.name == "cu-fcc-mock-al"
    assert example_campaign.budget.max_labels == 8
    assert example_campaign.failure_injection is not None


def test_content_hash_is_stable() -> None:
    a = CampaignSpec.model_validate(base_payload())
    b = CampaignSpec.model_validate(base_payload())
    assert a.content_hash() == b.content_hash()
    c = CampaignSpec.model_validate({**base_payload(), "seed": 1})
    assert a.content_hash() != c.content_hash()


def test_unsupported_backend_rejected() -> None:
    with pytest.raises(ValidationError, match="not supported yet"):
        CampaignSpec.model_validate({**base_payload(), "backend": "orca"})


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValidationError):
        CampaignSpec.model_validate({**base_payload(), "mode": "vibes_based"})


def test_budget_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        CampaignSpec.model_validate({**base_payload(), "budget": {"max_labels": 0}})
    with pytest.raises(ValidationError):
        CampaignSpec.model_validate(
            {**base_payload(), "budget": {"max_labels": 8, "max_retries_per_step": 99}}
        )


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        CampaignSpec.model_validate({**base_payload(), "gpu_hours": 10_000})
