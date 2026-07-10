"""Shared fixtures for top-level tests."""

from pathlib import Path

import pytest

from mlip_research_agent.schemas.campaign import CampaignSpec, load_campaign

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_CAMPAIGN = REPO_ROOT / "configs" / "example_campaign.yaml"


@pytest.fixture
def example_campaign() -> CampaignSpec:
    return load_campaign(EXAMPLE_CAMPAIGN)
