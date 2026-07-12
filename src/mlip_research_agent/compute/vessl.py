"""VESSL provider adapter: the authoritative production compute path.

The router refuses VESSL submissions without a passing, non-stale Colab
preflight for the same commit/deps/backend/CUDA-class/schema (see
compute/router.py). Authentication stays in the `vessl` CLI / environment;
this adapter never touches or stores credentials.
"""

from __future__ import annotations

from typing import ClassVar

from mlip_research_agent.compute._remote import RemoteProviderBase
from mlip_research_agent.compute.schemas import GpuObservation, ProviderName
from mlip_research_agent.compute.vessl_cloud import VesslCloudProvider

__all__ = ["VesslCloudProvider", "VesslProvider", "mock_a100_80gb_observation"]

VESSL_AUTH_GUIDANCE = (
    "Install the current `vesslctl` CLI after explicit approval, then authenticate with "
    "`vesslctl auth login` using its browser flow. Current Cloud uses organization/team; "
    "never use the legacy `vessl` CLI or read token/config files."
)


def mock_a100_80gb_observation() -> GpuObservation:
    """A realistic mock of what VESSL reports for an assigned A100 80GB."""
    return GpuObservation(
        name="NVIDIA A100-SXM4-80GB",
        uuid="GPU-mock-0000-a100-80",
        memory_total_mb=81920,
        driver_version="550.54.15",
        cuda_version="12.4",
        torch_version="2.5.0+cu124",
        compute_capability="8.0",
    )


class VesslProvider(RemoteProviderBase):
    name: ClassVar[ProviderName] = ProviderName.VESSL
    auth_guidance: ClassVar[str] = VESSL_AUTH_GUIDANCE
