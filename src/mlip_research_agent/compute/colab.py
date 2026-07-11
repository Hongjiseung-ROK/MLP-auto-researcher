"""Colab provider adapter: validation worker, never the authoritative environment.

Real sessions go through the security-reviewed `colab` CLI (see
docs/SKILL_SECURITY_REVIEW.md); authentication stays entirely in gcloud ADC —
this adapter never touches or stores credentials. Without a configured
transport it reports unauthenticated with setup guidance.
"""

from __future__ import annotations

from typing import ClassVar

from mlip_research_agent.compute._remote import (
    MockRemoteTransport,
    RemoteProviderBase,
    RemoteTransport,
)
from mlip_research_agent.compute.schemas import GpuObservation, ProviderName

__all__ = ["ColabProvider", "MockRemoteTransport", "RemoteTransport", "mock_l4_observation"]

COLAB_AUTH_GUIDANCE = (
    "Authenticate the reviewed `colab` CLI via Application Default Credentials: "
    "run `gcloud auth application-default login` with the openid, cloud-platform, "
    "userinfo.email, and colaboratory scopes (see "
    "src/mlip_research_agent/skills/compute/colab/security.md), then verify with "
    "`colab sessions`. Never paste tokens into chat, files, or shell history."
)


def mock_l4_observation() -> GpuObservation:
    """A realistic mock of what Colab reports for an assigned L4."""
    return GpuObservation(
        name="NVIDIA L4",
        uuid="GPU-mock-0000-l4",
        memory_total_mb=23034,
        driver_version="550.54.15",
        cuda_version="12.4",
        torch_version="2.5.0+cu124",
        compute_capability="8.9",
    )


class ColabProvider(RemoteProviderBase):
    name: ClassVar[ProviderName] = ProviderName.COLAB
    auth_guidance: ClassVar[str] = COLAB_AUTH_GUIDANCE
