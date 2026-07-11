"""Provider-neutral compute layer: local CPU -> Colab preflight -> VESSL primary.

Scientific SKILLs never contain provider-specific authentication or submission
commands; they submit typed JobSpec objects through the ComputeRouter, which
enforces configs/compute_policy.yaml (fail-closed) and runtime GPU attestation.
"""

from mlip_research_agent.compute.base import ComputePolicyDenied, ComputeProvider
from mlip_research_agent.compute.policy import ComputePolicy, load_policy, normalize_gpu_name
from mlip_research_agent.compute.router import ComputeRouter
from mlip_research_agent.compute.schemas import JobSpec, ProviderName, WorkloadClass

__all__ = [
    "ComputePolicy",
    "ComputePolicyDenied",
    "ComputeProvider",
    "ComputeRouter",
    "JobSpec",
    "ProviderName",
    "WorkloadClass",
    "load_policy",
    "normalize_gpu_name",
]
