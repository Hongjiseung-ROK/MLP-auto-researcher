"""Compute policy: GPU allowlists, alias normalization, fail-closed evaluation.

The policy file (configs/compute_policy.yaml) is the single source of truth.
Unknown accelerators, name mismatches, and missing attestations are denied.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from mlip_research_agent.compute.schemas import (
    PRODUCTION_WORKLOADS,
    AcceleratorId,
    JobSpec,
    ProviderName,
    WorkloadClass,
)

# Exact-match alias table (casefolded, whitespace-collapsed). GPU identity is
# never inferred from partial matches: unknown strings normalize to None.
GPU_NAME_ALIASES: dict[str, AcceleratorId] = {
    "cpu": AcceleratorId.CPU,
    "nvidia l4": AcceleratorId.NVIDIA_L4,
    "nvidia-l4": AcceleratorId.NVIDIA_L4,
    "nvidia a100-sxm4-40gb": AcceleratorId.NVIDIA_A100_40GB,
    "nvidia-a100-40gb": AcceleratorId.NVIDIA_A100_40GB,
    "nvidia a100-sxm4-80gb": AcceleratorId.NVIDIA_A100_80GB,
    "nvidia a100 80gb pcie": AcceleratorId.NVIDIA_A100_80GB,
    "nvidia-a100-80gb": AcceleratorId.NVIDIA_A100_80GB,
}


def normalize_gpu_name(raw: str) -> AcceleratorId | None:
    """Map a runtime-reported device name to a canonical id, or None if unknown."""
    collapsed = re.sub(r"\s+", " ", raw.strip()).casefold()
    return GPU_NAME_ALIASES.get(collapsed)


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    reasons: list[str] = Field(default_factory=list)
    canonical_accelerator: AcceleratorId | None = None

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW


class RoutingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_provider: ProviderName
    validation_provider: ProviderName
    local_provider: ProviderName
    require_preflight_before_primary: bool = True


class ProviderPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    purpose: list[str] = Field(default_factory=list)
    allowed_accelerators: list[AcceleratorId]
    default_accelerator_preference: list[AcceleratorId] = Field(default_factory=list)
    max_gpus_per_run: int = Field(default=1, ge=0)
    allow_full_campaign: bool = False
    allow_long_training: bool = False
    require_runtime_attestation: bool = False
    max_runtime_minutes: int | None = Field(default=None, gt=0)


class ApprovalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_without_additional_gpu_model_approval: list[AcceleratorId]
    require_explicit_approval_for: list[str]


class EnforcementPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deny_unrecognized_gpu: bool = True
    deny_gpu_name_mismatch: bool = True
    deny_missing_attestation: bool = True
    fail_closed: bool = True


class PreflightPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_age_hours: int = Field(default=168, gt=0)


class ComputePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    routing: RoutingPolicy
    providers: dict[ProviderName, ProviderPolicy]
    approval: ApprovalPolicy
    enforcement: EnforcementPolicy
    preflight: PreflightPolicy = PreflightPolicy()

    def provider(self, name: ProviderName) -> ProviderPolicy:
        if name not in self.providers:
            raise KeyError(f"provider {name.value!r} not configured in compute policy")
        return self.providers[name]

    def evaluate(
        self,
        provider_name: ProviderName,
        spec: JobSpec,
        observed_gpu_name: str | None,
    ) -> PolicyDecision:
        """Fail-closed policy check for a job on a provider.

        observed_gpu_name is the device name the runtime actually reported;
        pass None when no attestation has been performed yet.
        """
        try:
            return self._evaluate(provider_name, spec, observed_gpu_name)
        except Exception as exc:  # fail_closed: any internal error denies
            if self.enforcement.fail_closed:
                return PolicyDecision(
                    decision=Decision.DENY, reasons=[f"policy evaluation error: {exc!r}"]
                )
            raise

    def _evaluate(
        self,
        provider_name: ProviderName,
        spec: JobSpec,
        observed_gpu_name: str | None,
    ) -> PolicyDecision:
        reasons: list[str] = []
        provider = self.provider(provider_name)

        if not provider.enabled:
            reasons.append(f"provider {provider_name.value} is disabled")
        if spec.requested_accelerator not in provider.allowed_accelerators:
            reasons.append(
                f"accelerator {spec.requested_accelerator.value} not allowlisted "
                f"for provider {provider_name.value}"
            )
        if spec.n_gpus > provider.max_gpus_per_run:
            reasons.append(
                f"{spec.n_gpus} GPUs exceeds max_gpus_per_run={provider.max_gpus_per_run} "
                "(multiple_gpus requires explicit approval)"
            )
        if (
            spec.workload_class in PRODUCTION_WORKLOADS
            and provider_name is not self.routing.primary_provider
            and not provider.allow_full_campaign
        ):
            reasons.append(
                f"workload {spec.workload_class.value} is a full campaign; "
                f"provider {provider_name.value} is not authorized for full campaigns"
            )
        if (
            spec.workload_class is WorkloadClass.DEVELOPMENT_TRAINING
            and provider_name is ProviderName.COLAB
        ):
            bound = provider.max_runtime_minutes
            if not provider.allow_long_training and (
                bound is None or spec.max_runtime_minutes > bound
            ):
                reasons.append(
                    "development_training on colab must be bounded: "
                    f"max_runtime_minutes={spec.max_runtime_minutes} exceeds "
                    f"policy bound {bound}"
                )

        canonical: AcceleratorId | None = None
        if observed_gpu_name is None:
            if provider.require_runtime_attestation and self.enforcement.deny_missing_attestation:
                reasons.append(
                    f"provider {provider_name.value} requires runtime attestation; "
                    "no observed GPU was provided"
                )
        else:
            canonical = normalize_gpu_name(observed_gpu_name)
            if canonical is None:
                if self.enforcement.deny_unrecognized_gpu:
                    reasons.append(f"observed accelerator {observed_gpu_name!r} is unrecognized")
            else:
                if canonical not in provider.allowed_accelerators:
                    reasons.append(
                        f"observed accelerator {canonical.value} not allowlisted for "
                        f"provider {provider_name.value}"
                    )
                if (
                    canonical is not spec.requested_accelerator
                    and self.enforcement.deny_gpu_name_mismatch
                ):
                    reasons.append(
                        f"observed accelerator {canonical.value} does not match "
                        f"requested {spec.requested_accelerator.value}"
                    )

        if reasons:
            return PolicyDecision(
                decision=Decision.DENY, reasons=reasons, canonical_accelerator=canonical
            )
        return PolicyDecision(decision=Decision.ALLOW, canonical_accelerator=canonical)


def load_policy(path: Path) -> ComputePolicy:
    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"compute policy file {path} must contain a YAML mapping")
    return ComputePolicy.model_validate(raw)
