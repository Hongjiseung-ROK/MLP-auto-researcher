"""Compute router: workload-class routing, override auditing, preflight gate."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlip_research_agent.compute.base import ComputePolicyDenied, ComputeProvider
from mlip_research_agent.compute.policy import ComputePolicy
from mlip_research_agent.compute.preflight import PreflightStore
from mlip_research_agent.compute.schemas import (
    PRODUCTION_WORKLOADS,
    JobHandle,
    JobSpec,
    ProviderName,
    WorkloadClass,
)

ROUTING_TABLE: dict[WorkloadClass, ProviderName] = {
    WorkloadClass.UNIT_TEST: ProviderName.LOCAL,
    WorkloadClass.GPU_SMOKE_TEST: ProviderName.COLAB,
    WorkloadClass.PREFLIGHT: ProviderName.COLAB,
    WorkloadClass.FAILURE_REPRODUCTION: ProviderName.COLAB,
    WorkloadClass.DEVELOPMENT_TRAINING: ProviderName.COLAB,
    WorkloadClass.PRIMARY_TRAINING: ProviderName.VESSL,
    WorkloadClass.ACTIVE_LEARNING: ProviderName.VESSL,
    WorkloadClass.BENCHMARK: ProviderName.VESSL,
    WorkloadClass.PRODUCTION_RESEARCH: ProviderName.VESSL,
}

_SECRET_PATTERNS = (
    # Keyword may be prefixed word-joined (HF_TOKEN, WANDB_API_KEY, my-secret).
    re.compile(
        r"(?i)([a-z0-9_-]*(?:token|secret|password|passwd|api[_-]?key|bearer"
        r"|authorization|credential)s?)(\s*[=:]\s*)(\S+)"
    ),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bya29\.[A-Za-z0-9_-]{16,}\b"),
)


def redact_secrets(text: str) -> str:
    """Scrub token-like material from any text destined for logs or records."""
    scrubbed = _SECRET_PATTERNS[0].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    for pattern in _SECRET_PATTERNS[1:]:
        scrubbed = pattern.sub("[REDACTED]", scrubbed)
    return scrubbed


class ComputeRouter:
    """Routes typed JobSpecs to providers under the compute policy.

    Every provider override and every VESSL preflight-gate decision is written
    to the audit log (JSONL, secret-redacted).
    """

    def __init__(
        self,
        policy: ComputePolicy,
        providers: Mapping[ProviderName, ComputeProvider],
        preflight_store: PreflightStore,
        audit_log_path: Path | None = None,
    ) -> None:
        self.policy = policy
        self.providers = dict(providers)
        self.preflight_store = preflight_store
        self.audit_log_path = audit_log_path

    def route(self, spec: JobSpec) -> ProviderName:
        """Resolve the provider for a spec, honoring policy-permitted overrides."""
        default = ROUTING_TABLE[spec.workload_class]
        if spec.provider_override is None:
            return default
        override = spec.provider_override
        decision = self.policy.evaluate(override, spec, observed_gpu_name=None)
        # Pre-submission check ignores the attestation requirement (no device
        # has been assigned yet); every other reason is binding.
        binding = [r for r in decision.reasons if "runtime attestation" not in r]
        allowed = not binding
        self._audit(
            "provider_override",
            spec,
            {
                "default_provider": default.value,
                "override_provider": override.value,
                "allowed": allowed,
                "reasons": binding,
            },
        )
        if not allowed:
            raise ComputePolicyDenied(
                [f"provider override to {override.value} denied", *binding]
            )
        return override

    def submit(self, spec: JobSpec, emergency_override_approved_by: str | None = None) -> JobHandle:
        provider_name = self.route(spec)
        self._enforce_preflight_gate(spec, provider_name, emergency_override_approved_by)
        provider = self.providers.get(provider_name)
        if provider is None:
            raise ComputePolicyDenied([f"no adapter configured for provider {provider_name.value}"])
        handle = provider.submit(spec)
        self._audit(
            "job_submitted",
            spec,
            {"provider": provider_name.value, "job_id": handle.job_id, "run_id": handle.run_id},
        )
        return handle

    def _enforce_preflight_gate(
        self,
        spec: JobSpec,
        provider_name: ProviderName,
        emergency_override_approved_by: str | None,
    ) -> None:
        if provider_name is not self.policy.routing.primary_provider:
            return
        if not self.policy.routing.require_preflight_before_primary:
            return
        if spec.workload_class not in PRODUCTION_WORKLOADS and (
            spec.workload_class is not WorkloadClass.PRIMARY_TRAINING
        ):
            return
        record = self.preflight_store.find_valid(spec, self.policy.preflight.max_age_hours)
        if record is not None:
            self._audit("preflight_gate", spec, {"result": "valid", "preflight_run": record.run_id})
            return
        if emergency_override_approved_by:
            # Documented emergency path: requires explicit human approval and
            # is permanently recorded in the audit log.
            self._audit(
                "preflight_emergency_override",
                spec,
                {"approved_by": emergency_override_approved_by},
            )
            return
        self._audit("preflight_gate", spec, {"result": "missing_or_stale"})
        raise ComputePolicyDenied(
            [
                "VESSL submission blocked: no passing, non-stale Colab preflight matches "
                f"commit={spec.repo_commit[:12]} deps={spec.dependency_hash[:12] or 'unset'} "
                f"backend={spec.mlip_backend} cuda={spec.cuda_compat_class} "
                f"schema={spec.workflow_schema_version}"
            ]
        )

    def _audit(self, event: str, spec: JobSpec, payload: dict[str, Any]) -> None:
        if self.audit_log_path is None:
            return
        entry = {
            "timestamp": datetime.now(UTC).isoformat(timespec="microseconds"),
            "event": event,
            "job_name": spec.name,
            "workload_class": spec.workload_class.value,
            **payload,
        }
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log_path.open("a") as fh:
            fh.write(redact_secrets(json.dumps(entry, sort_keys=True)) + "\n")
