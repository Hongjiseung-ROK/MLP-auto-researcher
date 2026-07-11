"""Colab preflight skill: submit a bounded validation job through the router.

The skill never handles provider authentication: the mock transport is used by
default, and the real transport is only attachable once the reviewed `colab`
CLI is authenticated externally (docs/SKILL_SECURITY_REVIEW.md, security.md).
Policy denials (unauthorized/mismatched GPU) surface as structured failures
after the attestation evidence has been registered.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel

from mlip_research_agent.compute._remote import MockRemoteTransport
from mlip_research_agent.compute.colab import ColabProvider, mock_l4_observation
from mlip_research_agent.compute.local import LocalProvider
from mlip_research_agent.compute.policy import load_policy, normalize_gpu_name
from mlip_research_agent.compute.preflight import PreflightRecord, PreflightStore
from mlip_research_agent.compute.router import ComputeRouter
from mlip_research_agent.compute.schemas import (
    AcceleratorId,
    GpuObservation,
    JobSpec,
    JobState,
    ProviderName,
)
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.compute.colab.schema import (
    ColabPreflightInput,
    ColabPreflightOutput,
)
from mlip_research_agent.skills.compute.colab.validators import (
    validate_inputs,
    validate_policy_file,
)

RESULT_FILENAME = "preflight_result.json"


def _mock_observation(name: str) -> GpuObservation:
    if normalize_gpu_name(name) is AcceleratorId.NVIDIA_L4:
        return mock_l4_observation()
    return GpuObservation(name=name, uuid="GPU-mock-generic", memory_total_mb=16000)


@register_skill
class ColabPreflightSkill(Skill):
    name = "colab_preflight"
    input_model = ColabPreflightInput
    output_model = ColabPreflightOutput
    cost_class = "cheap"  # real Colab sessions consume compute units: still bounded
    permission_level = "auto"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, ColabPreflightInput)
        validate_inputs(params)
        policy = load_policy(validate_policy_file(params.policy_path))

        if not params.use_mock:
            raise SkillError(
                "real Colab transport is not enabled yet: it requires the reviewed "
                "`colab` CLI authenticated via gcloud ADC (see "
                "src/mlip_research_agent/skills/compute/colab/security.md) and the "
                "owner's Colab budget answers (docs/OPEN_QUESTIONS.md).",
                failure_class=FailureClass.TOOL_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )

        transport = MockRemoteTransport(_mock_observation(params.mock_observed_gpu))
        provider = ColabProvider(policy, ctx.step_dir, transport)
        store = PreflightStore(ctx.step_dir / "preflight-records")
        router = ComputeRouter(
            policy,
            {
                ProviderName.COLAB: provider,
                ProviderName.LOCAL: LocalProvider(policy, ctx.step_dir),
            },
            store,
            audit_log_path=ctx.step_dir / "audit.jsonl",
        )
        spec = JobSpec(
            name="colab-preflight",
            workload_class=params.workload_class,
            requested_accelerator=params.requested_accelerator,
            command=params.command,
            max_runtime_minutes=params.max_runtime_minutes,
            repo_commit=params.repo_commit,
        )
        handle = router.submit(spec)
        status = provider.status(handle)
        manifest = provider.collect_artifacts(handle)

        attestation_path = ctx.step_dir / "compute" / handle.run_id / "attestation.json"
        attestation_artifact = ctx.registry.register(
            attestation_path, kind="attestation", step_id=ctx.step_id
        )
        attestation = json.loads(attestation_path.read_text())

        record_saved = False
        if status.state is JobState.SUCCEEDED:
            store.save(
                PreflightRecord(
                    run_id=handle.run_id,
                    passed=True,
                    repo_commit=spec.repo_commit,
                    dependency_hash=spec.dependency_hash,
                    mlip_backend=spec.mlip_backend,
                    cuda_compat_class=spec.cuda_compat_class,
                    workflow_schema_version=spec.workflow_schema_version,
                    created_at=datetime.now(UTC).isoformat(),
                )
            )
            record_saved = True

        result_path = ctx.step_dir / RESULT_FILENAME
        result_path.write_text(
            json.dumps(
                {
                    "job_state": status.state.value,
                    "message": status.message,
                    "attestation": attestation,
                    "remote_artifacts": manifest.model_dump(mode="json"),
                    "preflight_record_saved": record_saved,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        result_artifact = ctx.registry.register(
            result_path, kind="preflight_result", step_id=ctx.step_id
        )

        if status.state is JobState.POLICY_DENIED:
            # Structured policy failure: evidence artifacts are registered
            # above, the workload never ran, and a bounded retry may request
            # a different runtime.
            raise SkillError(
                f"Colab runtime denied by GPU policy: {status.message}",
                failure_class=FailureClass.TOOL_ERROR,
                severity=Severity.HIGH,
                retryable=True,
                likely_causes=["Colab assigned an accelerator outside the allowlist"],
            )

        canonical_raw = attestation.get("canonical_accelerator")
        return ColabPreflightOutput(
            job_state=status.state,
            canonical_accelerator=(
                AcceleratorId(canonical_raw) if canonical_raw is not None else None
            ),
            attestation_artifact=attestation_artifact.artifact_id,
            result_artifact=result_artifact.artifact_id,
            preflight_record_saved=record_saved,
            n_remote_artifacts=len(manifest.artifacts),
        )
