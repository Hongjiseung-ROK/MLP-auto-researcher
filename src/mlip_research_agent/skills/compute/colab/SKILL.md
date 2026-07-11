# SKILL: colab_preflight

## Purpose
Run a bounded Colab validation workload (preflight / GPU smoke test / failure
reproduction) through the provider-neutral compute router, with runtime GPU
attestation, fail-closed policy enforcement, and evidence artifacts.

## Supported use cases
- Mandatory preflight before any VESSL production submission.
- GPU-policy validation with a mock runtime (offline, default).
- Structured denial evidence when Colab assigns an unauthorized device.

## Non-goals
- Full campaigns, long training, or anything whose results support scientific
  claims (Colab is a validation worker; VESSL is authoritative).
- Provider authentication: the reviewed `colab` CLI authenticates externally
  via gcloud ADC (see `security.md`); this skill never handles or stores
  credentials and never duplicates the CLI's auth logic.

## Typed inputs — `schema.ColabPreflightInput`
`workload_class` (validation classes only), `requested_accelerator`
(`nvidia-l4` default; CPU rejected), `command`, `max_runtime_minutes` (≤120),
`repo_commit`, `policy_path`, `use_mock` (default true), `mock_observed_gpu`.

## Typed outputs — `schema.ColabPreflightOutput`
`job_state`, `canonical_accelerator`, `attestation_artifact`,
`result_artifact`, `preflight_record_saved`, `n_remote_artifacts`.

## Required dependencies
`mlip_research_agent.compute` (router, policy, attestation), pyyaml.
Real mode additionally requires the security-reviewed `google-colab-cli`
(pinned v0.6.0, docs/SKILL_SECURITY_REVIEW.md) — not yet enabled.

## Preconditions
`configs/compute_policy.yaml` exists; for real mode, `colab sessions`
succeeds under an externally configured ADC login.

## Validation rules (`validators.py`)
- Workload must be a validation class (`VALIDATION_ERROR`, non-retryable).
- Requested accelerator must be a GPU id (`VALIDATION_ERROR`).
- Policy file must exist (`VALIDATION_ERROR`).

## Failure taxonomy
`VALIDATION_ERROR` (inputs/policy file), `TOOL_ERROR` (real transport not
enabled; policy denial of the assigned runtime — retryable, since a new
session may receive an authorized device).

## Retry and recovery policy
Policy denials are retryable within the executor's bounded budget (a retry
requests a fresh runtime). The denial attestation and result artifacts are
registered *before* the failure raises, so evidence is never lost.

## Produced artifacts
`compute/<run-id>/attestation.json` (kind `attestation`, immutable),
`preflight_result.json` (kind `preflight_result`), plus the remote job's
status/logs under `compute/<run-id>/`. On success a `PreflightRecord` is
saved for the VESSL gate. Note: these artifacts contain timestamps and run
ids; this is an infrastructure skill and must not be wired into
byte-reproducibility-gated scientific DAGs.

## Provenance fields
Attestation: provider, requested vs observed GPU, canonical id, UUID, VRAM,
CUDA/driver/torch versions, policy decision + reasons, commit, timestamps.

## Cost class / permission level
`cheap` (mock) — real sessions consume Colab compute units but stay bounded /
`auto` (real long sessions would move to `human_approval` with budget rules).

## Minimal example
`examples/example_input.json`.

## Correctness tests
`tests/test_colab_preflight_skill.py`: mock allow path (L4), structured
denial on unauthorized device with preserved evidence, CPU/production-class
input rejection, real-mode guidance error.
