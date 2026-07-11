# SKILL: claim_verification

## Purpose
Audit every registered scientific claim against the run's artifact manifest:
a claim is VERIFIED only if all referenced artifacts are registered and their
on-disk sha256 hashes still match; otherwise it is REJECTED with a reason.

## Supported use cases
- Final gate of every campaign before report generation.
- Standalone audits of a finished run directory.

## Non-goals
- Citation verification against external literature databases (post-v0).
- Semantic checking that the claim's number was computed correctly (that is
  the producing skill's test responsibility).

## Typed inputs — `schema.VerificationInput`
`claims_path` (default `claims.json`), `strict` (default true).

## Typed outputs — `schema.VerificationOutput`
`report_artifact`/`report_path`, `verified_claims_artifact`/`verified_claims_path`,
`n_claims`, `n_verified`, `n_rejected`, plus counts by claim class,
scientific evidence tier, artifact-verification status, and rejection category.

## Required dependencies
Standard library only.

## Preconditions
The executor has persisted `claims.json` and the manifest before this step
(it saves both after every step).

## Validation rules
Missing or malformed claims file → `VALIDATION_ERROR`, non-retryable. Agent
prose/reflection/report artifacts are rejected as evidence even when their
hashes are intact. `VERIFIED` means artifact integrity only; it never upgrades
the claim's scientific evidence tier.

## Failure taxonomy
`VALIDATION_ERROR`, `UNSUPPORTED_CLAIM` (strict mode, any rejection).

## Retry and recovery policy
`UNSUPPORTED_CLAIM` is non-retryable and escalates to a human. The audit
artifacts are registered *before* the failure is raised, so evidence of the
rejection is never lost.

## Produced artifacts
`steps/<step_id>/verification_report.json` (kind `verification_report`),
`steps/<step_id>/verified_claims.json` (kind `verified_claims`).

## Provenance fields
Rejected claim ids and reasons inside the report artifact.

## Cost class / permission level
`trivial` / `auto`.

## Minimal example
`examples/example_input.json`.

## Correctness tests
`tests/test_claim_verification.py`: verified path, rejection of claims with
unregistered references, rejection of hash-corrupted references, strict-mode
failure with preserved artifacts, non-strict counting mode.
