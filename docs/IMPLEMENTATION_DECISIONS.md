# Implementation Decisions

Traceable record of decisions made during bootstrap (2026-07-10). Each entry:
what was decided, why, and what would change it.

## D-1: First-party runtime instead of OpenHands for milestone 1
plan.md names OpenHands as the execution substrate. Milestone 1 implements the
substrate *patterns* first-party (append-only event log, typed skills,
bounded workspaces per step) with ~500 lines of audited code, because the
mock slice needs no LLM, no sandbox, and no remote runtime. OpenHands adoption
is an open question (OQ-8); the event/skill schemas were kept close to
OpenHands concepts so an adapter remains cheap.

## D-2: Python 3.11, conda-forge only
3.12/3.13 still lag in the MLIP ecosystem (torch/e3nn pins). `nodefaults` in
`environment.yml` avoids the Anaconda `defaults` channel licensing ambiguity.
Revisit at the first real backend integration.

## D-3: First-party JSON structure format instead of extxyz
Registered artifacts must be byte-reproducible across reruns and library
upgrades. `ase.io` text formats embed version-dependent formatting;
`skills/atomistics/structures_io.py` serializes positions/cell/pbc as sorted
JSON instead, with lossless ASE `Atoms` conversion both ways. Real backends
can still exchange extxyz at their boundaries.

## D-4: Manifest and checkpoint carry no timestamps
Timestamps live only in the event log and provenance bundle. This makes
`manifest.json` and `claims.json` byte-identical across reruns with equal
inputs and seed — verified live and enforced by
`tests/test_end_to_end.py::test_rerun_reproduces_artifact_hashes`.
Documented tolerance: run ids, event timestamps, provenance `timestamps` and
`command` fields differ; everything else must match.

## D-5: Mock labeler is Lennard-Jones, mock model is a mean baseline
The mock DFT skill computes real (deterministic) LJ energies via ASE so the
labeling contract is exercised with honest numerics; the mock trainer is a
mean-energy baseline whose model artifact carries its own train/holdout split
so evaluation cannot leak. Neither pretends to be science; both are named
`mock_*` and hard-fail on any non-mock method/model name.

## D-6: Failure injection is a first-class campaign field
`failure_injection: {step, times}` in the campaign spec (validated, bounded at
3) rather than an environment variable or monkeypatch, so the demo of bounded
recovery is itself reproducible and provenance-recorded. Injection is
implemented only by the labeling skill (`inject_failure_times` input); the
compiler rejects injection into steps that don't declare the field.

## D-7: REFINE repairs flow through recorded parameter overrides
A retryable `SkillError` may carry `repair_params`; the executor merges them
into the next attempt's inputs and records them in the failure record's
`attempted_repairs` and the `RECOVERY_DECISION` event. This satisfies the rule
that convergence criteria never change silently: the repair appears in the
event log, the failure object, and the produced artifact's `settings` block.

## D-8: Verification runs as the final workflow step, strict by default
The executor persists `claims.json` + `manifest.json` after every step, so the
`claim_verification` skill audits the same files a human would. It registers
its audit artifacts *before* raising `UNSUPPORTED_CLAIM`, so evidence of a
rejection survives the failed run. The run report renders verified claims
only; it is a rendering stage, never a source of truth (plan.md synthesis).

## D-9: No subagents during bootstrap
The instruction allows delegating SKILL construction to subagents when
decomposition reduces context or enables parallel specialization. For this
bootstrap the six skills share one contract, one context object, and one
serialization module that were being designed in the same pass; delegation
would have re-sent that evolving context repeatedly for no parallel gain.
Subagents become appropriate at the next milestone (isolated adapters for a
chosen MLIP backend, ORCA output-parser fixtures) per the token-optimization
policy in the bootstrap instructions.

## D-10: Host PYTHONPATH leak neutralized per-command, not fixed globally
`~/.zshrc` exports `PYTHONPATH=~/developer/chemsmart`, which injects an
unrelated dev checkout into every Python process and breaks bare `pip check`.
The user's shell config was left untouched (it serves their chemsmart work);
all verification commands here run with `env -u PYTHONPATH`. Recorded in the
bootstrap environment report. A `conda env config vars set PYTHONPATH=` on the
env would be a stickier fix — deferred, as it is user-visible state.

## D-11: Single-iteration AL loop in v0
`stopping.max_iterations` is validated but the compiler emits one
literature → structures → selection → labeling → training → evaluation →
verification pass. Multi-iteration looping (the real AL cycle) needs the
decision-gate design (refine/pivot/sufficient) and belongs to the next
milestone with the real backend; wiring a fake loop now would create schema
churn.

## D-13: Owner decisions from bootstrap Q&A (2026-07-10)
The project owner resolved: benchmark = **fcc Cu perturbed bulk,
materials-first** (OQ-1/2); first backend = **MACE, mandatory baseline**
(OQ-3/4); DFT backend = **ORCA** (already installed on host, academic
license) (OQ-5); LLM provider = **OpenAI**, with **OpenHands kept as an
architectural reference, not a runtime dependency** (OQ-7/8). Consequence:
the `agent` extra will carry `openai`; ORCA's lack of periodic boundary
conditions vs the periodic benchmark is tracked as OQ-13.

## D-14: Compute policy file deviations from the owner's template
`configs/compute_policy.yaml` follows the owner-provided policy verbatim with
two additions: `colab.max_runtime_minutes: 120` (makes "development_training
on Colab only when bounded" enforceable) and `preflight.max_age_hours: 168`
(defines "non-stale"). One semantic fix: the full-campaign denial does not
apply to the routing-designated primary provider (otherwise VESSL would deny
its own production workloads, since the template sets `allow_full_campaign`
nowhere for vessl).

## D-15: Remote providers are transport-pluggable; only mocks ship in v0
`RemoteProviderBase` owns lifecycle/attestation/policy; a `RemoteTransport`
does raw session ops. `MockRemoteTransport` is first-party and deterministic;
the real Colab transport will wrap the security-reviewed `google-colab-cli`
v0.6.0 (docs/SKILL_SECURITY_REVIEW.md) once the owner's budget/auth answers
land, and VESSL likewise. Attestation artifacts contain timestamps and run
ids by design — compute skills are infrastructure and are excluded from
byte-reproducibility-gated scientific DAGs.

## D-16: tea_time_with_reading_poem is agent-text by construction
The reflection skill (owner-requested creative-break capability) is
deterministic (seeded poem/technique selection, numpy-recomputed alternative
views) and registers `reflection` artifacts but never claims — enforced by an
in-code invariant raising `UNSUPPORTED_CLAIM` and by tests. It is registered
in the skill registry but deliberately not wired into the default compiled
campaign DAG; a campaign flag can opt in later.

## D-12: Skill tests live inside each skill package
Per the SKILL contract (`skills/<name>/tests/`), with cross-cutting tests
(executor, schemas, end-to-end) under top-level `tests/`. `pytest` testpaths
covers both; a skill is only "done" when its own test directory passes in
isolation.
