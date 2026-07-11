# Phase 2 Implementation Status

Tracks `plan_phase_2.md` work packages. Evidence entries must reference a
command result, artifact, or run record produced in this repository — never
narrative. Statuses: `not_started` / `in_progress` / `implemented` (local
tests pass) / `verified` (independent fresh-context review passed) /
`blocked(<on what>)`.

Branch: `phase2/real-colab-research`. Phase 1 behavior is preserved; every
interface migration is recorded in `docs/PHASE2_DECISIONS.md`.

Last updated: 2026-07-12.

## Security preconditions (done first)

| Item | Status | Evidence |
|---|---|---|
| `api.env` gitignored | done | `git check-ignore -v api.env` → `.gitignore:3:*.env` |
| `api.env` untracked | done | `git ls-files --error-unmatch api.env` exits 1 |
| `api.env` mode 600 | done | `ls -la` → `-rw-------` |
| Secret scanner on every commit | active | `scripts/check_staged.py` exit 0 on phase2 commits |

## Work packages

| WP | Scope | Target files | Depends on | Human gate | Status |
|---|---|---|---|---|---|
| META | Tea Time research pause + program guardrail | `skills/reflection/`, `research/program_guardrail.py`, `configs/research/program_guardrail.yaml` | — | — | implemented locally; focused/full gates pending this change |
| WP0 | Preregistration schema, experiment matrix, pilot config | `src/mlip_research_agent/research/{preregistration,experiment_matrix,pilot_status}.py`, `configs/research/cu_mace_al_pilot.yaml`, `docs/research/phase2_preregistration.md` | — | H2 before results | in_progress (WP1/WP2 values pinned; WP3/WP4 TBD) |
| WPS | Secure `api.env` loader + Materials Project SKILL | `src/mlip_research_agent/secrets/api_env.py`, `skills/data/materials_project/` | — | — | implemented locally (synthetic-key tests only; no live API call) |
| WP1 | Dataset due diligence, registry, qualification (Zuo 2019 Cu candidate) | `src/mlip_research_agent/data/{registry,manifests,qualification}.py`, `scripts/data/fetch_cu_benchmark.py`, `data_registry/datasets/cu_phase2/` | WPS (recon only) | **H1 approved** | implemented locally; 11 checks pass; energy/force promoted, stress excluded |
| WP2 | Leakage-safe grouped split + simulated oracle | `data/{split,oracle}.py`, `skills/active_learning/oracle_reveal/` | WP1 | — | implemented locally; independent leakage review passed |
| WP3 | Pinned MACE inference | `skills/mlip/mace_inference/`, `mace` extra pins | WP1 fixture | — | implemented locally + Colab-staged (CPU/GPU float64 parity on L4: max energy delta 0.0 eV, max force delta 1.14e-15 eV/Å; H2 checkpoint selection still pending) |
| WP5 | Independent evaluation suite | `skills/evaluation/mlip_metrics/` (+ learning_curve, calibration) | WP2, WP3 | — | implemented locally (audit repairs landed: val/test-only with one-shot protected authorization, unit gates, aggregates-only output, claim-binding metadata; 19 tests) |
| WP4 | Real MACE fine-tuning | `skills/mlip/mace_finetune/` | WP3 | — | implemented locally + Colab-staged (full SKILL contract; boundary test: one real optimizer step changes 26 tensors; checkpoint round-trip resumes at epoch boundary on CPU and L4; NaN→LR/OOM→batch repair taxonomy tested) |
| WP6 | Random + ensemble-UQ + diversity acquisition | `skills/active_learning/{ensemble_uq,diversity_select,decision_gate}/` | WP2, WP4 | — | not_started |
| WP7 | Multi-round controller + stopping rules | `research/` controller, round state schema | WP4–WP6 | — | not_started |
| WP8 | `bounded_research_pilot` policy + Colab research runner | `compute/*`, `configs/compute_policy.yaml`, `scripts/colab/{run_research.py,bootstrap_research.sh,colab_cli_run.sh}` | WP7 | **H3** before real run | not_started |
| WP9 | Claim classes, evidence bundle, pilot report, VESSL spec | `schemas/claims.py` extension, `docs/research/{human_gates,colab_pilot_runbook,vessl_replication_spec}.md` | all | H5 for claim release | in_progress (claim classes/evidence tiers implemented; remaining bundle/report/spec deferred) |

## Acceptance tests per WP (from plan_phase_2.md §10, §20)

- **WP0**: preregistration serializes deterministically; its SHA-256 is
  registered before any result-producing run; matrix fixes arms/budgets/seeds.
- **WPS**: loader never returns/logs key material (tests use synthetic keys
  only); tracked-file check; real MP authentication health check succeeds
  locally (status boolean only).
- **WP1**: hash mismatch aborts; malformed cell / missing forces / non-finite
  labels rejected; unit parity; deterministic duplicate detection; raw data
  immutable; qualification report generated; H1 recorded before promotion.
- **WP2**: zero source-group overlap across protected boundaries; acquisition
  cannot read hidden labels or test IDs; budget overrun rejected; reveal
  idempotent; split bytes reproduce from seed + manifest.
- **WP3**: checkpoint SHA-256 mismatch aborts; finite E/F on fixture;
  translation/rotation/atom-order invariance–equivariance within declared
  tolerance; explicit dtype/device; provenance manifests complete.
- **WP5**: hand-calculated golden metric fixtures; NaN/missing predictions
  fail; every reported metric links to a claim + artifacts.
- **WP4**: one real optimizer step changes trainable parameters; test data
  never reach training; resume from checkpoint; invalid E0 policy fails;
  failed run emits failure artifact, never a model artifact.
- **WP6**: selectors never access labels; equal budgets across arms;
  seed-reproducible random arm; diversity increases on synthetic fixture;
  machine-readable selection reasons.
- **WP7**: max rounds and label budget enforced; interrupted run resumes at
  round boundary; arm isolation; preregistered method immutable.
- **WP8**: L4/A100-40 accepted, unknown denied, mismatch denied; CPU-only mode
  is loudly non-scientific; dataset/model hash mismatch denied; partial
  completion cannot be marked complete.
- **WP9**: every pilot claim passes artifact verification; report status is
  `pilot_only`; VESSL replication spec frozen.

## Gate status

| Gate | Meaning | Status |
|---|---|---|
| H1 | Dataset promotion | approved for recorded energy/force dataset and whole-family split; stress excluded |
| H2 | Preregistration approval | deliberately not frozen; all owner-required evidence is now recorded (second checkpoint memo, D0 E0 diagnostic, real optimizer step + round-trip, Linux/Colab pins from the staging record) — awaiting owner selection/freeze |
| H3 | Remote execution approval | not reached (staging passed; H3 packet can now be assembled) |
| H4 | Scientific pivot | n/a |
| H5 | Claim release | not reached |
| Staging | Bounded Colab integration test | **executed and passed 2026-07-12** on one NVIDIA L4 at commit `467ad6a` via the colab CLI driver: attestation allow, pinned checkpoint (2ddb079c…) resolved, real-inference CPU/GPU float64 parity, one optimizer step, checkpoint round-trip, 13-file hash-verified pull-back (`artifacts/colab_staging/pullback-467ad6ab/`); VM released |

## Repository capability ladder

mock-only → real-inference capable → real-fine-tuning capable →
Colab-staging capable → Colab-pilot complete → VESSL-replicated →
publication-claim eligible

**Current: Level 4 — bounded Colab staging complete.** Real pinned MACE
inference and one controlled fine-tuning optimizer step (with checkpoint
round-trip) have run on a policy-attested NVIDIA L4 via the colab CLI
driver, with hash-verified artifact pull-back. No full pilot, scientific
fine-tuning result, or publication-eligible claim has occurred; boundary
runs use synthetic fixtures, never the protected Cu partitions.

Evidence (2026-07-12, through commit `8247a68` + staging record):
default `pytest` 281 passed / 7 skipped (opt-in real-model + optional-dep
paths) / 2 remote deselected; opt-in real MACE CPU fine-tune fixture passes
(one step + resume); Ruff and strict mypy (195 files) clean; tracked-file
secret/large-file scan clean. Colab staging record
`artifacts/colab_staging/pullback-467ad6ab/staging-*/staging_record.json`:
all five stages pass on NVIDIA L4, Linux pins torch 2.11.0+cu128 /
mace-torch 0.3.16 / e3nn 0.4.4 / numpy 2.0.2 (python 3.12). Scientific
approval: H1 granted; H2/H3/H5 remain open.
