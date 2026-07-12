# Implementation Status

Last updated: 2026-07-12 (Phase 2 Level 4.5 — bounded Colab staging and the
local synthetic Auto Research loop verified; PR #3 merged to `main` as
`99a7a724`).

Phase 1 below remains the deterministic regression baseline. Current Phase 2
truth is tracked in `docs/PHASE2_IMPLEMENTATION_STATUS.md`: qualified data,
leakage-safe split/oracle, claim tiers, pinned real MACE inference, the real
fine-tuning boundary, and the independent aggregate evaluator are implemented;
the bounded Colab staging run passed on an attested L4 (commit `467ad6a`,
`staging_only`). WP6 contracts and the resumable WP7 controller are implemented,
locally tested, and exercised by an independently graded connected two-iteration
synthetic trace. The bounded remote real-MACE replay completed at exact commit
`ec2d804` on one L4: two connected independently rejected iterations, host
specialist review between them, passing trace grade, hash-verified pull-back,
and stable cleanup. This is infrastructure evidence only; H2/H3/H5 remain open.

## Milestone 2 — compute providers and publication: **in final validation**

| Deliverable | Status |
|---|---|
| Provider-neutral compute interface (`compute/base.py` Protocol) | Done |
| GPU policy, alias normalization, fail-closed evaluation (`compute/policy.py`, `configs/compute_policy.yaml`) | Done — L4/A100-40/A100-80 allowlist per owner authorization |
| Runtime attestation, immutable records (`compute/attestation.py`) | Done |
| Preflight store + staleness gate (`compute/preflight.py`) | Done |
| Router with audit log, override control, VESSL preflight gate (`compute/router.py`) | Done |
| Local/Colab/VESSL adapters + mock transports (`local.py`, `_remote.py`, `colab.py`, `vessl.py`) | Done (mock transports; real transports gated on auth/budget answers) |
| Colab CLI security review (google-colab-cli v0.6.0, pinned) | Done — `docs/SKILL_SECURITY_REVIEW.md`, ACCEPT with caveats |
| Workspace skill `colab_preflight` (full contract + security.md) | Done |
| Reflection skill `tea_time_with_reading_poem` (full contract, agent-text only) | Done |
| Colab execution package (`scripts/colab/`, CLI launcher, `configs/colab/preflight.yaml`) | Done — notebooks retired 2026-07-12; `colab_cli_run.sh` is the launcher |
| Compute test suite (§11 acceptance list, `colab_remote`/`vessl_remote` markers) | Delegated to subagent, integrating |
| Secret/large-file staging gate (`scripts/check_staged.py`) | Done |
| GitHub publication to Hongjiseung-ROK/MLP-auto-researcher | Pending final gates |
| Real Colab preflight / real VESSL run | **Not run** — blocked on owner budget/auth answers (by design) |

## Milestone 1 — mock vertical slice: **complete** (2026-07-10)

## Milestone 1 — mock vertical slice: **complete**

| Deliverable | Status | Evidence |
|---|---|---|
| Dependency specification | Done | `DEPENDENCIES.md` |
| Minimal conda env (`mlip-research-agent`) | Done, created and verified | `artifacts/bootstrap/environment-report.json` |
| Packaging with optional dependency groups | Done | `pyproject.toml` |
| Campaign spec schema + validation | Done | `schemas/campaign.py`, `tests/test_campaign_schema.py` |
| Declarative workflow DAG + compiler | Done | `schemas/workflow.py`, `workflows/compiler.py` |
| Event-sourced executor, checkpoint resume | Done | `runtime/`, `tests/test_executor.py` |
| Failure object + bounded RETRY/REFINE/PIVOT/ESCALATE/ABORT | Done | `schemas/failure.py`, `runtime/recovery.py`, `tests/test_recovery.py` |
| Artifact registry (sha256 manifest) | Done | `artifacts/registry.py`, `tests/test_artifact_registry.py` |
| 6 SKILLs with full contract (SKILL.md, schema, impl, validators, tests, examples) | Done | `skills/{literature,atomistics,active_learning,dft,mlip,verification}/` |
| Claim registry + verification gate | Done | `schemas/claims.py`, `skills/verification/` |
| Provenance bundle (local files) | Done | `provenance/bundle.py` |
| Environment report generator | Done | `provenance/environment_report.py` |
| Markdown run report | Done | `provenance/report.py` |
| CLI (`mlip-agent validate/run/env-report`) | Done | `cli.py` |
| First execution target (demo campaign, injected failure, recovery, reproducible outputs) | Done, executed twice, manifests byte-identical | `configs/example_campaign.yaml`, `artifacts/runs/` |
| Testing gates | Passing: 62 tests, ruff clean, mypy strict clean | run commands below |

## Gate commands

```bash
env -u PYTHONPATH conda run -n mlip-research-agent python -m pytest
env -u PYTHONPATH conda run -n mlip-research-agent python -m ruff check .
env -u PYTHONPATH conda run -n mlip-research-agent python -m mypy src
```

(`env -u PYTHONPATH` guards against this host's global chemsmart PYTHONPATH;
on clean machines plain `conda run` works.)

## Not implemented (deliberately or still in Phase 2)

- A completed real-data remote Auto Research replay, DFT execution, a full
  active-learning campaign, or a completed GPU research pilot.
- Claim-bearing model selection or checkpoint promotion; the replay target is
  infrastructure-only.
- OpenHands runtime integration (OQ-8), MLflow/DVC/RO-Crate adapters (OQ-9).
- Committee mode, UQ beyond the mock proxy, molecular branch.

## Gap analysis vs plan.md

| plan.md element | Status | Blocking dependency |
|---|---|---|
| OpenHands execution kernel | Pattern implemented first-party | OQ-8 decision |
| PARNESS-style YAML DAGs | Implemented (compiler-generated; hand-written DAGs also validate) | — |
| Lang2MLIP decision loop | Stub planner only | OQ-7 (LLM provider) |
| AL controller (uncertainty/diversity/viability) | Mock proxy only | OQ-3 backend + real UQ |
| Foundation MLIP fine-tuning | Not started | OQ-3/OQ-4, GPU (OQ-6) |
| DFT labeling | Mock only | OQ-5 (ORCA found on host; periodic backend undecided) |
| Knowledge layer (PDF indexing, cross-run lessons) | Fixture corpus only | post-v0 |
| Writer/reviewer agents | Report renderer only (by design: verification precedes writing) | post-v0 |
| MLflow/DVC/RO-Crate | Local provenance bundle instead | OQ-9 |
| Vessel Cloud CI (nightly GPU) | Not started | OQ-6, OQ-11 |

## Next Phase 2 sequence

Validate the exact-commit real-data/real-MACE Auto Research pipeline in one
Colab L4 session, with a host review pause between two connected iterations,
then retain the result as infrastructure evidence only. H2 remains unfrozen;
H3 and H5 remain open.
