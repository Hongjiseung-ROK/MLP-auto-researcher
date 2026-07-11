# Implementation Status

Last updated: 2026-07-11 (compute-provider milestone).

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
| Colab execution package (`scripts/colab/`, notebook, `configs/colab/preflight.yaml`) | Delegated to subagent, integrating |
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

## Not implemented (deliberately)

- Any real MLIP backend, DFT execution, LLM call, network access, GPU code.
- Multi-iteration AL loop / decision gate (D-11).
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

## Next proposed step

Implement the first real SKILL behind an optional extra: **MACE fine-tuning
adapter** (`skills/mlip`, `mace` extra) with CPU smoke-test mode, pending
OQ-3/OQ-4 approval. Rationale: it exercises the executor against a real tool
without licenses or credentials, and unblocks the AL loop design.
