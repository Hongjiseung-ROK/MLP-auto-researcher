# Dependency Specification

Status legend — **Confirmed**: included in the bootstrap environment or an optional
group with a settled installation path. **Needs approval**: listed for planning, but
inclusion, channel, or license must be confirmed by the project owner before
installation (see `docs/OPEN_QUESTIONS.md`).

The bootstrap environment (`environment.yml`, env name `mlip-research-agent`) is
deliberately minimal: repo development, schema validation, orchestration skeleton,
small-structure manipulation, tests, lint, typing. Everything heavy lives behind
optional dependency groups in `pyproject.toml` and is **not** installed by default.

---

## 1. Bootstrap and packaging

| Package | Purpose | Required? | Channel | Version constraint | License | System req | Status |
|---|---|---|---|---|---|---|---|
| python | Interpreter | Required | conda-forge | `3.11.*` | PSF | any | Confirmed |
| pip | Package installer | Required | conda-forge | `>=24` | MIT | any | Confirmed |
| setuptools | Build backend for this package | Required | conda-forge (build-time) | `>=68` | MIT | any | Confirmed |
| conda | Environment manager | Required (tool) | system install | `>=23` | BSD-3 | any | Confirmed (host has 26.1.1) |

Python 3.11 chosen over 3.12/3.13 because the MLIP ecosystem (MACE, SevenNet,
e3nn/torch pins) still lags on newest interpreters; revisit at first real backend
integration.

## 2. Agent runtime

| Package | Purpose | Required? | Channel | Version constraint | License | System req | Status |
|---|---|---|---|---|---|---|---|
| openhands-sdk (or openhands-ai) | Execution substrate per plan.md (typed tools, event-sourced runs, sandboxed workspaces) | Optional (`agent` group) | pip | unpinned until confirmed | MIT | Docker for sandboxed mode | **Needs approval** (hard dependency vs. architectural reference — OQ-8) |
| anthropic | LLM API client (planner/failure-triage reasoning) | Optional (`agent`) | pip | `>=0.40` | Apache-2.0/proprietary API | API key | **Needs approval** (provider choice — OQ-7) |
| openai | Alternative LLM client (plan.md cites GPT-5.6 Sol guidance) | Optional (`agent`) | pip | `>=1.50` | Apache-2.0/proprietary API | API key | **Needs approval** (OQ-7) |
| litellm | Provider-agnostic LLM routing | Optional (`agent`) | pip | `>=1.40` | MIT | none | **Needs approval** (only if multi-provider is desired) |

The v0 vertical slice uses a deterministic stub planner and **no LLM at runtime**,
so none of these are in the bootstrap env.

## 3. Workflow and schema validation

| Package | Purpose | Required? | Channel | Version constraint | License | System req | Status |
|---|---|---|---|---|---|---|---|
| pydantic | Typed schemas for campaigns, workflows, events, failures, claims | Required | conda-forge | `>=2.6,<3` | MIT | any | Confirmed |
| pyyaml | Campaign/workflow YAML parsing | Required | conda-forge | `>=6,<7` | MIT | any | Confirmed |
| networkx | DAG utilities (only if hand-rolled topological sort proves insufficient) | Optional | conda-forge | `>=3` | BSD-3 | any | Not included — v0 uses a small built-in toposort |

## 4. Atomistic simulation

| Package | Purpose | Required? | Channel | Version constraint | License | System req | Status |
|---|---|---|---|---|---|---|---|
| ase | Structure I/O, small-structure manipulation, mock (Lennard-Jones) labeling | Required | conda-forge | `>=3.23,<4` | LGPL-2.1+ | any | Confirmed |
| numpy | Array math for mock metrics/potentials | Required | conda-forge | `>=1.26,<3` | BSD-3 | any | Confirmed |
| pymatgen | Materials Project interop, symmetry analysis | Optional (`atomistics`) | conda-forge | `>=2024.1` | MIT | any | Confirmed as optional |
| lammps | MD engine | Optional (`atomistics`, binary) | conda-forge (CPU) / source build (GPU) | latest stable | GPL-2.0 | GPU build needs CUDA toolchain | **Needs approval** for GPU build path |
| rdkit | Molecular branch cheminformatics | Optional (`atomistics`) | conda-forge | `>=2024.03` | BSD-3 | any | Deferred to molecular phase |

## 5. MLIP backends (all optional, mutually independent groups)

| Package | Purpose | Group | Channel | Version constraint | License | System req | Status |
|---|---|---|---|---|---|---|---|
| mace-torch | MACE foundation models — plan.md's recommended v1 default | `mace` | pip | `>=0.3.6` | MIT | torch; CUDA GPU strongly preferred for training | **Needs approval** (baseline choice — OQ-3/OQ-4) |
| chgnet | Charge-informed materials baseline | `chgnet` (add when needed) | pip | `>=0.3` | BSD-3 (model weights: check MP terms) | torch | Needs approval |
| sevenn | SevenNet multi-GPU MD | `sevennet` | pip | `>=0.10` | GPL-3.0 | torch, LAMMPS for parallel MD | Needs approval (GPL — infects derived distributions) |
| orb-models | ORB fast screening | `orb` | pip | `>=0.4` | Apache-2.0 (weights under ORB license terms) | torch | Needs approval |
| mattersim | Advanced materials arm | future group | pip | `>=1.0` | MIT | torch | Deferred (stretch branch) |
| grace-tensorpotential | GRACE comparison arm | future group | pip | check | ACE license | TensorFlow | Deferred (stretch branch) |
| torch | Shared DL runtime for all backends | pulled in by backend groups | pip | `>=2.2,<3` | BSD-3 | CPU works; CUDA wheels for Linux GPU | Confirmed as transitive-optional; never in bootstrap |

Per plan.md and operating rules, **no MLIP backend is a mandatory dependency**. The
v0 slice ships a deterministic mock trainer.

## 6. Active learning and uncertainty quantification

| Package | Purpose | Required? | Channel | Version constraint | License | Status |
|---|---|---|---|---|---|---|
| scikit-learn | Diversity scoring (clustering), calibration curves | Optional (`atomistics` or `al` future group) | conda-forge | `>=1.4` | BSD-3 | Deferred — v0 mock acquisition uses numpy only |
| scipy | Numerics for UQ metrics | Optional | conda-forge | `>=1.11` | BSD-3 | Deferred (numpy suffices for mocks) |

Committee/ensemble UQ arrives with the first real backend; no dedicated AL library
is planned (acquisition policies are first-party SKILLs).

## 7. Experiment tracking and data versioning

| Package | Purpose | Group | Channel | Version constraint | License | Status |
|---|---|---|---|---|---|---|
| mlflow | Params/metrics/model lineage tracking | `tracking` | pip | `>=2.14` | Apache-2.0 | **Needs approval** (server/remote config — OQ-9) |
| dvc | Data/pipeline versioning | `tracking` | pip | `>=3.50` | Apache-2.0 | **Needs approval** (remote choice — OQ-9) |
| dvc-s3 / dvc-gdrive / ... | DVC remote plugins | `tracking` | pip | match dvc | Apache-2.0 | Blocked on remote choice |

v0 provenance is local-file-based by design (plan.md §Reproducibility); these are
adapters added later.

## 8. Provenance and reproducibility

| Package | Purpose | Group | Channel | Version constraint | License | Status |
|---|---|---|---|---|---|---|
| rocrate | RO-Crate bundle emission | `provenance` | pip | `>=0.10` | Apache-2.0 | Confirmed as optional (adapter, post-v0) |
| prov | PROV-O document generation | `provenance` | pip | `>=2.0` | MIT | Confirmed as optional (adapter, post-v0) |

v0 emits a first-party `provenance.json` (stable schema, hashes, seeds, env
metadata) with no external dependency.

## 9. Testing, linting, typing, documentation

| Package | Purpose | Required? | Channel | Version constraint | License | Status |
|---|---|---|---|---|---|---|
| pytest | Test runner | Required | conda-forge | `>=8,<9` | MIT | Confirmed |
| pytest-cov | Coverage reporting | Required | conda-forge | `>=5` | MIT | Confirmed |
| ruff | Lint + import order | Required | conda-forge | `>=0.5` | MIT | Confirmed |
| mypy | Static typing gate | Required | conda-forge | `>=1.10` | MIT | Confirmed |
| types-PyYAML | Type stubs for yaml | Required (dev) | conda-forge | match pyyaml | Apache-2.0 | Confirmed |
| mkdocs-material | Docs site | Optional (`docs`) | pip | `>=9.5` | MIT | Confirmed as optional |
| mkdocstrings[python] | API docs from docstrings | Optional (`docs`) | pip | `>=0.25` | ISC | Confirmed as optional |

## 9b. Compute providers (remote execution)

| Tool | Purpose | Required? | Installation channel | License | Status |
|---|---|---|---|---|---|
| google-colab-cli (`colab`) | Colab preflight/validation sessions (validation provider) | Optional, external tool | `uv tool install google-colab-cli` / pip | Apache-2.0 (official googlecolab org) | **Reviewed and pinned v0.6.0** — docs/SKILL_SECURITY_REVIEW.md; auth via gcloud ADC, never bundled |
| gcloud SDK | ADC authentication for the colab CLI | Optional, external tool | Google installer | Apache-2.0 | Present on host; user-owned credentials |
| vessl (CLI) | Primary production compute (A100 allowlist) | Optional, post-approval | pip | Apache-2.0 | **Needs owner org/project/budget answers** before install/use |
| torch (in remote envs) | GPU attestation detail + future backends | Remote-env only | pip (CUDA wheels on provider VMs) | BSD-3 | Never in the local bootstrap env |

## 10. External binaries and licensed software

| Tool | Purpose | Required? | Installation channel | License consideration | Status |
|---|---|---|---|---|---|
| ORCA | DFT labeling (molecular/cluster) | Optional, post-v0 | Manual download, per-user academic license from FAccTs | **Free for academia but not redistributable; each user registers.** Never bundled, never in CI images. | **Needs approval** (DFT backend — OQ-5) |
| VASP | Periodic DFT labeling | Optional alternative | Site license | **Paid, per-group license required.** | Needs approval (OQ-5) |
| Quantum ESPRESSO | Open-source periodic DFT | Optional alternative | conda-forge (`qe`) or source | GPL-2.0, freely installable — the only DFT backend CI could ever run | Needs approval (OQ-5) |
| xTB | Semiempirical pre-screening | Optional | conda-forge (`xtb`) | LGPL-3.0 | Confirmed as optional, post-v0 |
| LAMMPS (GPU build) | Production MD | Optional | source build with CUDA | GPL-2.0 | Needs approval (hardware-dependent) |
| Docker | Sandboxed OpenHands runtime, release images | Optional | system install | Apache-2.0 (engine) | Needs approval with OQ-8 |
| git | Version control | Required (tool) | system | GPL-2.0 | Confirmed (host has 2.50.1) |

**v0 uses a mock DFT backend only.** No licensed binary is required to pass any
test or run the demonstration campaign.

## 11. Optional or mutually incompatible integrations

- **Torch-based MLIP groups (`mace`, `sevennet`, `orb`)** may pin conflicting
  `torch`/`e3nn` versions. They are separate extras; install one per environment,
  or use dedicated per-backend environments once real integrations begin. The
  committee mode will address this with per-backend subprocess isolation rather
  than co-installation.
- **GRACE** requires TensorFlow, which should never share an env with the torch
  backends. Deferred.
- **CUDA wheels** are Linux-only for most backends; this dev host is macOS arm64
  (CPU/MPS only). Real training targets Linux GPU nodes (Vessel Cloud per plan.md).
- **`all` extra** exists for completeness but intentionally excludes the
  conflicting MLIP backends; it is never installed by default.
