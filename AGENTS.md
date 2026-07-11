# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

An autonomous, evidence-grounded AI researcher for ML interatomic potentials.
`plan.md` defines the long-term architecture; `plan_phase_2.md` defines the current
real-model and bounded-Colab contract. Do not rewrite either without owner approval.
Phase 1 remains a deterministic mock vertical slice. Phase 2 is at the local
real-model boundary and is preparing bounded staging; it is not a completed Cu
study, production research system, or publication-claim pipeline.

The mlearn Cu/MACE path is a pipeline-hardening scaffold. Keep reusable controller,
evaluation, provenance, and compute interfaces model- and benchmark-agnostic.
Record consequential scientific choices in `docs/PHASE2_DECISIONS.md` and unresolved
owner gates in `docs/PHASE2_OPEN_QUESTIONS.md`.

## Commands

All commands run through the `mlip-research-agent` conda env. This host exports a
global `PYTHONPATH` pointing at an unrelated chemsmart checkout — prefix commands
with `env -u PYTHONPATH` here (harmless on clean machines).

```bash
env -u PYTHONPATH conda run -n mlip-research-agent python -m pytest            # full suite
env -u PYTHONPATH conda run -n mlip-research-agent python -m pytest tests/test_executor.py -k resume   # single test
env -u PYTHONPATH conda run -n mlip-research-agent python -m ruff check .      # lint (autofix: --fix)
env -u PYTHONPATH conda run -n mlip-research-agent python -m mypy src          # strict typing gate
env -u PYTHONPATH MPLCONFIGDIR=/tmp/mlip-mpl \
  MACE_TEST_CHECKPOINT="$PWD/data/cache/models/2023-12-10-mace-128-L0_energy_epoch-249.model" \
  conda run -n mlip-research-agent python -m pytest \
  src/mlip_research_agent/skills/mlip/mace_inference/tests/test_mace_inference.py -q
```

All three gates must pass before integrating anything real. Never weaken or delete
tests to make the suite pass. pytest collects both top-level `tests/` and per-skill
`src/**/tests/` (configured in pyproject `testpaths`).

Setup / demo:

```bash
conda env create -f environment.yml
env -u PYTHONPATH conda run -n mlip-research-agent python -m pip install -e .
env -u PYTHONPATH conda run -n mlip-research-agent mlip-agent run \
  --campaign configs/example_campaign.yaml --output-root artifacts/runs
```

`mlip-agent` also has `validate`, `env-report`, and `run --resume <run_dir>`.

## Architecture

Execution pipeline (one pass, wired in `workflows/compiler.py`):

```
CampaignSpec (schemas/campaign.py, YAML-validated)
  → compile_campaign() → WorkflowSpec DAG (schemas/workflow.py, cycle-checked)
  → WorkflowExecutor (runtime/executor.py)
      literature → structures → selection → labeling → training → evaluation → verification
  → provenance bundle (provenance/bundle.py) → run report (provenance/report.py)
```

Key contracts that everything obeys:

- **SKILL contract** (`skills/base.py`): every skill is a class with `name`,
  `input_model`/`output_model` (pydantic), and `run(inputs, ctx)`. The executor —
  not the skill — validates inputs/outputs. Skills write files only under
  `ctx.step_dir`, register every produced file in `ctx.registry`, and fail only by
  raising `SkillError` with a `FailureClass`, severity, retryability, and optional
  `repair_params`. Each skill directory must contain `SKILL.md`, `schema.py`,
  `implementation.py`, `validators.py`, `tests/`, `examples/`. Registration happens
  via the `@register_skill` decorator; `skills/base.py::_ensure_builtin_skills_loaded`
  imports the skill packages for their side effects — add new skill packages there.
- **Failure/recovery** (`runtime/recovery.py`): failures map to bounded
  RETRY/REFINE/PIVOT/ESCALATE/ABORT decisions. `repair_params` on a retryable error
  triggers REFINE: the executor merges them into the next attempt's inputs and
  records them in the event log and failure record — parameters never change
  silently. Retries are bounded by the campaign's `budget.max_retries_per_step`.
- **Events and checkpoints**: the executor emits an append-only JSONL event log
  (`runtime/events.py`) and saves a checkpoint plus `manifest.json` and
  `claims.json` after every step, so the verification skill and resume both read
  real on-disk state. Resume validates the workflow hash before skipping steps.
- **Claims**: numerical results are registered as `Claim`s referencing artifact ids;
  the final `claim_verification` step rejects any claim whose artifacts are missing
  or hash-mismatched (strict mode fails the run, evidence artifacts are registered
  before the failure raises). The run report renders verified claims only — agent
  text is never evidence.
- **Determinism/reproducibility**: registered artifacts, manifest, and claims must
  be byte-identical across reruns with the same campaign+seed (timestamps live only
  in the event log and provenance bundle). This is why structures use the
  first-party JSON format in `skills/atomistics/structures_io.py` instead of extxyz,
  and why manifest entries carry no timestamps. Seeds come only from `ctx.seed`
  (`np.random.default_rng`), never global state. The end-to-end test enforces this.
- **Step wiring**: workflow step params reference upstream outputs as
  `"$steps.<step_id>.<output_field>"`, resolved by the executor against completed
  step outputs.

Compute layer (`compute/`): scientific skills never talk to providers directly —
they submit typed `JobSpec`s through `ComputeRouter`, which enforces
`configs/compute_policy.yaml` fail-closed (GPU allowlist L4/A100-40/A100-80, exact
alias normalization in `compute/policy.py`, unknown devices denied). Remote jobs
run only after an immutable runtime attestation (`compute/attestation.py`) gets
ALLOW; VESSL (primary) submissions additionally require a passing, non-stale Colab
preflight record matching commit/deps/backend/CUDA-class/schema
(`compute/preflight.py`). Real transports are not wired yet: `MockRemoteTransport`
in `compute/_remote.py` simulates sessions; the real Colab transport must wrap the
security-reviewed `google-colab-cli` v0.6.0 (`docs/SKILL_SECURITY_REVIEW.md`) and
keep auth external (gcloud ADC). Attestation/compute artifacts carry timestamps —
never wire compute skills into byte-reproducibility-gated scientific DAGs.
`pytest -m colab_remote` / `-m vessl_remote` markers are opt-in real-resource
tests; ordinary CI runs mocks only.

Phase 1 campaign allowlists remain mock-only. Phase 2 real capabilities use separate
typed research/SKILL boundaries instead of silently widening those stable schemas.
MACE is the approved mandatory baseline; its checkpoint remains an H2 candidate
until the second-checkpoint and E0 review finishes. Install at most one MLIP extra
per environment because torch/e3nn pins may conflict.

## Phase 2 approval and safety boundaries

- H1 is approved for the recorded Cu energy/force dataset and split hashes; stress
  remains excluded because its unit is inferred.
- H2 is not frozen. Do not promote a checkpoint, freeze preregistration, or start a
  claim-bearing run without the exact approval artifact.
- Bounded Colab staging is authorized: one GPU, L4 first, at most 60 minutes, and a
  single A100-40GB fallback only for verified OOM. Full campaigns remain forbidden.
- `api.env` is sealed: never read, print, hash, upload, pass to a subagent, or place
  it in a command argument/artifact. Only verify ignore/tracking status and mode 600.
- Never commit raw datasets, cached checkpoints, secrets, notebook outputs, or run
  caches. Run `scripts/check_staged.py` before every commit.
- Tea Time is a deterministic reflection artifact, never evidence. Invoke it at the
  pause points declared in its SKILL contract before irreversible actions.

## mypy note

The project is mypy `strict`. ASE is partially typed: modules wrapping it
(`skills/atomistics/*`, `skills/dft/*`) have a `disallow_untyped_calls = false`
override in pyproject rather than per-line ignores.
