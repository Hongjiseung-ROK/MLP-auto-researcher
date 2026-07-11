# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An autonomous active-learning scientist for ML interatomic potentials, built as a
mock-first vertical slice. `plan.md` is the architectural source of truth — do not
rewrite it without explicit owner approval. `docs/OPEN_QUESTIONS.md` lists decisions
that gate all real scientific integrations (MLIP backend, DFT backend, LLM provider,
GPU budget): **do not add real backends, paid APIs, dataset downloads, or GPU code
until the corresponding question is answered by the owner.** Record new material
uncertainties there; record notable design choices in `docs/IMPLEMENTATION_DECISIONS.md`.

## Commands

All commands run through the `mlip-research-agent` conda env. This host exports a
global `PYTHONPATH` pointing at an unrelated chemsmart checkout — prefix commands
with `env -u PYTHONPATH` here (harmless on clean machines).

```bash
env -u PYTHONPATH conda run -n mlip-research-agent python -m pytest            # full suite
env -u PYTHONPATH conda run -n mlip-research-agent python -m pytest tests/test_executor.py -k resume   # single test
env -u PYTHONPATH conda run -n mlip-research-agent python -m ruff check .      # lint (autofix: --fix)
env -u PYTHONPATH conda run -n mlip-research-agent python -m mypy src          # strict typing gate
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

Mock-only guards: campaign `backend`, labeling `method`, and training `model_name`
are validated against explicit allowlists (`"mock"` / `"mock_lj"` /
`"mock_mean_baseline"`). Real backends get added by extending those allowlists
behind optional pyproject extras (`mace`, `orb`, `dft`, …) — the MLIP extras have
conflicting torch pins, so never co-install them in one env.

## mypy note

The project is mypy `strict`. ASE is partially typed: modules wrapping it
(`skills/atomistics/*`, `skills/dft/*`) have a `disallow_untyped_calls = false`
override in pyproject rather than per-line ignores.
