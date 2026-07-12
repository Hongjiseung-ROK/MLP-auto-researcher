# MLIP Research Agent

Autonomous, evidence-grounded AI researcher for ML interatomic potentials.
Architectural sources of truth: [`plan.md`](plan.md) and
[`plan_phase_2.md`](plan_phase_2.md).

The repository preserves a fully deterministic Phase 1 mock campaign while Phase 2
has reached Level 4.5: bounded Colab staging plus a locally trace-graded,
two-iteration synthetic Auto Research loop:

```text
research goal → preregistration + human gates → typed skills → artifact registry
→ leakage-safe data/oracle boundary → real-model/evaluation boundaries
→ verification → bounded remote staging → evidence-grounded report
```

Verified Phase 2 capabilities include a qualified Cu energy/force benchmark, a
whole-group split, a hidden-label simulated oracle, scientific evidence tiers,
pinned MACE inference, a real fine-tuning boundary, and an independent aggregate
evaluator. PR #3 merged the typed WP6 contracts and resumable WP7 controller to
`main` at `99a7a724`; the happy-path synthetic loop and its recovery path are
independently trace-graded. A bounded Colab staging run passed on a policy-attested NVIDIA L4
(commit `467ad6a`, CLI-driven, hash-verified pull-back, synthetic fixtures,
scientific status `staging_only`). The current target is an exact-commit,
infrastructure-only real-data/real-MACE replay on one Colab L4. Cu is a bounded pipeline-hardening scaffold
rather than the final research topic. No full pilot, scientific fine-tuning
result, or publication-grade claim has been completed; H2 (preregistration),
H3 (pilot authorization), and H5 (claim release) remain open owner gates.

## Setup

```bash
conda env create -f environment.yml          # env: mlip-research-agent
conda run -n mlip-research-agent python -m pip install -e .
```

## Run the demo campaign

```bash
conda run -n mlip-research-agent mlip-agent run \
  --campaign configs/example_campaign.yaml --output-root artifacts/runs
```

The campaign intentionally injects one recoverable mock-SCF failure in the
labeling step; the executor applies a recorded REFINE repair and completes.
Outputs land in `artifacts/runs/<run_id>/`: `events.jsonl`, `manifest.json`,
`claims.json`, `provenance.json`, `run_report.md`. Rerunning with the same
campaign and seed reproduces every registered artifact hash exactly.

Other commands: `mlip-agent validate --campaign …`, `mlip-agent env-report`,
`mlip-agent run --resume <run_dir>`.

## Development gates

```bash
conda run -n mlip-research-agent python -m pytest
conda run -n mlip-research-agent python -m ruff check .
conda run -n mlip-research-agent python -m mypy src
```

On this host, prefix all commands with `env -u PYTHONPATH` to prevent an unrelated
checkout from entering Python imports. Real MACE and remote-resource tests remain
explicitly opt-in.

## Where things live

- `src/mlip_research_agent/skills/` — bounded, typed SKILLs; each directory
  carries `SKILL.md`, `schema.py`, `implementation.py`, `validators.py`,
  `tests/`, `examples/`.
- `src/mlip_research_agent/runtime/` — event-sourced executor, checkpoints,
  bounded recovery policy.
- `configs/research/program_guardrail.yaml` — machine-readable statement that the
  current benchmark is a scaffold, including non-goals, claim ceiling, next gate,
  and criteria for leaving benchmark mode.
- `docs/PHASE2_OPEN_QUESTIONS.md` — current human scientific and remote-run gates.
- `docs/IMPLEMENTATION_DECISIONS.md`, `docs/IMPLEMENTATION_STATUS.md` —
  decision log and current gap analysis vs `plan.md`.
- `DEPENDENCIES.md` — full dependency policy; heavy backends are optional
  extras (`pip install -e ".[mace]"` etc.), never bootstrap requirements.
