# MLIP Research Agent

Autonomous, evidence-grounded AI researcher for ML interatomic potentials.
Architectural sources of truth: [`plan.md`](plan.md) and
[`plan_phase_2.md`](plan_phase_2.md).

The repository preserves a fully deterministic Phase 1 mock campaign. Phase 2 has reached Level 4 infrastructure capability: bounded Colab CLI staging has passed.

Not yet achieved:
- H2-frozen protocol
- full scientific pilot
- autonomous acquisition loop
- VESSL replication
- publication claim

Locally verified Phase 2 capabilities include a qualified Cu energy/force benchmark, a whole-group split, a hidden-label simulated oracle, scientific evidence tiers, pinned MACE CPU inference, WP4 boundary training, and WP5 independent aggregate evaluation. Cu is a bounded pipeline-hardening scaffold rather than the final research topic.

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
