# MLIP Research Agent

Autonomous active-learning scientist for ML interatomic potentials —
bootstrap vertical slice. Architectural source of truth: [`plan.md`](plan.md).

The current milestone is a fully mock, fully deterministic research campaign:

```text
research goal → validated campaign spec → workflow DAG → mock skills
→ event log → artifact registry → claim verification → reproducible report
```

It runs without GPU, DFT binaries, paid APIs, or datasets, and demonstrates
bounded failure recovery, checkpoint resume, and byte-reproducible artifacts.

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

## Where things live

- `src/mlip_research_agent/skills/` — bounded, typed SKILLs; each directory
  carries `SKILL.md`, `schema.py`, `implementation.py`, `validators.py`,
  `tests/`, `examples/`.
- `src/mlip_research_agent/runtime/` — event-sourced executor, checkpoints,
  bounded recovery policy.
- `docs/OPEN_QUESTIONS.md` — decisions awaiting the project owner (scientific
  target, MLIP backend, DFT backend, GPU budget, LLM provider, …). Real
  scientific integrations are gated on these.
- `docs/IMPLEMENTATION_DECISIONS.md`, `docs/IMPLEMENTATION_STATUS.md` —
  decision log and current gap analysis vs `plan.md`.
- `DEPENDENCIES.md` — full dependency policy; heavy backends are optional
  extras (`pip install -e ".[mace]"` etc.), never bootstrap requirements.
