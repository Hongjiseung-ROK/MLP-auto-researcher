# Phase 2 Open Questions

Owner decisions that gate Phase 2 steps. Implementation proceeds around them;
nothing claim-eligible ships across an unanswered gate. (General project
questions remain in `docs/OPEN_QUESTIONS.md`.)

**P2-Q1 (Gate H1). Dataset promotion.**
Approve the qualified Cu dataset: source, license/redistribution terms,
units, level of theory, quality metrics, leakage-risk assessment, and the
qualified manifest hash. Presented after WP1 produces
`data_registry/datasets/cu_phase2/qualification_report.json`.
Status: **open — due diligence in progress.**

**P2-Q2 (Gate H2). Preregistration.**
Approve `docs/research/phase2_preregistration.md` (research question, primary
endpoint = force MAE eV/Å on frozen test set, four arms, budgets, split rule,
checkpoint, seed schedule, stopping rules). The preregistration hash must be
recorded before the first result-producing run.
Status: **open — draft written, values pending WP1/WP3 due diligence.**

**P2-Q3 (Gate H3). First remote execution.**
Approve exact commit, `bounded_research_pilot` workload, requested GPU (L4),
60-minute runtime, artifact pull-back destination. Presented only after all
local gates and the Colab staging test pass.
Status: **open — not yet presentable.**

**P2-Q4. Colab staging run authorization.**
The manual staging gate (real MACE import, checkpoint download + hash, one
inference batch, one optimizer step, checkpoint round-trip on a real Colab
GPU) consumes a small amount of Colab Pro compute before H3. Plan §15.2
treats it as human-triggered. Does the owner authorize the staging run as
soon as it is ready, or require a separate ask?
Recommended default: authorize staging when local gates pass, since OQ-14
already approved Pro-tier preflights ≤60 min.
Status: **open.**

**P2-Q5. Local `mace` extra installation.**
Real CPU-boundary tests (WP3/WP4) require `mace-torch` + CPU torch in the
`mlip-research-agent` conda env on this Mac (several-GB install, macOS arm64
wheels). CLAUDE.md forbids co-installing *multiple* MLIP extras, not one.
Recommended default: install the pinned `mace` extra locally; keep remote
tests opt-in.
Status: **proceeding under recommended default unless the owner objects**
(reversible; recorded here for transparency).

**P2-Q6. Dataset redistribution.**
If mlearn's license permits redistribution, tiny fixtures (≤ a few
configurations) are committed for tests; the full dataset is always fetched
by the downloader with checksum verification, never committed. If the license
is unclear, fixtures become synthetically perturbed derivatives and the
downloader remains the only acquisition path.
Status: **open — resolved by WP1 license review.**
