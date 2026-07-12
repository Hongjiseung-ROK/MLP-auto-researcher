# Phase 2 Open Questions

Owner decisions that gate Phase 2 steps. Implementation proceeds around them;
nothing claim-eligible ships across an unanswered gate. (General project
questions remain in `docs/OPEN_QUESTIONS.md`.)

**P2-Q1 (Gate H1). Dataset promotion.**
Approve the qualified Cu dataset: source, license/redistribution terms,
units, level of theory, quality metrics, leakage-risk assessment, and the
qualified manifest hash. Presented after WP1 produces
`data_registry/datasets/cu_phase2/qualification_report.json`.
Status: **resolved — owner approved energy/force use with documented
limitations on 2026-07-11; stress remains excluded.** Approval subject:
qualified dataset content SHA-256
`bc9b78ca5e3b95f91bf34bbc3641a3d6e3f92338b4e3d97065165157848cfc48`,
normalized-manifest file SHA-256
`ac3655e41ce4327ba18e2a603866b97b00b839ff04f16bb2cb86a3d8ba8a70d3`,
and whole-family split-manifest SHA-256
`80d9b95083ebba9d8f988a461525b326ba807c79da834b066d1917afc9d8a15e`.

**P2-Q2 (Gate H2). Preregistration.**
Approve `docs/research/phase2_preregistration.md` (research question, primary
endpoint = force MAE eV/Å on frozen test set, four arms, budgets, split rule,
checkpoint, seed schedule, stopping rules). The preregistration hash must be
recorded before the first result-producing run.
Status: **open by explicit owner decision — do not freeze yet.** WP1/WP2 are
pinned. Two checkpoints are content-addressed with D0-only raw energy-offset
evidence recorded without selection. The remaining prerequisites are now
complete (2026-07-12): a real optimizer step + checkpoint round trip passed
locally (CPU) and on a policy-attested Colab L4, and the Linux/Colab pins are
recorded in the staging record (torch 2.11.0+cu128, mace-torch 0.3.16,
e3nn 0.4.4). The H2 packet is ready to assemble; checkpoint selection and
preregistration freeze remain owner decisions.

**P2-Q3 (Gate H3). First remote execution.**
Approve exact commit, `bounded_research_pilot` workload, requested GPU (L4),
60-minute runtime, artifact pull-back destination. Presented only after all
local gates and the Colab staging test pass.
Status: **open — staging passed (D-P2-19), so the H3 packet can now be
assembled and presented; no pilot authorization exists yet.**

**P2-Q4. Colab staging run authorization.**
The manual staging gate (real MACE import, checkpoint download + hash, one
inference batch, one optimizer step, checkpoint round-trip on a real Colab
GPU) consumes a small amount of Colab Pro compute before H3. Plan §15.2
treats it as human-triggered. Does the owner authorize the staging run as
soon as it is ready, or require a separate ask?
Recommended default: authorize staging when local gates pass, since OQ-14
already approved Pro-tier preflights ≤60 min.
Status: **resolved and consumed** — bounded staging was authorized for one
L4, one GPU, at most 60 minutes, with one A100-40GB fallback only after
verified OOM, and the authorization was consumed by the completed 2026-07-12
staging run at commit `467ad6a` (D-P2-19). Any new remote execution,
including an infrastructure replay at a later commit, requires a new explicit
owner authorization. This never authorized a full acquisition campaign and
does not satisfy H3.

**P2-Q5. Local `mace` extra installation.**
Real CPU-boundary tests (WP3/WP4) require `mace-torch` + CPU torch in the
`mlip-research-agent` conda env on this Mac (several-GB install, macOS arm64
wheels). CLAUDE.md forbids co-installing *multiple* MLIP extras, not one.
Recommended default: install the pinned `mace` extra locally; keep remote
tests opt-in.
Status: **resolved. Local installation complete:**
`mace-torch 0.3.16`, conda-forge `torch 2.12.1`; real CPU inference fixture
passes. Colab/Linux pins and CPU/GPU parity remain WP3 staging work.

**P2-Q7. Exact-commit Auto Research infrastructure replay.**
One Colab CLI session may run two connected real-data/real-MACE infrastructure
iterations on exactly one NVIDIA L4 for at most 60 minutes, pausing after the
first iteration for three aggregate-only specialist reviews. A100 fallback,
notebooks, protected partitions, acquisition-pool labels, checkpoint promotion,
and claim-bearing output are forbidden. This authorization is distinct from H3
and does not authorize a full active-learning pilot.
Status: **authorized once by the owner for the exact implementation commit
produced on `phase3/remote-mlip-loop`; not yet consumed.** A transport-only
reconnect may target the same live session only when it is proven not to repeat
scientific execution.

**P2-Q6. Dataset redistribution.**
If mlearn's license permits redistribution, tiny fixtures (≤ a few
configurations) are committed for tests; the full dataset is always fetched
by the downloader with checksum verification, never committed. If the license
is unclear, fixtures become synthetically perturbed derivatives and the
downloader remains the only acquisition path.
Status: **resolved — BSD-3-Clause permits the committed four-record fixture
with notice retention; full data remain fetched and gitignored.**
