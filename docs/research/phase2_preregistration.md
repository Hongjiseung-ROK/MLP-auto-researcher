# Phase 2 Preregistration — Cu MACE Active-Learning Pilot

**Status: DRAFT — not H2-approved. No result-producing run may start until
this document's values are frozen, its SHA-256 recorded, and the owner's H2
approval is stored in the run state.**

Fields marked `TBD(<source>)` are pinned by the named work package before H2.

## 1. Research question

Under a fixed target-label budget, does uncertainty-plus-diversity
acquisition reduce force error on held-out fcc Cu configurations more
efficiently than random acquisition when adapting a pinned MACE foundation
model?

## 2. Hypotheses

- **H-primary:** at equal initial set D₀, added-label budget B, model
  checkpoint, optimizer budget, and seed schedule, the hybrid policy yields
  lower force MAE on the frozen test set T and a better force-error learning
  curve than random acquisition.
- **H-null:** hybrid acquisition offers no reproducible benefit over random
  acquisition under the Phase 2 data and compute regime. A null result is a
  valid, reportable outcome.

## 3. Endpoints

- **Primary:** force MAE on frozen test set, eV/Å. Immutable after H2 without
  a recorded PIVOT + human approval (which starts a new campaign lineage).
- **Secondary:** energy MAE (eV/atom); force RMSE; P95 per-atom force error;
  fraction of structures over the force-error threshold of
  `TBD(WP1 — set from dataset error scale)` eV/Å; Spearman correlation of
  ensemble disagreement vs observed force error; learning curve
  (error vs labels revealed) and its normalized AUC; wall time; peak GPU
  memory; recovery events.

## 4. Dataset

- **Candidate:** fcc Cu portion of Zuo et al. 2019
  ("A Performance and Cost Assessment of Machine Learning Interatomic
  Potentials", arXiv:1906.08888; "mlearn" benchmark).
- Dataset ID: `cu_phase2`. Version/hashes: `TBD(WP1 qualification report)`.
- License: `TBD(WP1)`. DFT provenance (code, XC, cutoffs, k-points,
  smearing): `TBD(WP1)`.
- Pretraining-overlap risk vs the selected MACE checkpoint:
  `TBD(WP1 overlap assessment)` — recorded as a limitation either way.
- MPTrj is **not** an eligible independent test source for a MACE-MP
  checkpoint (pretraining overlap).

## 5. Partitioning (leakage-safe)

- Grouping rule priority: original trajectory/generation batch → structural
  prototype/defect class → strain family → fingerprint cluster → raw-file
  provenance group. Exact rule for the qualified dataset:
  `TBD(WP2, from WP1 grouping metadata)`.
- Targets: initial labeled set D₀ = 32; acquisition pool 128–512; validation
  ≥10%; frozen test ≥20%; counts scale down proportionally if the qualified
  Cu subset is small, with the actual numbers frozen here before H2.
- No source group crosses a protected boundary. Test labels are reachable
  only through the evaluation interface, which returns metrics, never
  ranked examples.
- Split seed: 20260711. Split manifest SHA-256 recorded at freeze.

## 6. Model

- Family: MACE (mace-torch). Versions: `TBD(WP3 — mace-torch, torch, e3nn
  pins verified by install)`.
- Checkpoint: `TBD(WP3 selection artifact — MACE-MP-0 small vs current
  compatible materials checkpoint)`, pinned by file SHA-256 + license.
- Default dtype: float64 for evaluation, float32 permitted for training if
  recorded. Device: cuda on Colab, cpu for fixtures.
- E0 policy: explicit atomic reference energies with source recorded;
  silent average-fallback forbidden in evidence runs; pre-training
  energy-offset diagnostic must pass or escalate.

## 7. Fine-tuning protocol (Protocol A — naive)

- Max epochs: 50 with early stopping (patience 10 on validation force MAE).
- Gradient clipping: 10.0. Optimizer/LR: `TBD(WP4 — from mace-torch
  defaults, recorded exactly)`. Seeds: base 42; arm-specific offsets
  recorded in the experiment matrix.
- Checkpoint every epoch; resume supported; NaN/OOM classified failures with
  one bounded repair each (plan §13.2).
- Protocol B (multihead replay) is a Phase 2b robustness arm, not part of
  this pilot.

## 8. Active learning

- Arms: zero-shot; static fine-tune (D₀ only); random acquisition; hybrid
  (3-seed ensemble force disagreement → high-uncertainty window →
  farthest-point diversity).
- Added-label budget B: 32 (identical across random and hybrid arms),
  in ≤3 rounds; acquisition batch size 8–16 as frozen in the experiment
  matrix.
- The acquisition controller never receives pool labels before reveal, test
  IDs, test labels, or any artifact derived from hidden labels.
- Every selection carries a machine-readable reason record.

## 9. Stopping rules

Stop when any holds: max rounds reached; label budget exhausted; primary
metric improves < 2% (relative) for two consecutive rounds; no valid
candidates; training unstable after bounded repair; Colab runtime near
exhaustion (finish atomic stage, checkpoint, stop partial); human gate
rejects continuation.

## 10. Compute

- Workload class `bounded_research_pilot` on Colab: 1 GPU, L4 first, one
  A100-40GB retry only on verified OOM, 60-minute bound, checkpoints at every
  stage and round, artifact pull-back required.
- Scientific status of all outputs: `pilot_only`. Publication-grade claims
  require ≥3-seed VESSL replication under
  `docs/research/vessl_replication_spec.md`.

## 11. Seed schedule

Base seed 42. Ensemble member seeds 42/43/44. Random-arm selection seed 1042.
Split seed 20260711. All recorded per-run in the evidence bundle.

## 12. Freeze record

- Preregistration SHA-256: recorded at H2 approval (not before).
- H2 approval: pending (`docs/PHASE2_OPEN_QUESTIONS.md` P2-Q2).
