# Open Questions

Material uncertainties requiring the project owner's decision. Work that is
independent and reversible (schemas, mocks, tests) proceeded; everything
listed here **blocks the corresponding real integration**, not the skeleton.

Status (updated 2026-07-10, owner answered via bootstrap Q&A):
**resolved** — OQ-1 (fcc Cu perturbed bulk), OQ-2 (materials-first),
OQ-3 (MACE first), OQ-4 (MACE mandatory baseline), OQ-5 (ORCA — see caveat
below), OQ-7 (OpenAI provider), OQ-8 (OpenHands as architectural reference).
**still open** — OQ-6 (GPU type/budget), OQ-9 (DVC/MLflow config),
OQ-10 (dataset credentials/storage), OQ-11 (Linux-only release),
OQ-12 (approval thresholds).

> **OQ-5 caveat (needs follow-up):** ORCA has no periodic boundary conditions,
> while the chosen benchmark (OQ-1, bulk fcc Cu) is periodic. Either labeling
> uses embedded/finite-cluster models cut from the bulk (method to be designed
> and validated), or a periodic code (Quantum ESPRESSO) is added later for the
> production labels. Raised as OQ-13.

**OQ-13. Periodic-labeling strategy given ORCA + bulk Cu.**
- Recommended default: prototype the ORCA integration on finite Cu clusters
  (exercises the real parser/failure triage either way), and decide
  cluster-embedding vs adding QE before the first production AL campaign.

## Group E — Remote compute activation (added 2026-07-11)

GPU families L4 / A100-40GB / A100-80GB are **already authorized** by the
owner (configs/compute_policy.yaml); these questions cover only budgets,
accounts, and fallback behavior needed before real remote execution.

> **Resolved 2026-07-11 (owner Q&A):** OQ-14 — Colab **Pro/Pro+** (session
> bound relaxed to 60 min for preflights, unit accounting in the audit log);
> OQ-15 — VESSL org/project to be provided later; until then mock-only, and
> real jobs cap at **4 GPU-hours** each; OQ-16 — pull small
> manifests/logs/metrics back to the local run directory (gitignored),
> checkpoints stay provider-side pending OQ-9; OQ-17 — **A100-40GB is an
> acceptable automatic substitute** when 80GB is unavailable; OQ-18 — a
> failed L4 preflight **auto-retries once on A100-40GB only for
> resource-class failures (OOM)**, anything else escalates.
> Remaining prerequisite for real VESSL runs: the org/project name.

**OQ-14. Colab account tier and budget.**
- Which tier (Free / Pay As You Go / Pro / Pro+), and what is the maximum
  session duration / compute-unit consumption per preflight?
- Recommended default: treat as Pay-As-You-Go, cap preflights at 30 min.

**OQ-15. VESSL organization, project, and per-job limits.**
- Which org/project/workspace receives production jobs, and what maximum
  cost or runtime per job? Recommended default: cap 4 GPU-hours/job until
  raised explicitly (matches OQ-12 draft).

**OQ-16. Remote artifact persistence.**
- Where should Colab/VESSL artifacts persist (Drive mount, object storage,
  pull-to-local)? Recommended default: pull small manifests/logs to the run
  directory; large checkpoints stay provider-side until OQ-9 is settled.

**OQ-17. A100 40GB acceptability.**
- When A100 80GB is unavailable, may jobs run on A100 40GB automatically?
  Recommended default: yes (both are allowlisted; preference order already
  encodes 80GB-first).

**OQ-18. Failed L4 preflight fallback.**
- Should a failed L4 preflight retry on A100 automatically or require
  approval? Recommended default: automatic single retry on A100-40GB only
  when the failure class is resource-related (OOM), otherwise escalate.

---

## Group A — Scientific target and scope

**OQ-1. First scientific benchmark system.**
- Recommended default: elemental fcc **Cu** (or another simple fcc metal)
  perturbed-bulk + vacancy configurations, matching the mock campaign shape.
  Consequence: cheapest possible validation of the AL loop; limited novelty.
- Alternatives: (a) a binary alloy family (more interesting, more DFT cost);
  (b) an oxide with charged defects (needs careful DFT settings, higher risk);
  (c) an OC20-style adsorbate/surface system (largest ecosystem pull, heaviest).

**OQ-2. Materials-first vs molecular-first.**
- Recommended default: **materials-first**, per plan.md (ecosystem maturity).
  Consequence: molecular/reactive branch (MACE-OFF, NequIP/Allegro) deferred to
  phase 2.
- Alternative: molecular-first would fit the host's existing ORCA licence and
  chemsmart tooling better, but contradicts plan.md's evidence-based choice.

## Group B — Models and baselines

**OQ-3. First supported MLIP backend.**
- Recommended default: **MACE (mace-torch)**, plan.md's own v1 default.
  Consequence: `mace` extra becomes the first real integration; CPU smoke
  tests possible on this Mac, real fine-tuning needs a Linux GPU node.
- Alternatives: CHGNet (lighter, materials-only), ORB (fast screening first).

**OQ-4. Is MACE the mandatory baseline for every experiment row?**
- Recommended default: **yes** — the experiment matrix in plan.md anchors on
  MACE; other backends are comparison arms. Consequence: one blessed
  fine-tuning path to stabilize before committee mode.

## Group C — Labeling and infrastructure

**OQ-5. DFT backend and license.**
- Detected: **ORCA 6.1.1 is installed on this host**
  (`~/programs/orca_6_1_1/orca`), so an academic ORCA licence appears to exist.
- Recommended default: **keep the mock backend through milestone 1** (done),
  then integrate **ORCA** as the first real labeler for molecular/cluster
  systems; if the materials-first scope holds (OQ-2), periodic labeling needs
  **Quantum ESPRESSO** (free, CI-compatible) or **VASP** (needs licence).
- Consequence: ORCA-only limits periodic-solid labeling; QE adds a second
  backend integration but is the only DFT engine CI could ever run.

**OQ-6. Vessel Cloud GPU type and budget.**
- Recommended default: per plan.md — A100 for smoke tests/medium fine-tunes,
  H100 for main campaigns, B200 only for memory-heavy committee/distillation.
  Need: confirmation of account, budget envelope (e.g. cap per campaign), and
  who approves jobs above the cap.

**OQ-7. LLM provider and credentials.**
- Recommended default: **Anthropic API (Claude)** for planner/failure-triage,
  since the planner seam is provider-agnostic and this environment has Claude
  tooling; plan.md's GPT-5.6 Sol guidance is treated as design guidance, not a
  provider mandate. Consequence: `agent` extra gets `anthropic`; a
  `litellm`-style router is added only if multi-provider is required.
- Note: v0 runs with **no LLM at all** (deterministic planner stub).

**OQ-8. OpenHands: hard runtime dependency or architectural reference?**
- Recommended default: **architectural reference only for milestone 1**
  (already implemented that way: event sourcing, typed tools, sandboxing ideas
  are first-party). Adopt the OpenHands SDK as the execution substrate when
  the first LLM-driven agent loop lands, behind the `agent` extra.
- Consequence: deferring avoids a heavy dependency before it is exercised;
  adopting early would lock in their event/tool schema now.

**OQ-9. DVC remote and MLflow tracking configuration.**
- Recommended default: **defer both**; local-file provenance is the source of
  truth (implemented). When approved: MLflow local server first, DVC remote on
  whatever object store the team already has.

**OQ-10. Dataset access, storage limits, licensing.**
- Recommended default: Materials Project API (free key, but per-user) and
  MPtrj snapshot only when the first real training lands; record dataset
  licence in `DEPENDENCIES.md` at that time. Need: who owns the MP API key,
  and where large artifacts may live (laptop is not the place).

## Group D — Release and governance

**OQ-11. Linux-only first release?**
- Recommended default: **develop on macOS (CPU mocks), support Linux x86_64 as
  the only tier-1 platform for real backends**. Consequence: no macOS wheels
  battles for torch/MLIP extras; CI targets Linux.

**OQ-12. Human-approval boundaries.**
- Recommended default: human approval required for (a) any real DFT batch
  above N=50 labels or any periodic DFT job, (b) any GPU job above 4
  GPU-hours, (c) any scope pivot (PIVOT decision), (d) any externally shared
  report. ESCALATE decisions always stop the run (implemented). Need: your
  numbers for (a) and (b).
