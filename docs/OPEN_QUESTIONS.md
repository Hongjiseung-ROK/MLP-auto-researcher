# Open Questions

Material uncertainties requiring the project owner's decision. Work that is
independent and reversible (schemas, mocks, tests) proceeded; everything
listed here **blocks the corresponding real integration**, not the skeleton.

Status: **all open** (recorded 2026-07-10).

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
