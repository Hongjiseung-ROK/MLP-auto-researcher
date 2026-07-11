# Phase 2 Design Decisions

Notable choices made while implementing `plan_phase_2.md`. Numbering continues
the D-P2 series independently of Phase 1's `docs/IMPLEMENTATION_DECISIONS.md`.
Each records what was decided, why, and what it preserves or migrates from
Phase 1 verified behavior.

**D-P2-1. Phase 2 runs on a dedicated branch, not `main`.**
Branch `phase2/real-colab-research`; small milestone commits; draft PR after
all mandatory local gates pass. Phase 1 `main` stays green throughout.

**D-P2-2. `api.env` is a sealed secret container.**
Covered by `.gitignore` (`api.env`, `*.env`), mode 600, never read by a
model-visible tool, never passed to a subagent, never uploaded to Colab. The
only consumer is `src/mlip_research_agent/secrets/api_env.py`, which exports
`MP_API_KEY` into the process environment and reports non-sensitive booleans
only (no value, prefix, suffix, length, or hash). Colab receives qualified
artifacts, never the key.

**D-P2-3. Labeled configurations get a new first-party schema; extxyz is the
MACE interchange format, not the provenance format.**
Phase 1 determinism relies on `structures_io.StructureSet` (byte-stable JSON).
Phase 2 adds `data/manifests.py` labeled-configuration records (energy,
forces, optional stress, source/group/level-of-theory ids, raw + normalized
hashes) with deterministic JSON as the registry format, and an exact,
deterministic extxyz *export* step for mace-torch consumption. The export
records the SHA-256 of the emitted extxyz so training inputs remain
content-addressed. Rationale: extxyz float formatting is library-version
dependent; the qualification registry must not be.

**D-P2-4. Reproducibility contract split: byte-identical for deterministic
CPU stages, tolerance-based for GPU learning stages.**
Phase 1's byte-identical gate continues to apply to campaign/mock DAGs,
manifests, splits, selections, and metrics *computation* (given identical
predictions). Real GPU training/inference artifacts are content-addressed and
seeded but reproduce within declared tolerances (plan §15.4). Each golden
tolerance fixture states why exact bytes are not expected.

**D-P2-5. Phase 2 research pipeline is orchestrated by a dedicated research
controller, reusing skills + registry + claims, without widening Phase 1's
`CampaignSpec` mock allowlists yet.**
`CampaignSpec.backend` stays `("mock",)` for the Phase 1 vertical slice.
The Phase 2 pilot is driven by `configs/research/cu_mace_al_pilot.yaml`
(a `ResearchPilotSpec`), compiled onto the same executor/skill machinery.
Rationale: keeps the verified Phase 1 end-to-end test untouched while the
real stack matures; merging the two entry points is a recorded follow-up.

**D-P2-6. Claim schema gains `claim_class` and `scientific_status` with
backward-compatible defaults.**
Existing Phase 1 claims validate unchanged (`infrastructure_claim` default).
Phase 2 may emit `infrastructure_claim` and `pilot_scientific_claim` only;
`replicated_scientific_claim` and `publication_claim` are defined but
unmintable in Phase 2 (enforced in code, not convention).

**D-P2-7. Dataset candidate: Zuo et al. 2019 (mlearn) Cu subset, subject to
real due diligence and the H1 gate.** MPTrj is explicitly not the independent
benchmark (MACE-MP pretraining overlap). MPTrj/Materials Project API use is
limited to reconnaissance, loader development, and non-claim fixtures unless
the owner approves a recorded PIVOT.

**D-P2-8. MACE checkpoint candidates are evaluated by due diligence, not
recency.** Comparison set: MACE-MP-0 small vs a current compatible materials
checkpoint. Selection requires: Cu support, L4 memory fit, documented level
of theory, stable download URL, SHA-256 pinning, license recorded, E0
compatibility with the qualified dataset. Unqualified `mace_mp()` convenience
aliases are forbidden in evidence-producing code paths.

**D-P2-9. `bounded_research_pilot` is a new workload class, distinct from
`production_research`.** Colab-allowed, max 1 GPU, 60 min, ≤3 acquisition
rounds, checkpoint + artifact pull-back required, publication claims
forbidden, human approval (H3) required. Policy stays fail-closed: L4 first;
one A100-40GB retry only on verified OOM-class failure; everything else
escalates.

**D-P2-10. The three-seed ensemble is a ranking signal, not uncertainty
truth.** Its correlation with observed error is a *reported measurement*
(Spearman, post-reveal), never an assumption. Diversity (farthest-point
sampling on descriptors) is applied after the uncertainty window to prevent
near-duplicate batches.
