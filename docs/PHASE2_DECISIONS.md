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

**D-P2-6. Claim schema gains `claim_class` and `scientific_evidence_tier` with
backward-compatible defaults.**
Existing Phase 1 claims validate unchanged (`infrastructure_claim` plus
`non_scientific` defaults). Artifact `VERIFIED` status remains integrity-only
and cannot upgrade scientific maturity. Numerical scientific claims require
metric artifacts and applicable dataset/split/model manifests; replicated
and publication classes require multiple runs, with H5 approval additionally
required for publication eligibility.
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

**D-P2-11. Local macOS dev env uses conda-forge pytorch 2.12.1, not the pip
torch wheel.** The pip wheel bundles its own libomp and aborts on import next
to conda-forge's OpenMP (`OMP: Error #15`). The unsafe
`KMP_DUPLICATE_LIB_OK=TRUE` workaround was rejected because it can silently
corrupt numerics. Consequence: local CPU fixtures run torch 2.12.1; the
Colab/Linux evidence environment pins its own torch in the staging
constraints file, and CPU/GPU parity is a declared-tolerance test, not an
assumption. `pip check` confirms mace-torch 0.3.16 accepts the conda torch.

**D-P2-10. The three-seed ensemble is a ranking signal, not uncertainty
truth.** Its correlation with observed error is a *reported measurement*
(Spearman, post-reveal), never an assumption. Diversity (farthest-point
sampling on descriptors) is applied after the uncertainty window to prevent
near-duplicate batches.

**D-P2-12. WP2 protects whole physical/source families before finer split
units.** Bulk AIMD source numbering is zero-based, so the WP1 time-block
formula is `snapshot // 10`; vacancy numbering is one-based and remains
`(snapshot - 1) // 5`. This correction removes three invalid `block-1`
lineage units. For actual partitioning, `group_id` has priority: complete
temperature-specific AIMD/vacancy trajectories, elastic strain modes, and
individual slabs never cross boundaries. The splitter derives its tie-break
seed from the qualified-manifest hash, canonical spec, declared seed,
grouping field, and algorithm version. Group indivisibility changes requested
32/172/30/59/0 targets to actual 32/161/40/60/0 counts; both are recorded.
Record-to-group membership is revalidated against the qualified manifest, so
a self-consistent but forged test-to-pool reassignment fails.

**D-P2-13. Oracle reveals are immutable atomic transactions, not mutable
budget counters.** Selectors receive a label-free `AcquisitionView`; only the
oracle receives the normalized labels. Each `(campaign_id, round_id)` has one
canonical request, and request/decision/batch/ledger/lineage/state artifacts
are committed by a single atomic directory rename. A resumed identical
request reuses the transaction without charging again; a conflict, protected
ID, unknown ID, or budget excess fails before any label-bearing state commit.
Separate campaign IDs isolate random and hybrid arm budgets. Within one
campaign, every new reveal must name the unique registered prior-state head.
State commits use a file lock plus compare-and-swap hash, so stale writers
cannot overwrite a newer budget ledger.
