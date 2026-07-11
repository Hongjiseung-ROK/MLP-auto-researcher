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
The first integration candidate is MACE-MP-0 small
`2023-12-10-mace-128-L0_energy_epoch-249.model`, 32,581,838 bytes, SHA-256
`2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736`.
It is explicitly `candidate_only`, not H2-selected. The SKILL passes this
verified local path to MACE and never asks the library to resolve or download
an alias.

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

**D-P2-14. Owner gate decisions recorded on 2026-07-11.** H1 promotes the
qualified mlearn Cu energy/force dataset at content SHA-256
`bc9b78ca5e3b95f91bf34bbc3641a3d6e3f92338b4e3d97065165157848cfc48`
and the whole-family split at file SHA-256
`80d9b95083ebba9d8f988a461525b326ba807c79da834b066d1917afc9d8a15e`.
Stress remains excluded because its kbar unit is inferred. H2 is deliberately
not frozen: compare one additional compatible checkpoint, run an E0/reference
diagnostic, demonstrate one optimizer step, and pin the Linux staging
environment first. Bounded Colab staging is authorized for one L4, at most
60 minutes, with one A100-40GB retry only after verified OOM; this is not H3
approval for a full pilot. Branch push plus a draft PR is authorized; merge is
not.

**D-P2-15. Cu is a benchmark scaffold, not the research program.**
`configs/research/program_guardrail.yaml` records the long-term program goal,
current benchmark role, non-goals, claim ceiling, next human gate, and exit
criteria. Reusable controller, evaluator, provenance, and compute interfaces
must not encode Cu-specific assumptions. The current ceiling is `pilot_only`;
the guardrail cannot authorize replicated or publication evidence.

**D-P2-16. Tea Time is the deterministic pause-and-review layer.** The
existing reflection SKILL is extended rather than duplicated. At recovery,
post-checkpoint/E0 review, preregistration freeze, Colab prelaunch, and staged
failure boundaries it emits: stage purpose, benchmark-lock-in audit, two or
three bounded alternatives, and an owner-question draft. It cannot accept a
data-artifact path, cannot add claims, and its `reflection` artifacts remain
forbidden claim evidence.

**D-P2-17. The sole second checkpoint candidate is content-pinned MACE-MPA-0
medium, without selecting a winner.** The official 79,462,305-byte asset has
SHA-256 `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`,
MIT license, 89-element MPTrj+sAlex/PBE+U training coverage, and passes the
same local CPU fixture as MACE-MP-0 small. A D0-only raw energy-offset
diagnostic was run for both candidates with no fitted shift and no access to
protected evaluation labels. Because H2 contains no frozen diagnostic
threshold, the memo records evidence and limitations only; both checkpoints
remain `candidate_only` until owner selection after optimizer/staging gates.

**D-P2-18. Colab execution is CLI-driven; notebooks are retired (owner
directive, 2026-07-12).** The research agent drives every Colab run through
the security-reviewed `google-colab-cli` v0.6.0 from the local machine:
`scripts/colab/colab_cli_run.sh` ships the exact commit as a git bundle
(no GitHub dependency), installs `scripts/colab/staging_requirements.txt`
pins, executes the repository script, pulls the artifact bundle back, and
verifies its SHA-256 manifest with `scripts/colab/verify_pullback.py` before
anything is cited as evidence. The notebook writers
(`scripts/colab/make_notebook.py`, generated `notebooks/*.ipynb`) were
deleted; plan_phase_2.md §7.5 became "Launcher policy (colab CLI)". Remote
exceptions do not propagate through `colab exec` exit codes, so the driver
requires an explicit STEP_OK sentinel per remote step and reads the task's
exit code from a TASK_EXIT marker; the VM is always released via a shell
trap.

**D-P2-19. Bounded Colab staging executed and passed (2026-07-12).** Under
the P2-Q4 authorization, `scripts/colab/colab_cli_run.sh staging` ran commit
`467ad6a` on one policy-attested NVIDIA L4 (colab CLI, git-bundle commit
shipping). All five stages passed: GPU attestation (allow), pinned
MACE-MP-0-small checkpoint resolution (SHA-256 `2ddb079c…5736`),
real-inference CPU/GPU float64 parity (max energy delta 0.0 eV, max force
component delta 1.14e-15 eV/Å against declared 5e-5 tolerances), one
controlled optimizer step (26 parameter tensors changed) with checkpoint
round-trip resuming at the epoch boundary, and a 13-file SHA-256 artifact
manifest whose pull-back verified locally. Boundary fixtures are synthetic
bulk-Cu structures — protected Cu partitions were never touched. Linux pins
recorded: torch 2.11.0+cu128, mace-torch 0.3.16, e3nn 0.4.4, numpy 2.0.2,
Python 3.12. Two earlier attempts surfaced and fixed real defects
(attestation enum access; CUDA resume required CPU-deserialized RNG states);
a third was lost to a transient CLI transport timeout, after which the
driver gained one bounded retry. Staging consumes staging-only status: it
does not satisfy H3.

**D-P2-MERGE-RECONCILIATION. PR #2 merge publishes infrastructure, not
scientific approval (2026-07-12).** PR #2 was merged into main through an
owner-controlled GitHub action (merge commit `cb5b927`). This publishes
infrastructure implementation and staging evidence. It does not grant H2
preregistration approval, H3 pilot authorization, or H5 claim-release
approval. Current capability wording is "Level 4 — bounded Colab staging
verified"; it must not be described as full pilot complete, scientific
fine-tuning complete, Auto Research loop complete, or publication eligible.
The successful staging run targeted commit `467ad6a`; any remote run at a
later commit is a new request requiring new explicit owner authorization
(the P2-Q4 staging authorization is consumed).
