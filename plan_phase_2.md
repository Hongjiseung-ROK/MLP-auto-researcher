# Phase 2 Plan — Real MLIP Research on Google Colab

**Repository:** `Hongjiseung-ROK/MLP-auto-researcher`  
**Document:** `plan_phase_2.md`  
**Status:** Proposed implementation contract  
**Owner:** Project Executive / Human Principal Investigator  
**Scope:** Real-data, real-model, bounded active-learning research pilot on Colab  
**Primary production target after Phase 2:** VESSL  
**Last revised:** 2026-07-11

---

## 0. Executive decision

Phase 1 proved that the repository can execute a deterministic research workflow, preserve artifacts, recover from a bounded failure, resume from checkpoints, and reject unsupported claims. Phase 2 must prove something materially harder:

> The repository can ingest a real atomistic dataset, run a real MACE foundation model, fine-tune it on GPU, execute a leakage-safe active-learning comparison, and produce a reproducible evidence bundle from a bounded Google Colab session.

Phase 2 is **not** an attempt to build the final universal autonomous scientist. It is a controlled transition from a mock research operating system to a functioning scientific instrument.

The canonical Phase 2 experiment is:

> **Under an equal labeling budget, does uncertainty-plus-diversity acquisition improve the data efficiency of MACE fine-tuning for perturbed and defective fcc Cu configurations compared with random acquisition?**

The first execution uses a **simulated oracle**: all candidate configurations already have trusted DFT labels, but labels in the acquisition pool are hidden until selected. This gives the repository a real active-learning loop without making a fragile, expensive periodic DFT service a prerequisite for the first Colab study.

The scientific hierarchy is fixed:

1. **Real data before autonomous reasoning.**
2. **Real MACE inference before fine-tuning.**
3. **A static fine-tuning baseline before active learning.**
4. **Equal-budget random acquisition before claiming active-learning gains.**
5. **A single-seed Colab pilot before multi-seed production replication.**
6. **Verification before prose.**
7. **VESSL replication before publication-grade claims.**

A Colab result is tagged `pilot_only` until replicated under the production policy. Colab is allowed to prove that the scientific pipeline works; it is not allowed to turn one convenient run into a paper claim.

---

## 1. Why Phase 2 exists

### 1.1 What the repository already has

The current repository has a strong control plane:

- typed campaign and workflow schemas;
- a declarative DAG compiler;
- an event-sourced executor;
- checkpoint and resume support;
- bounded `RETRY`, `REFINE`, `PIVOT`, `ESCALATE`, and `ABORT` decisions;
- an artifact registry with content hashes;
- a scientific claim registry and verification gate;
- provenance and environment reports;
- provider-neutral local, Colab, and VESSL interfaces;
- GPU attestation and a fail-closed accelerator policy;
- a pinned and reviewed Colab execution path;
- unit, lint, typing, secret, and large-file gates.

Phase 1 therefore answered:

> Can the harness execute a controlled research-shaped workflow?

It did **not** answer:

> Can the harness perform useful atomistic machine-learning research?

### 1.2 The production gap

At the start of Phase 2, the following remain mock or absent:

- public scientific dataset acquisition and qualification;
- dataset licensing, units, level-of-theory, and lineage checks;
- real MACE checkpoint download and hash pinning;
- real MACE energy and force inference;
- real MACE fine-tuning;
- external holdout evaluation;
- uncertainty calibration;
- multi-round active learning;
- equal-budget random and uncertainty baselines;
- periodic DFT execution;
- downstream physical validation;
- a real Colab research artifact bundle;
- a VESSL replication.

Phase 2 closes the first six gaps and implements a bounded version of the next three. Periodic DFT and full VESSL replication are prepared but remain the next production milestone.

---

## 2. Phase 2 objective and non-goals

### 2.1 Objective

Deliver one complete, reproducible, real-data research pilot with the following path:

```text
qualified Cu DFT dataset
→ leakage-safe partition
→ pinned MACE zero-shot inference
→ initial-set fine-tuning
→ candidate-pool scoring
→ uncertainty + diversity acquisition
→ simulated label reveal
→ retraining
→ equal-budget baseline comparison
→ independent evaluation
→ verified pilot report
```

### 2.2 Phase 2 definition of “real research”

A run qualifies as real research only when all of the following are true:

- the structures and labels come from a documented external DFT source;
- the dataset version and raw hashes are recorded;
- the train, acquisition-pool, validation, and test partitions are physically separated by a declared grouping rule;
- a real pretrained MACE model generates predictions;
- at least one real gradient-based MACE fine-tuning run completes;
- the acquisition policy operates on model-derived information;
- labels are revealed only after candidate selection;
- a random baseline receives the same initial data, label budget, model family, and training budget;
- primary metrics are calculated on a frozen test set;
- every reported number is linked to a registered artifact;
- the full environment, commit, model, data, seeds, and recovery history are preserved.

### 2.3 Non-goals

Phase 2 will not:

- claim autonomous materials discovery;
- generate new periodic DFT labels inside Colab;
- support every MACE foundation checkpoint;
- support every external computational-chemistry SKILL;
- run an unrestricted long training campaign;
- permit an LLM to change the test split, primary metric, label budget, or scientific method after results are visible;
- treat one Colab run as publication-grade evidence;
- replace the existing compute router with DPDispatcher or another scheduler;
- build manuscript-generation agents before scientific verification is reliable.

---

## 3. Canonical scientific study

## 3.1 Research question

> Under a fixed target-label budget, does uncertainty-plus-diversity acquisition reduce force error on held-out fcc Cu configurations more efficiently than random acquisition when adapting a pinned MACE foundation model?

## 3.2 Primary hypothesis

Let:

- \(D_0\) be the fixed initial labeled set;
- \(P\) be the hidden-label acquisition pool;
- \(T\) be a frozen external test set;
- \(B\) be the total number of additional labels revealed;
- \(M_{\text{random}}\) be MACE fine-tuned with random acquisition;
- \(M_{\text{hybrid}}\) be MACE fine-tuned with uncertainty-plus-diversity acquisition.

The primary hypothesis is:

> At equal \(D_0\), \(B\), model checkpoint, optimizer budget, and seed schedule, the hybrid policy produces lower force MAE on \(T\) and a better force-error learning curve than random acquisition.

## 3.3 Null hypothesis

> Under the Phase 2 data and compute regime, hybrid acquisition offers no reproducible benefit over random acquisition.

A null result is acceptable. Phase 2 succeeds as an engineering and scientific milestone if the comparison is fair, reproducible, and honestly reported.

## 3.4 Primary and secondary endpoints

**Primary endpoint**

- force MAE on the frozen test set, in eV/Å.

**Secondary endpoints**

- energy MAE per atom, in eV/atom;
- force RMSE;
- 95th-percentile per-atom force error;
- fraction of structures exceeding a declared force-error threshold;
- Spearman correlation between ensemble disagreement and observed force error;
- error-versus-label-budget learning curve;
- area under the normalized learning curve;
- training wall time and peak GPU memory;
- recovery events and interrupted-run continuity;
- optional short ASE relaxation or MD sanity checks.

The primary endpoint cannot be changed after the preregistration artifact is signed without a recorded `PIVOT` and human approval. A changed endpoint starts a new campaign lineage.

---

## 4. Dataset strategy

## 4.1 Dataset decision

The first candidate is the public fcc Cu portion of the dataset associated with:

> Zuo et al., “A Performance and Cost Assessment of Machine Learning Interatomic Potentials,” 2019.

That study includes DFT energies and forces for several elemental materials, including fcc Cu, and evaluates properties such as elastic constants and phonons. It is a suitable independent benchmark candidate because it predates current MACE foundation models and was designed for cross-model MLIP assessment.

The candidate is **not automatically approved**. Before use, the data lead must verify:

- accessible canonical download source;
- license and redistribution terms;
- exact file version;
- structure count;
- label completeness;
- units;
- energy convention;
- force sign convention;
- periodic cells;
- stress or virial availability;
- DFT code and settings;
- duplicate and near-duplicate rate;
- whether the same structures are known to appear in the selected foundation model’s training corpus.

If this candidate fails qualification, Phase 2 may select another public Cu DFT dataset through the same process. It must not silently fall back to MPTrj.

## 4.2 Why MPTrj is not the main Phase 2 benchmark

MACE-MP models were trained on MPTrj. Using a Cu subset of MPTrj as both the target study and evaluation source creates a material risk of pretraining leakage and inflated zero-shot performance.

MPTrj may be used for:

- data-loader development;
- schema tests;
- a non-scientific installation smoke test;
- multihead replay when the selected MACE protocol explicitly calls for compatible replay data.

It must not be treated as an independent test of a MACE-MP checkpoint unless the overlap question is resolved and documented.

## 4.3 Dataset qualification pipeline

Add a deterministic dataset pipeline:

```text
download
→ checksum
→ unpack in quarantine
→ parse
→ normalize units
→ validate cells/species/labels
→ detect duplicates
→ group configurations
→ produce immutable manifest
→ human review
→ promote to qualified registry
```

### Required artifacts

```text
data_registry/
  datasets/
    cu_phase2/
      dataset_card.md
      source.json
      raw_manifest.json
      normalized_manifest.json
      license.txt
      schema.json
      qualification_report.json
      split_manifest.json
```

Large data remain outside Git. Git tracks only manifests, cards, schemas, checksums, and tiny fixtures.

## 4.4 Canonical normalized format

Use extended XYZ as the first interchange format because ASE and MACE support it directly.

Each configuration must contain:

- atomic numbers or chemical symbols;
- Cartesian positions;
- periodic cell;
- periodic boundary flags;
- total energy;
- atomic forces;
- optional stress/virial;
- source identifier;
- source-group identifier;
- level-of-theory identifier;
- original record hash;
- normalized record hash.

No unit inference is allowed during training. Units must be explicit in the dataset card and normalization report.

## 4.5 Leakage-safe partitioning

Do not use a naive frame-level random split.

The splitter must group related configurations before partitioning. The grouping priority is:

1. original trajectory or generation batch;
2. structural prototype and defect class;
3. strain/deformation family;
4. configuration fingerprint cluster;
5. raw-file provenance group.

Target partitions for the first pilot:

- initial labeled set: 32 configurations;
- acquisition pool: 128–512 configurations;
- validation set: at least 10% of qualified records;
- frozen test set: at least 20% of qualified records;
- optional stress test set: high-energy or defect-heavy configurations excluded from normal fitting.

Exact counts may be reduced for a small dataset, but the proportions and grouping logic must be preregistered.

The test labels are inaccessible to the acquisition policy. Evaluation occurs through a separate interface that returns metrics, not test examples ranked by error.

---

## 5. Model strategy

## 5.1 Canonical model family

MACE is the mandatory Phase 2 backbone because the repository has already selected it as the first production MLIP integration.

The implementation must pin:

- `mace-torch` version;
- PyTorch version;
- CUDA compatibility class;
- e3nn and related transitive versions;
- exact foundation checkpoint identifier;
- model file SHA-256;
- model license;
- default dtype;
- device;
- atomic reference energies;
- training configuration.

MACE documentation warns that defaults can change between versions. Therefore, convenience aliases such as an unqualified `mace_mp()` are forbidden in evidence-producing runs.

## 5.2 Candidate checkpoint

The initial candidate is a small MACE materials foundation checkpoint that:

- supports Cu;
- fits within an L4 Colab session;
- has a documented level of theory;
- can be downloaded from a stable release;
- can be pinned by file hash;
- is compatible with the qualified target labels.

A model is not selected solely because it is newest. The target dataset’s level of theory and energy reference must be compatible enough to support meaningful adaptation.

The selection artifact must compare at least:

- MACE-MP-0 small;
- MACE-MPA-0 or another current materials checkpoint, when compatible;
- training from scratch as a later optional control, not a Phase 2 blocker.

## 5.3 Fine-tuning protocol

Phase 2 implements two protocols in order.

### Protocol A — naive fine-tuning

Use for the first bounded pilot because it is simpler and exposes the integration surface quickly.

Required controls:

- explicit atomic reference energy policy;
- low maximum epoch count;
- early stopping;
- gradient clipping;
- deterministic seed;
- checkpoint every epoch or fixed step interval;
- validation monitoring;
- finite-loss checks;
- exact command and config capture.

### Protocol B — multihead replay

Add after Protocol A is stable.

MACE documentation recommends multihead replay for Materials Project foundation models because replay helps reduce catastrophic forgetting. The 2026 comparative fine-tuning study likewise reports that naive fine-tuning can converge best for a narrow single-system task, while replay is more reliable for preserving out-of-distribution behavior.

For Phase 2:

- naive fine-tuning is the canonical Colab pilot;
- multihead replay is an ablation or Phase 2b robustness arm;
- LoRA is optional and only activated if memory or severe small-data overfitting justifies it;
- no protocol is selected by an LLM after seeing test results.

## 5.4 Atomic reference energy policy

Atomic reference energies are a first-class scientific parameter.

The model adapter must:

- record the source of E0 values;
- prohibit silent fallback to an average setting in production mode;
- calculate or inherit E0 values under a declared compatibility rule;
- run a pre-training energy-offset diagnostic;
- fail or request approval when the initial per-atom error indicates an incompatible reference.

The E0 decision is part of preregistration.

---

## 6. Active-learning design

## 6.1 Simulated oracle

The candidate pool contains trusted labels, but the acquisition controller receives only structures and model outputs.

The oracle interface:

```python
class LabelOracle(Protocol):
    def reveal(self, candidate_ids: list[str], campaign_id: str) -> LabelBatch: ...
```

The implementation must enforce:

- only selected candidate IDs can be revealed;
- each reveal is immutable and timestamped;
- the label budget cannot be exceeded;
- the acquisition policy cannot query test IDs;
- hidden labels are not serialized into acquisition inputs;
- revealed labels are appended to the training lineage, never overwritten.

## 6.2 Acquisition arms

The minimum experiment has four arms:

1. **Zero-shot foundation model**
   - no target fine-tuning;
   - evaluates initial transfer.

2. **Static fine-tune**
   - fine-tune only on \(D_0\);
   - no additional acquisition.

3. **Random acquisition**
   - equal label budget;
   - equal number of rounds;
   - equal model and optimizer budget.

4. **Hybrid acquisition**
   - ensemble uncertainty;
   - diversity-aware batch selection.

An uncertainty-only arm is recommended but not required for the first successful pilot.

## 6.3 Uncertainty estimator

Use a three-member seed ensemble as the first defensible UQ mechanism.

For each candidate structure, calculate:

- mean predicted energy;
- mean force prediction;
- maximum or RMS force disagreement;
- optional energy-per-atom disagreement;
- ensemble member validity flags.

The primary acquisition uncertainty is the structure-level aggregate of force disagreement.

The system must not call disagreement “true uncertainty.” It is a ranking signal whose usefulness must be tested against observed errors.

## 6.4 Diversity control

Pure uncertainty selection can repeatedly choose near-duplicates or pathological configurations.

The hybrid policy is:

```text
filter invalid or extreme candidates
→ rank by uncertainty
→ form a high-uncertainty candidate window
→ apply farthest-point or clustering-based diversity
→ select batch under budget
```

Preferred first implementation:

- descriptor: pinned MACE descriptor if stable and reproducible;
- fallback: SOAP descriptor through a separately pinned dependency;
- selection: farthest-point sampling;
- batch size: 8–16;
- maximum rounds: 3.

Every selected structure receives a reason record containing its uncertainty score, diversity distance, source group, and final rank.

## 6.5 Stopping rules

Stop the pilot when any condition is met:

- maximum acquisition rounds reached;
- label budget reached;
- primary metric improvement is below the preregistered tolerance for two consecutive rounds;
- no valid candidates remain;
- training becomes unstable after bounded repair;
- Colab runtime budget is near exhaustion;
- human gate rejects continuation.

The pilot may stop early and still be valid if the partial state is complete and honestly labeled.

---

## 7. Colab operating model

## 7.1 Colab’s role

Google Colab provides a real GPU environment for the Phase 2 pilot, but its resources and session duration are not guaranteed. Therefore, the pipeline must assume interruption, dynamic GPU assignment, and ephemeral storage.

The compute hierarchy remains:

```text
local CPU
→ Colab bounded pilot
→ VESSL production replication
```

The existing rule that prohibits a full Colab campaign remains valid. Phase 2 introduces a narrow workload class:

```text
bounded_research_pilot
```

This is not the same as `production_research`.

## 7.2 Policy amendment

Add to `configs/compute_policy.yaml`:

```yaml
workload_classes:
  bounded_research_pilot:
    provider: colab
    allowed: true
    scientific_status: pilot_only
    max_gpus: 1
    max_runtime_minutes: 60
    max_acquisition_rounds: 3
    checkpoint_required: true
    artifact_pullback_required: true
    publication_claims_allowed: false
    requires_human_approval: true
```

Allowed accelerators remain:

- NVIDIA L4;
- NVIDIA A100 40GB;
- NVIDIA A100 80GB.

Routing:

- request L4 first;
- if and only if an observed resource-class failure is OOM, retry once on A100 40GB;
- any non-OOM failure escalates;
- an unrecognized GPU fails closed;
- a requested-versus-observed GPU mismatch fails closed.

## 7.3 Runtime rules

Every Colab run must:

1. clone the exact Git commit;
2. use a detached checkout;
3. install from a pinned lock or tested constraints file;
4. attest the actual GPU;
5. verify the model checkpoint hash;
6. verify the dataset manifest hash;
7. create a run ID before training;
8. checkpoint after each scientific stage;
9. persist small artifacts after each acquisition round;
10. produce a final or partial run-status artifact;
11. avoid silently continuing after a session discontinuity.

## 7.4 Storage strategy

Do not depend on notebook state.

Use:

- local Colab disk for active computation;
- periodic compressed checkpoint bundles;
- explicit pull-back or approved remote persistence for:
  - configs;
  - event logs;
  - metric files;
  - manifests;
  - selected-candidate records;
  - small checkpoints;
  - failure reports.

Large model checkpoints may remain provider-side only when their URI and checksum are recorded. For the Phase 2 pilot, prefer a small enough checkpoint to retrieve.

Google Drive mounting is optional, not a default requirement. The production path should not depend on interactive Drive behavior.

## 7.5 Notebook policy

The notebook is a thin launcher, not the source of scientific logic.

Required files:

```text
scripts/colab/run_research.py
scripts/colab/bootstrap_research.sh
notebooks/colab_phase2_research.ipynb
configs/research/cu_mace_al_pilot.yaml
workflows/cu_mace_al_pilot.yaml
```

The notebook must:

- contain no hidden scientific parameters;
- call versioned repository code;
- avoid committed output cells;
- print the exact commit, run ID, dataset hash, model hash, and policy decision;
- export a final artifact bundle.

---

## 8. Production architecture changes

## 8.1 New package structure

```text
src/mlip_research_agent/
  data/
    registry.py
    manifests.py
    qualification.py
    split.py
    oracle.py
  skills/
    data/
      cu_benchmark/
    mlip/
      mace_inference/
      mace_finetune/
    active_learning/
      ensemble_uq/
      diversity_select/
      oracle_reveal/
      decision_gate/
    evaluation/
      mlip_metrics/
      calibration/
      learning_curve/
      physical_sanity/
  research/
    preregistration.py
    experiment_matrix.py
    pilot_status.py
```

## 8.2 Typed SKILL contracts

Every real SKILL must retain the repository’s stronger contract:

```text
SKILL.md
schema.py
implementation.py
validators.py
tests/
examples/
```

Each real SKILL additionally declares:

- scientific status: `experimental`, `validated`, or `claim_eligible`;
- external dependency versions;
- input-data level-of-theory requirements;
- unit contract;
- deterministic tolerance;
- GPU memory class;
- expected artifact types;
- side effects;
- human approval class;
- known failure signatures;
- provenance completeness test.

## 8.3 Skill maturity levels

Adopt four maturity levels:

| Level | Meaning | Allowed use |
|---|---|---|
| L0 | Knowledge only | Planning and documentation |
| L1 | Task preparation | Input generation and dry-run |
| L2 | Executable | Controlled scientific execution |
| L3 | Claim eligible | May support numerical claims after independent verification |

Phase 2 target levels:

- Cu dataset loader: L3 after qualification;
- MACE inference: L3 after parity tests;
- MACE fine-tuning: L2 in first pilot, L3 after replay and multi-seed validation;
- ensemble UQ: L2;
- hybrid acquisition: L2;
- simulated oracle: L3;
- QE input preparation: L1;
- QE execution: deferred;
- pilot report renderer: L3 for `pilot_only` claims.

---

## 9. Selective integration of computational-chemistry SKILLs

Use `jinzhezenggroup/computational-chemistry-agent-skills` as a pinned, read-only domain knowledge source, not as an unrestricted executable bundle.

Pin the reviewed upstream commit:

```text
93ea0c4c716ad116869fba2ade26cccfd5cd05fc
```

## 9.1 Phase 2 Core Pack

Import or reference:

- `ase`;
- `pymatgen-structure`;
- `dpdata-cli`;
- `dft-qe`;
- `phonopy`.

Use them as follows:

| Upstream SKILL | Phase 2 use | Local treatment |
|---|---|---|
| ASE | structure and calculator workflow knowledge | map to existing typed atomistics interfaces |
| pymatgen-structure | supercell, vacancy, symmetry operations | typed structure-transformation adapter |
| dpdata-cli | format-conversion knowledge | offline pinned adapter, no unpinned `uvx` in evidence runs |
| dft-qe | periodic input preparation | L1 only; no direct submission |
| phonopy | later physical validation | prepare adapter and fixtures; execution is optional Phase 2b |

## 9.2 Integration rules

- preserve upstream license and attribution;
- record upstream repository and commit;
- do not copy all 32 SKILLs into the active prompt;
- lazy-load only the selected skill;
- do not allow an upstream SKILL to bypass the local compute router;
- do not allow runtime internet package resolution in an evidence-producing stage;
- do not allow a knowledge document to declare a calculation successful;
- wrap any executable path with local schema, artifact, failure, and provenance rules.

## 9.3 Explicit exclusions

Do not enable in Phase 2:

- `dpdisp-submit` as a second scheduler;
- VASP execution without license and POTCAR policy;
- Gaussian execution;
- unrestricted ReaxFF workflows;
- molecular RDKit/Packmol/Uni-Mol branches;
- full DeePMD training as a competing primary stack.

DeepMD/DPA3 may become a Phase 3 comparison arm after MACE is stable.

---

## 10. Work packages

## WP0 — Freeze the Phase 2 protocol

**Deliverables**

- `docs/research/phase2_preregistration.md`;
- `configs/research/cu_mace_al_pilot.yaml`;
- machine-readable experiment matrix;
- human approval record.

**Must fix**

- dataset candidate;
- primary metric;
- foundation checkpoint;
- initial-set size;
- acquisition batch size;
- maximum rounds;
- seeds;
- fine-tuning epoch or step budget;
- recovery limits;
- Colab runtime limit.

**Exit criterion**

The preregistration hash is recorded before the first result-producing run.

---

## WP1 — Dataset registry and qualification

**Deliverables**

- downloader with retry and checksum;
- raw and normalized manifests;
- unit-normalization code;
- duplicate detection;
- dataset card;
- qualification report;
- tiny test fixture.

**Tests**

- hash mismatch aborts;
- malformed cell rejected;
- missing force rejected;
- non-finite labels rejected;
- unit conversion parity;
- duplicate IDs deterministic;
- raw data remain immutable.

**Exit criterion**

A human approves one qualified Cu dataset and its license.

---

## WP2 — Leakage-safe partition and simulated oracle

**Deliverables**

- group-aware splitter;
- frozen test split;
- hidden-label acquisition pool;
- oracle reveal interface;
- budget ledger;
- access-control tests.

**Tests**

- no source group crosses protected boundaries;
- no test ID enters training or acquisition;
- acquisition cannot read hidden labels;
- budget overrun rejected;
- repeated reveal is idempotent;
- split bytes reproduce from seed and manifest.

**Exit criterion**

An independent audit confirms zero protected-group overlap.

---

## WP3 — MACE inference adapter

**Deliverables**

- pinned MACE environment;
- checkpoint resolver;
- model hash verifier;
- ASE calculator adapter;
- batch inference;
- energy and force output schema;
- CPU fixture and Colab GPU test.

**Tests**

- model hash mismatch aborts;
- CPU/GPU predictions agree within tolerance;
- rotation, translation, and atom-order invariance/equivariance checks;
- finite outputs;
- explicit dtype and device;
- provenance completeness.

**Exit criterion**

A real pinned MACE checkpoint predicts energies and forces for the qualified fixture on Colab.

---

## WP4 — MACE fine-tuning adapter

**Deliverables**

- config generator;
- naive fine-tuning execution;
- checkpoint capture;
- metrics parser;
- early stopping;
- NaN/OOM handling;
- E0 diagnostic;
- model card.

**Tests**

- one optimizer step changes trainable parameters;
- frozen test data are never passed to training;
- deterministic tiny-run tolerance;
- resume from checkpoint;
- invalid E0 policy fails;
- OOM repair changes only allowed resource parameters;
- failed runs do not emit a successful model artifact.

**Exit criterion**

A small target dataset completes real GPU fine-tuning and improves training/validation behavior without touching the test set.

---

## WP5 — Evaluation suite

**Deliverables**

- energy MAE per atom;
- force MAE and RMSE;
- P95 force error;
- group-wise metrics;
- learning-curve aggregation;
- calibration diagnostics;
- claim-ready metric artifacts.

**Tests**

- hand-calculated metric fixtures;
- unit consistency;
- aggregation invariance;
- missing predictions fail;
- NaN metrics fail;
- claim registry links every reported metric.

**Exit criterion**

Zero-shot and fine-tuned baselines are evaluated on the frozen test split.

---

## WP6 — Ensemble UQ and hybrid acquisition

**Deliverables**

- three-seed ensemble runner;
- force-disagreement scorer;
- descriptor extraction;
- farthest-point selector;
- candidate-reason records;
- random policy using the same interface.

**Tests**

- selectors do not access labels;
- random selection is seed reproducible;
- duplicate selections rejected;
- diversity increases under a synthetic fixture;
- uncertainty ranking behaves on a controlled ensemble;
- all arms consume identical budgets.

**Exit criterion**

One acquisition round completes for random and hybrid policies with fully traceable selections.

---

## WP7 — Multi-round campaign and decision gate

**Deliverables**

- round state schema;
- loop controller;
- stopping logic;
- round checkpoint;
- partial-run report;
- branch-comparison artifact.

**Tests**

- maximum rounds enforced;
- no label budget overflow;
- interrupted run resumes at the correct boundary;
- preregistered method cannot change silently;
- low-improvement stop works;
- one arm’s state cannot contaminate another arm.

**Exit criterion**

The full bounded single-seed pilot completes on Colab.

---

## WP8 — Colab research runner

**Deliverables**

- exact-SHA bootstrap;
- dependency lock;
- GPU attestation;
- research launcher;
- round-level persistence;
- result bundle export;
- remote-test marker.

**Tests**

- CPU-only mode is clearly non-scientific;
- unauthorized GPU denied;
- L4 accepted;
- A100 40GB accepted;
- unknown device denied;
- dataset/model hash mismatch denied;
- session restart preserves lineage;
- partial completion cannot be marked complete.

**Exit criterion**

A Colab run produces an attested, downloadable evidence bundle.

---

## WP9 — Verification, trace grading, and release candidate

**Deliverables**

- Phase 2 claim policy;
- trace grader for tool and routing behavior;
- human review checklist;
- `phase2_pilot_report.md`;
- machine-readable release manifest;
- VESSL replication spec.

**Exit criterion**

Every pilot claim passes artifact verification and the entire run can be replayed from an exact commit.

---

## 11. Agent architecture

## 11.1 Deterministic core first

The Phase 2 scientific loop must run without an LLM.

Deterministic code owns:

- data access;
- partitioning;
- training commands;
- acquisition mathematics;
- budget accounting;
- evaluation;
- stopping rules;
- artifact hashing;
- claim verification.

An LLM may assist with:

- literature-backed protocol drafting;
- diagnosing an unfamiliar failure after deterministic classifiers fail;
- comparing conflicting evidence;
- proposing a human-reviewed pivot;
- rendering a report from registered claims.

## 11.2 Manager and bounded specialists

Use one manager agent. Specialists are called as tools for bounded tasks and do not own shared mutable campaign state.

Potential specialists:

- dataset due-diligence agent;
- MACE configuration critic;
- failure-triage agent;
- claim-review agent.

Do not create a “swarm” for the numerical inner loop.

OpenAI’s manager-style orchestration is appropriate here: the manager owns the final state and shared guardrails, while specialists return typed results for narrow subtasks.

## 11.3 GPT-5.6 model policy

When OpenAI models are enabled:

- use the strongest reasoning model only for protocol design, contradiction resolution, difficult failure triage, and final claim review;
- use a lower-cost model or deterministic code for extraction, formatting, schema conversion, and routine summaries;
- set reasoning effort explicitly;
- log model, version, settings, prompt hash, tool trace, and output hash;
- do not allow model-generated text to count as scientific evidence;
- apply tool-level guardrails to every side-effecting function;
- require human approval for method changes, remote GPU execution, and external publication.

## 11.4 Trace evaluation

Add an agent trace evaluation dataset containing:

- correct dataset selection behavior;
- rejection of a leaking split;
- refusal to alter a metric after results;
- correct OOM classification;
- escalation on a model/data level-of-theory mismatch;
- correct removal of an unsupported claim;
- correct routing of a QE request to preparation rather than false execution.

The trace grader scores:

- tool selection;
- argument correctness;
- policy compliance;
- unnecessary tool use;
- hidden-label access attempts;
- approval-boundary compliance;
- final claim grounding.

---

## 12. Human-in-the-loop gates

Human approval is a scientific control, not an inconvenience.

## Gate H1 — Dataset promotion

The PI approves:

- source;
- license;
- units;
- level of theory;
- data quality;
- leakage risk;
- qualified manifest hash.

## Gate H2 — Preregistration

The PI approves:

- research question;
- primary endpoint;
- baselines;
- budget;
- split;
- model checkpoint;
- seed schedule;
- stopping rules.

## Gate H3 — Remote execution

The PI approves:

- exact Git commit;
- Colab workload class;
- requested GPU;
- maximum runtime;
- expected artifact location.

## Gate H4 — Scientific pivot

Approval is required for:

- changing dataset;
- changing model family;
- changing foundation checkpoint;
- changing E0 policy;
- changing primary metric;
- changing split;
- excluding a failed seed;
- increasing label or compute budget;
- changing fine-tuning protocol after inspecting test results.

## Gate H5 — Claim release

The PI reviews:

- all primary and secondary metrics;
- failures and exclusions;
- comparison fairness;
- calibration evidence;
- limitations;
- whether the run is still `pilot_only`.

Human approvals must be serializable and resumable. They belong in the run state and provenance bundle.

---

## 13. Failure-recovery policy

## 13.1 Dataset failures

| Failure | Action |
|---|---|
| checksum mismatch | `ABORT` |
| license unavailable | `ESCALATE` |
| missing forces or cell | `ABORT` or select another qualified source |
| unknown units | `ESCALATE`; never infer |
| duplicate explosion | `REFINE` deduplication, then reapprove manifest |
| split leakage | `ABORT` campaign lineage and regenerate split |

## 13.2 Model failures

| Failure | First action | Maximum automatic repair |
|---|---|---|
| checkpoint hash mismatch | redownload trusted asset | one retry |
| package/version incompatibility | use pinned compatibility environment | no ad hoc upgrade |
| OOM | reduce batch size or accumulation schedule | one repair |
| L4 OOM after repair | request A100 40GB fallback | one provider retry |
| NaN loss | lower learning rate and enable/verify clipping | one repair |
| exploding gradients | restore last checkpoint and apply preregistered stabilizer | one repair |
| train improves, validation degrades | early stop | no automatic cherry-picking |
| incompatible E0 | stop and request scientific review | no automatic guess |

## 13.3 Colab failures

| Failure | Action |
|---|---|
| session disconnect | resume from last round checkpoint; record continuity break |
| ephemeral files lost | recover from persisted bundle; otherwise mark incomplete |
| unauthorized GPU | fail closed |
| no GPU assigned | stop; do not silently run the research pilot on CPU |
| runtime nearly exhausted | finish current atomic stage, checkpoint, stop partial |
| install drift | fail environment attestation |

## 13.4 Scientific failures

| Failure | Action |
|---|---|
| hybrid does not beat random | report null result |
| uncertainty does not correlate with error | reject UQ claim and propose a new campaign |
| selected structures are near-duplicates | refine diversity policy within preregistered bounds |
| one arm fails more often | include failure rate in comparison |
| test improvement appears only after tuning to test | invalidate campaign |
| unsupported numerical claim | remove claim or rerun required analysis |

No recovery action may erase the original failure artifact.

---

## 14. Reproducibility and evidence bundle

Every Phase 2 run emits:

```text
artifacts/research/<run-id>/
  preregistration.json
  approvals.jsonl
  environment.json
  attestation.json
  source_commit.json
  dataset_manifest.json
  split_manifest.json
  model_manifest.json
  workflow_snapshot.yaml
  config_snapshot.yaml
  events.jsonl
  budget.json
  rounds/
    round-000/
    round-001/
    round-002/
    round-003/
  arms/
    zero_shot/
    static_finetune/
    random/
    hybrid/
  metrics/
  selections/
  failures/
  claims.json
  manifest.json
  pilot_report.md
  completion_status.json
```

## 14.1 Claim classes

- `infrastructure_claim`: installation, attestation, or execution behavior;
- `pilot_scientific_claim`: supported by a completed Colab pilot;
- `replicated_scientific_claim`: supported by preregistered multi-seed VESSL replication;
- `publication_claim`: approved by the PI and independent verification.

Phase 2 may emit the first two classes. It cannot emit the latter two.

## 14.2 Evidence rules

A numerical claim requires:

- metric artifact ID;
- source predictions;
- test-split manifest;
- model artifact or model hash;
- exact evaluation code version;
- units;
- uncertainty or seed information;
- declared scientific status.

A plot requires:

- plot-generation script;
- source data artifact;
- plot config;
- image hash.

Agent prose cannot be cited as evidence.

---

## 15. Test and evaluation strategy

## 15.1 Pull-request CI

Run on CPU:

- schema tests;
- dataset-fixture parsing;
- split-leakage tests;
- oracle access-control tests;
- metric correctness;
- selector synthetic tests;
- checkpoint-state tests;
- provenance validation;
- policy tests;
- secret and file-size checks;
- lint and strict typing.

No pull request automatically consumes Colab or VESSL resources.

## 15.2 Manual Colab staging gate

A human-triggered remote test runs:

- exact commit;
- real GPU attestation;
- real MACE import;
- real checkpoint download and hash;
- one inference batch;
- one optimizer step;
- checkpoint round trip;
- tiny qualified-data run.

It is separate from the full pilot.

## 15.3 Colab pilot gate

The full pilot begins only after staging passes for the same:

- commit;
- lock hash;
- dataset manifest;
- model checkpoint;
- CUDA compatibility class;
- workflow schema.

## 15.4 Release regression suite

Maintain tiny golden fixtures for:

- model inference parity;
- metric outputs;
- split manifests;
- acquisition selections;
- claim verification;
- recovery transitions.

Tolerance-based golden data must state why exact bytes are not expected.

---

## 16. Experiment matrix

### Phase 2a — required

| Arm | Initial labels | Added labels | Policy | Fine-tuning | Required |
|---|---:|---:|---|---|---|
| Zero-shot | 0 | 0 | none | none | yes |
| Static | 32 | 0 | none | naive | yes |
| Random | 32 | 24–48 | random in 2–3 rounds | naive | yes |
| Hybrid | 32 | same as random | ensemble disagreement + diversity | naive | yes |

### Phase 2b — optional robustness

| Arm | Purpose |
|---|---|
| Uncertainty-only | isolate diversity contribution |
| Multihead replay | test robustness against catastrophic forgetting |
| LoRA | test small-data or memory-constrained adaptation |
| MACE checkpoint comparison | test sensitivity to foundation model choice |
| Short physical sanity | relaxation or short MD stability |
| Phonon preparation | validate downstream task-generation path |

### Production replication

Run at least three preregistered seeds per active arm on VESSL. The replication uses the same frozen test set and budget.

---

## 17. Production hardening requirements

Before calling the repository production-ready:

- every external dependency is pinned through a lock or image digest;
- every model asset is content-addressed;
- every dataset is content-addressed;
- the Colab runner is idempotent at stage boundaries;
- the VESSL adapter is real, not mock;
- staging and production configurations are separate;
- rate, cost, and runtime limits are enforced;
- interrupted jobs can resume without label duplication;
- remote artifacts have retention and retrieval policies;
- the test split cannot be accessed by planner or acquisition tools;
- trace evaluation detects policy regressions;
- the claim verifier rejects missing or stale artifacts;
- one full campaign replays on a clean environment;
- one independent reviewer reproduces the headline metric.

Phase 2 prepares these conditions but does not claim all are complete until the VESSL replication milestone passes.

---

## 18. Implementation sequence

Implement in this order:

```text
1. preregistration schema and Phase 2 config
2. dataset registry and qualification
3. leakage-safe splitter and simulated oracle
4. pinned MACE inference
5. independent evaluation
6. MACE fine-tuning
7. random acquisition
8. ensemble disagreement
9. diversity selector
10. multi-round controller
11. Colab research runner
12. real staging run
13. real single-seed pilot
14. verification and pilot report
15. VESSL replication specification
```

Do not begin with an LLM planner, QE automation, or manuscript writer.

---

## 19. Proposed repository changes

```text
plan_phase_2.md
docs/research/phase2_preregistration.md
docs/research/dataset_review_checklist.md
docs/research/human_gates.md
docs/research/colab_pilot_runbook.md
docs/research/vessl_replication_spec.md

configs/research/cu_mace_al_pilot.yaml
configs/research/cu_mace_al_staging.yaml
workflows/cu_mace_al_pilot.yaml

scripts/data/fetch_cu_benchmark.py
scripts/data/qualify_cu_benchmark.py
scripts/colab/bootstrap_research.sh
scripts/colab/run_research.py

notebooks/colab_phase2_research.ipynb

src/mlip_research_agent/data/
src/mlip_research_agent/research/
src/mlip_research_agent/skills/data/cu_benchmark/
src/mlip_research_agent/skills/mlip/mace_inference/
src/mlip_research_agent/skills/mlip/mace_finetune/
src/mlip_research_agent/skills/active_learning/ensemble_uq/
src/mlip_research_agent/skills/active_learning/diversity_select/
src/mlip_research_agent/skills/active_learning/oracle_reveal/
src/mlip_research_agent/skills/active_learning/decision_gate/
src/mlip_research_agent/skills/evaluation/mlip_metrics/
src/mlip_research_agent/skills/evaluation/calibration/
src/mlip_research_agent/skills/evaluation/learning_curve/
```

---

## 20. Phase 2 acceptance criteria

Phase 2 is complete only when all mandatory conditions pass.

### Scientific

- [ ] one public Cu DFT dataset is qualified;
- [ ] leakage-safe partitions are frozen;
- [ ] a pinned MACE model completes real inference;
- [ ] a real MACE fine-tune completes;
- [ ] zero-shot, static, random, and hybrid arms are evaluated;
- [ ] random and hybrid arms use equal budgets;
- [ ] the acquisition policy never sees hidden or test labels;
- [ ] the primary result is reported even when null or negative.

### Colab

- [ ] actual GPU is attested;
- [ ] the exact commit is used;
- [ ] dataset and model hashes match;
- [ ] the pilot respects the 60-minute bound or stops partial;
- [ ] checkpoints survive stage boundaries;
- [ ] an evidence bundle is retrieved;
- [ ] no unauthorized accelerator or silent CPU fallback occurs.

### Engineering

- [ ] all local gates pass;
- [ ] remote tests are opt-in;
- [ ] package, model, and data versions are pinned;
- [ ] artifact and provenance schemas validate;
- [ ] failure paths are tested;
- [ ] the run can resume without duplication;
- [ ] no secrets or large raw assets enter Git.

### Integrity

- [ ] preregistration predates results;
- [ ] all method changes have approvals;
- [ ] every numerical claim links to artifacts;
- [ ] limitations and failed attempts are preserved;
- [ ] report status is `pilot_only`;
- [ ] a clean rerun reproduces metrics within declared tolerance.

---

## 21. Critic, rebuttal, synthesis

## Round 1 — “The dataset may leak into MACE pretraining”

**Critic**

A foundation model can appear highly capable when the evaluation configurations, or close relatives, were present in pretraining. Using MPTrj-derived Cu data would make the experiment particularly weak.

**Rebuttal**

Use an independent benchmark candidate, document its chronology and provenance, fingerprint configurations, and treat unknown overlap as a limitation. Keep MPTrj outside the independent test role.

**Synthesis**

Phase 2 includes a mandatory dataset-overlap risk assessment. Zero-shot performance is contextual, not the main proof. The main comparison is equal-budget adaptation under a frozen, group-separated target dataset.

---

## Round 2 — “Colab cannot be production infrastructure”

**Critic**

Colab resources are dynamic, ephemeral, and not guaranteed. A production research system should not depend on it.

**Rebuttal**

The project already intends VESSL as the production provider. Colab is valuable as a low-friction, real-GPU validation environment and can execute a tightly bounded pilot if interruption and attestation are first-class.

**Synthesis**

Introduce `bounded_research_pilot`, explicitly prohibit publication claims, checkpoint every stage, and require VESSL replication. Colab proves scientific integration, not production durability.

---

## Round 3 — “Active learning may add complexity without beating curated fine-tuning”

**Critic**

Modern foundation MLIPs may perform well after simple fine-tuning. An elaborate acquisition loop may provide little gain for elemental Cu.

**Rebuttal**

That is an empirical question. A static fine-tune and random acquisition are mandatory baselines. A negative result is still useful if it identifies when active learning is unnecessary.

**Synthesis**

Active learning is conditional, not ideological. Phase 2 succeeds if it fairly determines whether the complexity pays off under the chosen budget.

---

## Round 4 — “One seed can produce a misleading result”

**Critic**

A single Colab run cannot establish a robust scientific conclusion.

**Rebuttal**

The single-seed run is an integration pilot. Its claims are explicitly tagged `pilot_only`.

**Synthesis**

Phase 2 ends with a real pilot and a frozen VESSL replication specification. Publication-grade claims require at least three seeds and independent replay.

---

## Round 5 — “Importing many computational-chemistry SKILLs will destabilize the repo”

**Critic**

The external SKILL collection spans many tools, licenses, package managers, and execution assumptions. Installing everything creates a new attack surface and dependency graph.

**Rebuttal**

The repository needs domain breadth, but it does not need every tool active at once.

**Synthesis**

Pin one upstream commit, lazy-load a five-SKILL Core Pack, and promote capabilities through local typed adapters. No upstream SKILL can bypass the compute router or claim verifier.

---

## 22. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| qualified Cu dataset cannot be redistributed | medium | high | store downloader and hashes; require user-side retrieval |
| foundation checkpoint/data level mismatch | medium | high | level-of-theory review and E0 diagnostic |
| MACE dependency conflict on Colab | medium | high | pinned lock, staging matrix, no ad hoc upgrades |
| L4 OOM | medium | medium | small checkpoint, low batch, accumulation, one A100 fallback |
| Colab session termination | medium | medium | stage checkpoints and partial completion state |
| hybrid acquisition selects duplicates | medium | medium | descriptor diversity and source-group constraints |
| uncertainty poorly calibrated | high | medium | report correlation and false-confidence cases; no trust claim |
| active learning does not beat random | medium | low | valid null result; static fine-tune remains useful |
| test leakage | low after controls | critical | protected oracle/evaluator boundary and audit tests |
| post-hoc method changes | medium | critical | preregistration and human `PIVOT` gate |
| external SKILL supply-chain drift | medium | high | pinned commit, local review, offline evidence runs |
| one-run overclaiming | medium | critical | `pilot_only` claim class and VESSL replication gate |

---

## 23. First execution contract

The first real Colab research run must be launched from a reviewed commit and must print a pre-run summary equivalent to:

```text
Campaign: cu-mace-al-phase2-pilot
Scientific status: pilot_only
Commit: <exact SHA>
Dataset: <qualified dataset ID and SHA>
Split: <split manifest SHA>
Model: <checkpoint ID and SHA>
MACE environment: <lock SHA>
GPU requested: NVIDIA L4
Runtime limit: 60 minutes
Initial labels: 32
Acquisition arms: random, hybrid
Added-label budget: identical across arms
Acquisition rounds: <= 3
Primary endpoint: force MAE on frozen test set
Human approvals: H1, H2, H3 present
```

The run must refuse to start when any value is missing.

---

## 24. Post-Phase 2 roadmap

### Phase 2.5 — production replication

- activate the real VESSL transport;
- run at least three seeds;
- reproduce the Colab pilot under a pinned container;
- compare Colab and VESSL inference parity;
- generate replicated claims;
- add a release model card.

### Phase 3 — real labeling

- promote `dft-qe` from L1 preparation to a typed L2 execution adapter;
- select and pin pseudopotentials;
- preregister cutoffs, k-points, smearing, and convergence;
- label a small acquisition batch with periodic DFT;
- compare simulated-oracle and online-labeling behavior;
- add phonon, vacancy, and short-MD physical validation.

### Phase 4 — broader research autonomy

- add multihead replay and LoRA decision templates;
- add a second MLIP family;
- add cross-run scientific memory;
- add literature-grounded hypothesis proposals;
- enable a verified writer only over registered claims;
- extend to alloys, surfaces, or molecular/reactive systems.

---

## 25. Final production principle

The Phase 2 repository should behave like a cautious computational researcher:

- it knows exactly where its data came from;
- it separates exploration from evaluation;
- it records every model and method choice;
- it does not inspect answers before choosing experiments;
- it treats uncertainty as a hypothesis to validate;
- it preserves failures;
- it pauses at consequential decisions;
- it distinguishes a pilot from replicated evidence;
- it writes only what its artifacts can support.

The milestone is not that an agent can launch MACE in a notebook.

The milestone is that a skeptical researcher can inspect the run, reproduce it, challenge it, and still understand exactly why the reported result should or should not be trusted.

---

## References and design sources

1. Repository architectural plan, `plan.md`, `Hongjiseung-ROK/MLP-auto-researcher`.
2. Repository implementation status, `docs/IMPLEMENTATION_STATUS.md`.
3. Zuo, Y. et al. *A Performance and Cost Assessment of Machine Learning Interatomic Potentials* (2019), arXiv:1906.08888.
4. Zhang, Y. et al. *DP-GEN: A Concurrent Learning Platform for the Generation of Reliable Deep Learning Based Potential Energy Models* (2019), arXiv:1910.12690.
5. MACE documentation, *Foundation Models*, version 0.3.13.
6. MACE documentation, *Fine-tuning Foundation Models*, version 0.3.13.
7. Tompa, T. L. et al. *Fine-tuning MLIP Foundation Models: Strategies for Accuracy and Transferability* (2026), arXiv:2606.12704.
8. Google Colab FAQ, resource and virtual-machine lifecycle guidance.
9. OpenAI Agents SDK documentation, *Agent Orchestration*.
10. OpenAI Agents SDK documentation, *Guardrails*.
11. OpenAI Agents SDK documentation, *Human-in-the-loop*.
12. Ding, M. et al. *Automating Computational Chemistry Workflows via OpenClaw and Domain-Specific Skills* (2026), J. Chem. Theory Comput., DOI: 10.1021/acs.jctc.6c00622.
13. `jinzhezenggroup/computational-chemistry-agent-skills`, pinned integration candidate commit `93ea0c4c716ad116869fba2ade26cccfd5cd05fc`.
