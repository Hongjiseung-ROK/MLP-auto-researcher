# Remote real-MACE Auto Research replay plan

Status: implementation plan synthesized before source-code changes.

Starting point: merged `main` at
`99a7a724ec3e774bee67c9eff64844f3bb590fe7`; work branch
`phase3/remote-mlip-loop`.

This plan governs one infrastructure-validation replay. It does not freeze H2,
grant H3, release H5, promote a checkpoint, reveal acquisition labels, inspect
the frozen test partition, run DFT, complete an active-learning pilot, or make a
scientific claim.

## Planning inputs

Four read-only roles reviewed the repository before implementation:

- `repo_mapper`: mapped WP6/WP7, the synthetic trace, MACE WP1-WP5, the
  trace grader, documentation, and Colab driver.
- `mlip_scientist`: reviewed checkpoint/E0 handling, units, real mutation
  semantics, parameter changes, save/reload, and evidence boundaries.
- `active_learning_scientist`: reviewed pause/review/resume, proposal lineage,
  mutation legality, Tea Time history, evaluation independence, and exactly-once
  execution.
- `scientific_auditor`: attempted to falsify label isolation, exact-commit
  delivery, retry safety, trace grading, pull-back, and cleanup.

All four identified the same blocking themes: the full labeled dataset must not
reach Colab; iteration 2 must not be generated before host reviews; transport
retries must not relaunch optimizer work; the real runner must implement every
advertised mutation; and the existing synthetic grader cannot certify the
remote trace.

## Resolved architecture

### 1. Preserve and harden the local loop

- Move `BenchmarkAdapter` and `AdapterRunResult` into
  `research/auto_research/adapters/base.py` and retain the synthetic fixture as
  `adapters/synthetic.py`. Compatibility imports may remain temporarily, but
  the controller may not import the synthetic module.
- Store repository-relative trace paths. Historical traces remain immutable.
- Derive Tea Time mutation history from recorded legality/mutation artifacts and
  measure a true trailing streak.
- Add a new current-commit happy-path trace and a separate recovery-path trace;
  both must pass the independent grader.

### 2. Create a least-privilege data package

- A host-side trusted projector creates a content-addressed label view containing
  exactly 32 `initial_labeled` and 40 `validation` configurations.
- The projector verifies every record against the committed normalized manifest,
  the exact split membership, and the immutable hashes:
  - dataset content: `bc9b78ca5e3b95f91bf34bbc3641a3d6e3f92338b4e3d97065165157848cfc48`
  - normalized manifest: `ac3655e41ce4327ba18e2a603866b97b00b839ff04f16bb2cb86a3d8ba8a70d3`
  - split manifest: `80d9b95083ebba9d8f988a461525b326ba807c79da834b066d1917afc9d8a15e`
- The remote package contains no acquisition-pool, frozen-test, stress-test, or
  unknown record and no stress labels. Remote code never opens the full labeled
  dataset. Reviews receive aggregate validation metrics only.

### 3. Add an explicit infrastructure-only MACE boundary

- Add `MACEPhase2Adapter` using the existing pinned checkpoint verification,
  WP4 controlled training, MACE inference, and independent WP5 metric
  definitions. The adapter emits evidence and never calls the acceptance policy.
- Add an `infrastructure_replay` mode: candidate-only checkpoint, no H2
  assertion, exactly one optimizer step per candidate, exact D0/validation
  counts, `scientific_status: infrastructure_only`, and
  `claim_eligible: false`.
- Verify checkpoint SHA-256
  `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736`
  before deserialization.
- Keep `e0_policy=foundation`; record the descriptive D0 diagnostic without a
  pass claim. Hash and compare the Cu reference-energy state before training,
  after training, and after reload.
- Wire `learning_rate`, `gradient_clip`, `batch_size`, `scheduler_patience`,
  `energy_loss_weight`, `force_loss_weight`, and `trainable_layer_policy` into
  the real runner. Fail closed on unknown topology, empty trainable sets,
  changed frozen tensors, non-finite values, or inert requested mutations.
- Save/reload verification performs no optimizer step.

Baseline and both candidate iterations start from the same immutable foundation
checkpoint, D0, validation partition, seed, and one-step budget. This resolves a
review conflict: "resume" means the same controller and Colab session continue;
it does not mean iteration 2 continues training from iteration 1, which would
confound the mutation with an extra optimizer step.

### 4. Add a typed review pause and proposal lineage

- Add `AWAITING_EXTERNAL_REVIEW` after iteration 1. No iteration-2 proposal or
  optimizer work may exist before the pause.
- Add strict `AgentReview` and `ReviewSynthesis` schemas. They are sealed
  `agent_text`, never evidence, and `claim_eligible: false`.
- The synthesis references the three review hashes plus iteration-1 proposal,
  evaluation, decision, and lesson hashes; it records accepted and rejected
  recommendations.
- A deterministic host-reviewed proposal provider maps the synthesized mutation
  class/direction to an exact legal in-bounds value. Proposal 2 references
  iteration 1 and the synthesis and contains a falsification condition.
- `stop` or `escalate` recommendations fail closed; agent consensus cannot
  override immutable constants, legality, Tea Time, evaluator decisions, or
  owner gates.
- WP6 selection/oracle skills are not executed in this replay. They remain
  locally verified components; this is configuration Auto Research, not active
  learning and it reveals zero pool labels.

### 5. Enforce exactly-once remote execution

- Each optimizer operation gets a sealed request fingerprint and append-only
  start/completion receipt binding commit, proposal, config, seed, data/split,
  checkpoint, E0 policy, and step budget.
- Re-entry may reuse a completed receipt or reconnect to a proven live operation.
  An ambiguous started operation escalates; it is never relaunched.
- Scientific snippets are excluded from the generic resend-on-timeout helper.
- The trace grader rejects duplicate operation IDs, duplicate starts, excess
  optimizer steps, changed resume inputs, or iteration 2 before review synthesis.

### 6. Add a strict host-orchestrated Colab task

- Add task `ralphthon_mace_replay` using a full 40-character SHA equal to clean
  `HEAD`, an exact git bundle, detached checkout, pinned dependencies, and one
  NVIDIA L4. Reject branch names, short SHAs, dirty tracked/untracked trees,
  CPU/T4/A100, more than one GPU, runtime over 60 minutes, protected data,
  checkpoint promotion, and claim-bearing status before allocation.
- The total deadline starts before session creation and covers bootstrap,
  install, both iterations, review pause, pull-back, and cleanup.
- One host process owns the live session across: bootstrap/attest; baseline and
  iteration 1; iteration-1 pull-back; host reviews and synthesis; upload of the
  sealed review bundle; exact resume and iteration 2; final pull-back.
- Pull-back uses a new staging directory, rejects extra/unmanifested files,
  verifies every SHA-256, and atomically publishes the final directory.
- Cleanup runs in `finally` for every failure and interruption. Stop failures are
  fatal to completion; provider state is polled until the named session is
  confirmed absent. Final host grading occurs after cleanup evidence is added.

### 7. Extend independent grading

Add a remote-infrastructure profile that verifies required artifacts, relative
paths, exact commit, L4/one-GPU/60-minute authorization, immutable hashes,
validation-only data, evaluator/adapter/controller identity separation, three
reviews and synthesis, review-before-proposal ordering, proposal-2 evidence
dependence, one optimizer receipt per iteration, save/reload without training,
`infrastructure_only`, `claim_eligible: false`, complete manifest coverage,
hash-verified pull-back, and confirmed inactive session.

## Recommendation disposition

Accepted:

- partition-scoped label views instead of shipping the full labeled dataset;
- explicit infrastructure replay mode rather than misusing `pilot`;
- same-foundation candidate restarts;
- real loss-weight and trainable-layer mutation wiring;
- E0 invariance evidence without claiming E0 compatibility passed;
- aggregate-only review packets;
- typed pause/review/synthesis contracts;
- exactly-once operation receipts and no scientific resend;
- L4-only task policy, total deadline, fresh pull-back, and confirmed cleanup;
- a remote-specific independent grader;
- keeping WP6 out of the executed replay.

Rejected or narrowed:

- continuing iteration 2 from iteration-1 optimizer state: rejected because it
  confounds mutation attribution and violates the mutable resume contract;
- treating ensemble disagreement as calibrated uncertainty: rejected; it is not
  exercised here and remains only a ranking signal;
- using `pilot_only`: rejected because H2/H3 remain open;
- A100 fallback: rejected by this task's authorization;
- reusing the generic Colab transport retry around scientific work: rejected;
- adding a second controller, evaluator, artifact registry, or Colab stack:
  rejected; existing interfaces are extended.

## Validation and commit gates

Before the one remote allocation:

1. focused unit and real-MACE CPU boundary tests pass;
2. current-commit happy and recovery traces pass;
3. Ruff, strict mypy, full pytest, tracked-file scan, staged-file scan, and
   `git diff --check` pass;
4. a fresh scientific-auditor pass has no blocking finding;
5. all source/config/tests/agent definitions are committed locally;
6. the launcher validates that exact clean commit and L4-only contract.

The remote authorization is consumed only when `colab new` succeeds. No second
session is allowed. A transport reconnect may target only the same proven live
session and may not repeat scientific execution.

## Capability wording

Before the remote trace passes: Level 4.5 means WP6/WP7 and the local synthetic
loop are verified while the real remote loop is the target. After a successful
trace, only these statements become true: local synthetic Auto Research
verified; local real-MACE Auto Research verified; remote real-MACE Auto Research
verified. Full active-learning pilot complete, VESSL replicated, and publication
eligible remain false.
