# mace_finetune

Resumable, lineage-bound MACE fine-tuning boundary. In Phase 2 it runs only as
a `boundary_test`: at most two full-batch epochs/optimizer steps that prove the
training loop, checkpoint round-trip, and failure taxonomy on real MACE code
without crossing the H2 preregistration gate.

## Contract

- Every data input is a registered, hash-verified artifact: normalized dataset,
  dataset manifest, split manifest, train/validation subset manifests, and the
  oracle training-lineage manifest. Any hash, kind, or lineage mismatch fails
  closed before training starts.
- Test partitions are unreachable: the training subset must exactly equal the
  registered lineage head, stay inside `initial_labeled ∪ acquisition_pool`,
  and never intersect `validation`, `frozen_test`, or `stress_test`. The
  validation subset must equal the frozen validation partition.
- The base checkpoint resolves through the same pinned manifest as
  `mace_inference` (filename, size, SHA-256, `mace-torch==0.3.16`,
  `candidate_only` status) before any bytes are deserialized.
- Training is seeded (`python`/`numpy`/`torch`/loader generator), uses
  Adam/AdamW with `ReduceLROnPlateau` and gradient clipping, and trains only at
  full-epoch boundaries: `batch_size` must cover the training subset so resume
  happens at complete epochs.
- After every epoch a complete continuation state (model/optimizer/scheduler/
  RNG states, metric records) is checkpointed atomically. `resume_state_artifact`
  resumes only when the resume-contract SHA-256 (hyperparameters + data
  identity + base checkpoint) matches exactly.
- Non-finite labels or metrics abort. Failures are classified:
  CUDA OOM → `RESOURCE_EXHAUSTED`, retryable with halved `batch_size`
  (`repair_params`); NaN/non-finite metrics → `SIMULATION_INSTABILITY`,
  retryable with halved `learning_rate` and clipped `gradient_clip`; anything
  else escalates. A failed attempt registers a `mace_fine_tune_failure`
  artifact and never emits a model artifact.
- Success registers the fine-tuned model, controlled checkpoint, config,
  metrics, resume state, and a model manifest binding dataset content hash,
  split semantic hash, lineage artifact, E0 policy, and parameter-change
  evidence (`changed_parameter_tensors > 0` is mandatory).

## Maturity and limits

- Maturity: boundary test locally executable on CPU; `pilot` mode is disabled
  in code until H2 selects a checkpoint and freezes the preregistration.
- Scientific status: `training_boundary_only` — outputs are infrastructure
  evidence, never scientific claims.
- E0 policy: only `foundation` (keep the foundation model's atomic references)
  is implemented; other policies are rejected at the schema.
- Multihead/replay fine-tuning is out of scope for this SKILL version.
- Side effects: artifacts only below `ctx.step_dir` (staged in a transaction
  directory, committed with `os.replace`); the checkpoint cache is read-only.
- Opt-in real CPU fixture: set `MACE_TEST_CHECKPOINT` (and optionally
  `MACE_TEST_MANIFEST`) to run one real optimizer step plus a checkpoint
  round-trip against the pinned candidate checkpoint.
