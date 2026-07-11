# decision_gate

The acquisition decision gate composing the WP6 pipeline:

```
invalid filtering → uncertainty-window construction → diversity selection
→ exact budget enforcement → reason artifact
```

## Contract

- Inputs: pool candidate ids, label-free metadata, one `UncertaintySignal`
  per pool candidate (ensemble disagreement + invalid flag, produced by
  `ensemble_uq`), a `DescriptorSet` covering the pool, an exact budget, and
  the uncertainty-window fraction.
- Invalid candidates (flagged, or non-finite disagreement) never rank; a
  non-finite disagreement without an invalid flag is rejected outright.
- The window keeps the top fraction of valid candidates by disagreement
  (deterministic id tiebreak) and is never smaller than the budget.
- Diversity selection runs farthest-point sampling inside the window using
  the same fail-closed descriptor validation as `diversity_select`.
- Exactly `budget` candidates emerge; anything else is a defect.
- Every selection record carries: `candidate_id`, `uncertainty_score`,
  `diversity_distance`, `source_group`, `rank`, `selection_reason`,
  `policy_version` — the complete machine-readable reason record.
- No labels, hidden data, or label-derived errors ever enter this skill.
- Nothing here is Cu-specific; the gate works for any pool with descriptors
  and disagreement signals.

## Scientific and security status

- Maturity: deterministic infrastructure.
- Scientific status: infrastructure; makes no scientific claim.
- External dependencies: numpy only.
- Side effects: files only below `ctx.step_dir`.
- Known failures: budget under-coverage after filtering, signal/pool
  mismatch, missing descriptors.
