# ensemble_uq

Ensemble force-disagreement ranking for acquisition. Explicitly a
*disagreement* signal — the contract never calls it true uncertainty.

## Contract

- Inputs: pool candidate ids, label-free metadata, and at least three
  explicitly identified ensemble members. Every member carries a unique
  `member_id` and a distinct 64-hex `provenance_sha256` (model manifest or
  checkpoint hash); identity collisions are rejected at schema validation.
- Every member must predict forces for exactly the pool; when energy
  disagreement is enabled, energy-per-atom predictions must cover the pool
  too. Force arrays must be `(n_atoms, 3)`.
- Disagreement: per-atom standard deviation of force components across
  members, aggregated as the mean per-atom disagreement norm; optionally the
  std of energy-per-atom across members.
- NaN/infinite predictions never rank: the candidate is excluded and the
  offending members are recorded in `invalid_member_flags`.
- Ranking is stable and deterministic: disagreement descending, candidate id
  ascending as the tiebreak.
- The skill receives no labels and no label-derived errors (label-shaped
  metadata keys are structurally impossible; see `selection_types.py`).
- Synthetic deterministic ensemble members are the default CI path; real
  MACE ensembles plug in through the same schema without contract changes.

## Scientific and security status

- Maturity: deterministic infrastructure over supplied predictions.
- Scientific status: infrastructure; disagreement values make no calibrated
  uncertainty claim.
- External dependencies: numpy only.
- Side effects: files only below `ctx.step_dir`.
- Known failures: <3 members, duplicate identities, pool/prediction
  mismatch, malformed force shapes.
