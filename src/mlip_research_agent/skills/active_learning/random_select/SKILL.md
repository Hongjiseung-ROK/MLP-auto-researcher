# random_select

Label-free random acquisition arm (the active-learning control baseline).

## Contract

- Inputs: pool candidate ids, label-free metadata (`source_group`,
  `n_atoms`, `formula` only), an exact budget, a seed, and campaign/round
  ids. The skill never receives energies, forces, stresses, hidden labels,
  or any label-derived error — the metadata model forbids extra keys and the
  shared validators reject label-shaped keys fail-closed.
- Selection is deterministic: the pool is sorted, then permuted with
  `np.random.default_rng([seed, pool_size])`; the first `budget` entries are
  selected. Identical inputs and seed always yield identical selections.
- Exact budget: selecting fewer or more than `budget` is a defect; a budget
  larger than the pool is rejected before selection.
- Duplicates in the pool, metadata for ids outside the pool, and selections
  escaping the pool are all rejected.
- Output: one registered `random_selection.json` artifact with a
  machine-readable `SelectionRecord` per candidate (candidate_id,
  source_group, rank, selection_reason, policy_version).

## Scientific and security status

- Maturity: deterministic infrastructure; no model signal used.
- Scientific status: infrastructure; makes no scientific claim.
- External dependencies: numpy only.
- Side effects: files only below `ctx.step_dir`.
- Known failures: over-budget request, duplicate pool ids, missing metadata.
