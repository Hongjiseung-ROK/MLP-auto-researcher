# oracle_reveal

Strict L3 simulated-oracle boundary for Phase 2. The acquisition controller
passes selected acquisition-pool record IDs; this SKILL alone receives the
qualified label-bearing dataset.

## Contract

- Inputs are run-directory-relative dataset/qualified-manifest/split paths,
  the config-pinned split-manifest SHA-256, immutable
  `campaign_id` + `round_id`, selected IDs, campaign/round budgets, and an
  optional prior oracle-state artifact.
- Outputs register six artifacts: request, decision, label batch, budget
  ledger, append-only training lineage, and resumable oracle state.
- Validation authenticates dataset lineage, record-to-group assignments, and
  split bytes before rejecting duplicates, unknown/protected IDs, changed
  repeated requests, stale campaign heads, changed budgets on resume, and
  over-budget batches before labels or state are written.
- The six files are built in a temporary directory and committed by one atomic
  directory rename. A retry registers the existing verified transaction and
  does not consume budget again. Locked compare-and-swap state commits reject
  stale writers; later rounds must name the unique registered campaign head.

## Scientific and security status

- Maturity: L3 for deterministic simulated-oracle access control.
- Scientific status: infrastructure supporting `pilot_only` runs; it makes no
  scientific performance claim.
- External dependencies: none beyond the repository core.
- Units: energy eV, forces eV/angstrom, optional virial stress kbar Voigt-6.
- GPU memory: none; CPU deterministic.
- Side effects: files only below `ctx.step_dir`.
- Human approval: the source dataset still requires H1 before claim-eligible
  use; this SKILL does not grant or bypass H1.
- Known failures: protected-id request, immutable-round conflict, campaign or
  round budget exhaustion, malformed/stale state.
