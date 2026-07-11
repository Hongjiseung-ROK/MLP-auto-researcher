# mlip_metrics

Independent, leakage-safe aggregate evaluation for registered MLIP prediction
artifacts.

## Contract

- Inputs are artifact IDs for predictions, the normalized dataset, dataset
  manifest, frozen split manifest, and model manifest. Every artifact must be
  registered under its expected kind and pass SHA-256 integrity verification.
- The evaluator accepts only `validation`, `frozen_test`, or `stress_test` and
  derives the complete expected record-ID set internally from the split.
- Prediction records carry explicit `record_id`, `energy_unit=eV`, and
  `force_unit=eV/angstrom`. Missing, extra, duplicate, unknown, malformed, or
  non-finite predictions fail closed.
- Target labels must also declare exact `eV` and `eV/angstrom` units. The
  prediction structure-set hash must match the model manifest.
- The output contains aggregate and group-wise metrics only. It never emits
  labels, record IDs, per-record errors, rankings, or high-error examples.
- Group metrics use coarse `top_group` values and suppress groups smaller than
  three structures, preventing singleton groups from becoming per-record
  errors under another name.
- Frozen-test and stress-test calls require an exact registered authorization
  binding dataset, split, prediction, and model identities. One authorization
  may be consumed only once per run; validation evaluation needs none.
- The metric artifact records every input artifact ID and hash plus the
  exact source-code hash and JSON pointers a later scientific claim must bind.
  This SKILL mints no claim.

## Metric definitions

- Energy MAE: structure-weighted mean absolute total-energy error divided by
  the structure atom count, in eV/atom.
- Force MAE/RMSE: componentwise over every evaluated Cartesian force component.
- P95 force error: Type-7 linear 95th percentile of per-atom Euclidean
  force-vector errors.
- High-error fraction: fraction of structures whose mean per-atom force-vector
  error is strictly greater than the explicitly supplied threshold.
- Group-wise metrics repeat the same definitions for each dataset `group_id`.

## Scientific and security status

- Scientific status: deterministic infrastructure for `pilot_only` evidence;
  metric integrity alone does not establish replication or publication status.
- Side effects: one deterministic `mlip_metrics.json` below `ctx.step_dir`,
  registered as `mlip_metrics`.
- Test-label boundary: only this evaluator reads selected protected labels;
  callers receive aggregates, never examples or error rankings.
- Registry hashes prove integrity, not producer authenticity; trusted workflow
  assembly and human evaluation authorization remain separate controls.
- GPU/dependencies: none; CPU-only repository core.
