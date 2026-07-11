# external_pymatgen_structure

Space-group analysis for a `StructureSet` via pymatgen's `SpacegroupAnalyzer`.
Adapter over the locked upstream skill `data-processing/pymatgen-structure`
(`upstream.yaml`); no upstream code is executed.

## Contract

- Fails closed (`tool_error`, non-retryable) when pymatgen is not installed;
  it ships in the optional `atomistics` extra.
- Input: run-dir-relative `StructureSet` path plus `symprec`.
- Output: one registered JSON report — per-structure space-group symbol and
  number — carrying the upstream pin. Analysis failures on degenerate cells
  fail the whole call rather than skipping entries.
- Scientific status: `external_adapter_infrastructure_only`.

## Maturity and limits

- Symmetry reporting only; no structure transformation or file conversion.
