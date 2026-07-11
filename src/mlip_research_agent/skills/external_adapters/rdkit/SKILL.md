# external_rdkit

Seeded ETKDGv3 conformer generation for a SMILES string via RDKit. Adapter
over the locked upstream skill `molecular-representation/rdkit-repr`
(`upstream.yaml`); no upstream code is executed.

## Contract

- Fails closed when `rdkit` is not installed (optional `rdkit` extra).
- Unparseable SMILES and empty embeddings fail closed; conformer coordinates
  must be finite.
- The embedding seed comes only from `ctx.seed`, so identical inputs and seed
  reproduce identical conformers.
- Output: one registered JSON molecule record (canonical SMILES, symbols with
  explicit hydrogens, per-conformer angstrom coordinates) with the upstream
  pin. Optional MMFF94 optimization is opt-in.
- Scientific status: `external_adapter_infrastructure_only`.

## Maturity and limits

- No energies, charges, or descriptors; molecules only (no periodic systems).
