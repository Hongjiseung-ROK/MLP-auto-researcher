# external_phonopy

Finite-displacement supercell generation via phonopy. Adapter over the locked
upstream skill `analysis/phonopy` (`upstream.yaml`); no upstream code is
executed.

## Contract

- Fails closed when `phonopy` is not installed (optional `phonopy` extra).
- Input: one structure from a run-dir `StructureSet`, a diagonal supercell
  (entries 1–4), and a bounded displacement distance.
- Output: a `StructureSet` of displaced supercells (deterministic phonopy
  systematic displacements — no random displacements are requested) plus a
  provenance artifact with the upstream pin.
- Scientific status: `external_adapter_infrastructure_only`.

## Maturity and limits

- Displacement generation only: no force-constant fitting, band structures,
  or thermal properties; those need labeled forces and a gated compute path.
