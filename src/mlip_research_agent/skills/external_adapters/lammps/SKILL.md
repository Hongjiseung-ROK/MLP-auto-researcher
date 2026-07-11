# external_lammps

Deterministic LAMMPS input/data preparation for the DeepMD pair style.
Adapter over the locked upstream skill `molecular-dynamics/lammps-deepmd`
(`upstream.yaml`); no upstream code is executed and **`lmp` is never run** —
MD execution is gated on a separate owner decision.

## Contract

- Input is a registered-run `StructureSet` path, one structure index, a safe
  DeepMD model filename, and bounded NVE/timestep/step parameters.
- v1 supports orthorhombic cells only; anything else fails closed.
- Outputs a byte-deterministic `in.lammps` + `data.lammps` (atomic style,
  sorted species→type map, ASE masses) plus a provenance artifact recording
  the upstream pin and the execution prohibition.
- Scientific status: `external_adapter_infrastructure_only`.

## Maturity and limits

- Prepare-only; no trajectory, thermo output, or restart handling.
- The referenced DeepMD model file is not fetched, hashed, or validated here;
  execution-time validation belongs to the (future, gated) run path.
