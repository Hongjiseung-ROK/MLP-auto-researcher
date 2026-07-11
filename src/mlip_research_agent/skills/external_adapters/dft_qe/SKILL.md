# external_dft_qe

Deterministic Quantum ESPRESSO `pw.x` SCF input preparation. Adapter over the
locked upstream skill `quantum-chemistry/dft-qe` (`upstream.yaml`); no
upstream code is executed and **no QE binary is ever run** — the approved
Phase 2 DFT backend is ORCA, so this adapter is `mode="prepare"` only.

## Contract

- Input is a registered-run `StructureSet` path, one structure index, plane-wave
  cutoffs, a Monkhorst–Pack grid, and an explicit species→UPF filename map.
- Every species in the structure must have a pseudopotential entry; unsafe
  filenames (path separators) and `ecutrho < 4·ecutwfc` are rejected.
- Output is a byte-deterministic `pw_scf.in` plus a provenance artifact that
  records the upstream pin and states that execution is forbidden.
- Scientific status: `external_adapter_infrastructure_only`.

## Maturity and limits

- SCF calculation type only; `ibrav = 0` with explicit angstrom cell.
- No pseudopotential files are fetched or validated beyond the name.
- Running pw.x requires a separate owner decision and a compute-policy path.
