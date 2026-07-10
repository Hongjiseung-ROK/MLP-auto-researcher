# SKILL: mock_dft_labeling

## Purpose
Deterministic stand-in for DFT labeling: computes Lennard-Jones energies and
max forces for selected structures, records settings provenance, and exposes a
convergence-failure path with a bounded REFINE repair.

## Supported use cases
- Exercise the labeling stage of the mock AL loop end to end.
- Demonstrate recoverable-failure injection (`inject_failure_times`).

## Non-goals
- Physically meaningful energies.
- Real ORCA/VASP/QE execution, queueing, or license handling (post-v0, gated
  on the DFT-backend decision in `docs/OPEN_QUESTIONS.md`).

## Typed inputs — `schema.LabelingInput`
`structures_path`, optional `selection_path`, `method` (`mock_lj` only),
`scf_damping` (repair knob, recorded), `inject_failure_times` (0–3, demo only).

## Typed outputs — `schema.LabelingOutput`
`labels_artifact`, `labels_path`, `n_labeled`, `method`.

## Required dependencies
`ase` (LennardJones calculator), `numpy`.

## Preconditions
Structure set (and selection, if given) exist under the run directory.

## Validation rules
- Unknown method → `VALIDATION_ERROR`, non-retryable.
- Non-finite energy/force → `CONVERGENCE_FAILURE`, retryable.

## Failure taxonomy
`VALIDATION_ERROR`, `CONVERGENCE_FAILURE` (mock SCF analogue).

## Retry and recovery policy
Injected convergence failures recommend `REFINE` with `scf_damping=0.7`; the
executor merges the repair into the next attempt's inputs and records it in
`attempted_repairs` — convergence settings never change silently. Attempts are
bounded by the campaign's `max_retries_per_step`.

## Produced artifacts
`steps/<step_id>/labels.json` (kind `label_set`) including the full settings
block (`sigma`, `epsilon`, `rc`, `scf_damping`).

## Provenance fields
Method, settings, selection indices, artifact hash.

## Cost class / permission level
`cheap` / `auto` (the real DFT skill will be `gated` / `human_approval`).

## Minimal example
`examples/example_input.json`.

## Correctness tests
`tests/test_mock_dft_labeling.py`: determinism, selection subsetting,
failure injection honors attempt count, repair params surface, unknown method
rejected.
