# SKILL: mock_acquisition

## Purpose
Select a labeling batch from candidate structures using a deterministic
uncertainty proxy (perturbation amplitude + seeded noise) or seeded random
selection.

## Supported use cases
- Exercise the acquisition stage of the mock AL loop.
- Baseline `random` strategy for future AL-vs-random comparisons.

## Non-goals
- Real committee disagreement, diversity scoring, energetic-viability filters
  (post-v0 AL policy mixer).

## Typed inputs — `schema.AcquisitionInput`
`structures_path`, `n_select` (>0), `strategy` (`mock_uncertainty` | `random`).

## Typed outputs — `schema.AcquisitionOutput`
`selection_artifact`, `selection_path`, `n_selected`, `strategy`.

## Required dependencies
`numpy`; reads the atomistics skill's `StructureSet` format.

## Preconditions
Structure-set artifact exists under the run directory.

## Validation rules
- Missing structure set → `VALIDATION_ERROR`, non-retryable.
- `n_select` must not exceed the candidate count → `VALIDATION_ERROR`.

## Failure taxonomy
`VALIDATION_ERROR`.

## Retry and recovery policy
Validation failures are non-retryable; the executor escalates.

## Produced artifacts
`steps/<step_id>/selection.json` (kind `selection`) with indices and scores.

## Provenance fields
Strategy, per-candidate scores, seed (via executor), artifact hash.

## Cost class / permission level
`trivial` / `auto`.

## Minimal example
`examples/example_input.json` (paths are run-dir-relative).

## Correctness tests
`tests/test_mock_acquisition.py`: determinism, uncertainty proxy prefers
high-perturbation candidates, oversized selection rejected.
