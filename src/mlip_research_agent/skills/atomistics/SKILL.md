# SKILL: structure_generation

## Purpose
Generate a deterministic set of candidate atomistic configurations (pristine
bulk plus seeded Gaussian perturbations of increasing amplitude) for downstream
acquisition and labeling.

## Supported use cases
- Bootstrap candidate pools for the mock active-learning loop.
- Deterministic fixtures for tests of acquisition/labeling/training skills.

## Non-goals
- MD- or relaxation-driven candidate proposal (post-v0, needs an MLIP backend).
- Defects, surfaces, interfaces, multi-component alloys (post-v0).
- Any DFT-quality geometry.

## Typed inputs — `schema.StructureGenerationInput`
`formula` (single element), `crystal_structure` (fcc|bcc|sc|hcp|diamond),
`lattice_constant` (1–20 Å), `supercell`, `n_candidates` (>1), `max_perturbation` (Å).

## Typed outputs — `schema.StructureGenerationOutput`
`structures_artifact`, `structures_path` (run-dir-relative `structures.json`,
first-party format `structures_io.StructureSet`), `n_structures`,
`n_atoms_per_structure`, `elements`.

## Required dependencies
`ase` (bulk builder), `numpy` (seeded rng). Bootstrap env only.

## Preconditions
Writable `ctx.step_dir`; seed provided by the executor via `ctx.seed`.

## Validation rules (`validators.py`)
- Lattice keyword must be supported (`VALIDATION_ERROR`, non-retryable).
- No interatomic distance below 0.5 Å after perturbation
  (`SIMULATION_INSTABILITY`, retryable with repair `max_perturbation=0.05`).

## Failure taxonomy
`VALIDATION_ERROR` (bad lattice/inputs), `SIMULATION_INSTABILITY` (unphysical
geometry after perturbation).

## Retry and recovery policy
Geometry failures are retryable once with the reduced-amplitude repair
(`REFINE`); the executor bounds total attempts by the campaign budget.

## Produced artifacts
`steps/<step_id>/structures.json` (kind `structure_set`), registered with
sha256 in the run manifest.

## Provenance fields
Seed, perturbation scale per structure (stored in each record), artifact hash.

## Cost class / permission level
`trivial` / `auto`.

## Minimal example
See `examples/example_input.json`; run via `tests/test_structure_generation.py`.

## Correctness tests
`tests/test_structure_generation.py`: deterministic bytes for equal seeds,
differing output for different seeds, unsupported-lattice rejection,
minimum-distance rejection with repair suggestion.
