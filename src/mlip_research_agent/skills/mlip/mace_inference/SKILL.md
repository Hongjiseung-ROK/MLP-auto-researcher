# mace_inference

Pinned MACE energy/force/stress inference over the repository's deterministic
`StructureSet` format.

## Contract

- Requires an exact local checkpoint path and a committed checkpoint manifest.
  The filename, byte size, SHA-256, `mace-torch` version, species, source URL,
  license, and training-data statement are checked before deserialization.
- Convenience aliases and runtime downloads are forbidden. A missing or
  mismatched checkpoint fails closed.
- Inputs bind the ordered structures to dataset record ids plus exact dataset
  content and split semantic hashes; duplicate or count-mismatched ids fail.
- Produces registered `mlip_predictions` and `model_manifest` artifacts with
  explicit eV, eV/angstrom, and eV/angstrom^3 Voigt-6 units. Both artifacts
  carry the dataset/split identities required by the independent evaluator.
- Finite outputs are mandatory. Requested CUDA without a visible device fails;
  CPU never silently becomes a scientific GPU run.

## Maturity and limits

- Maturity: L2 locally executable; L3 requires CPU/GPU parity and a reviewed
  Colab fixture for the H2-selected checkpoint.
- Scientific status: candidate/infrastructure only until H1/H2/H3 gates.
- External dependency: exact `mace-torch` version in the checkpoint manifest.
- Determinism: CPU results are tolerance-tested; GPU parity uses declared
  tolerances because bitwise equality is not promised.
- Side effects: artifacts only below `ctx.step_dir`; checkpoint cache is
  read-only to the SKILL.
- Known failures: missing/hash-mismatched checkpoint, unsupported species,
  dtype/device mismatch, non-finite model output.
