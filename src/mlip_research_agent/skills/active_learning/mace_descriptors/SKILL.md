# mace_descriptors

Label-free, fixed-foundation MACE representations for deterministic batch
diversity selection.

## Contract

- The only structure input is a registered `AcquisitionView`; its schema
  structurally excludes energy, force, stress, error, and label fields.
- The checkpoint filename, size, SHA-256, MACE version, and species contract
  are verified before model loading. Runtime aliases and downloads are
  forbidden.
- `MACECalculator.get_descriptors(..., invariants_only=True)` produces per-atom
  features. The adapter mean-pools each structure, standardizes each feature
  over the current unlabeled pool only, drops dimensions with population
  standard deviation at or below `1e-12`, and L2-normalizes each retained row.
- Candidate order and retained-feature order are deterministic. The complete
  normalization rule and provenance are written beside the `DescriptorSet`.
- Output is compatible with `diversity_select` and `decision_gate`; descriptor
  distances are an acquisition geometry, not calibrated uncertainty.

## Scientific limits

- The fixed foundation representation is used to isolate diversity filtering
  from representation drift during fine-tuning.
- Mean pooling can hide a rare atomic environment. Tail-oriented conclusions
  must therefore be supported by force-error diagnostics, not descriptor
  distance alone.
- GPU execution requires the campaign's compute authorization and attestation.
