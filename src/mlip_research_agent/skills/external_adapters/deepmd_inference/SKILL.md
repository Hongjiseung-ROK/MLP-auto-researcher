# external_deepmd_inference

Pinned DeePMD-kit model inference over a `StructureSet`. Adapter over the
locked upstream skill `machine-learning-potentials/deepmd-python-inference`
(`upstream.yaml`); no upstream code is executed.

## Contract

- The local model file's SHA-256 must match the pinned input hash **before**
  any deserialization; downloads are forbidden.
- Fails closed when `deepmd-kit` is not installed. deepmd-kit is an MLIP
  backend extra with its own framework pins — never co-install it with the
  `mace` extra in one environment.
- Species outside the model's type map, and non-finite predictions, fail
  closed. Output units are explicit eV and eV/angstrom.
- Output predictions are registered as `external_deepmd_predictions` — a
  deliberately different kind from `mlip_predictions`, so the WP5 evaluator
  cannot consume them without an explicit, owner-approved bridge.
- Scientific status: `external_adapter_infrastructure_only`.

## Maturity and limits

- CPU only; single-frame evaluation loop; no virial output.
