# external_dpdata_cli

Convert the repository's first-party labeled `NormalizedDataset` JSON into
deepmd/npy systems via dpdata. Adapter over the locked upstream skill
`data-processing/dpdata-cli` (`upstream.yaml`); no upstream code is executed
(the upstream skill wraps `uvx dpdata`; this adapter uses the dpdata Python
API directly under the same pinned registration).

## Contract

- Fails closed when `dpdata` is not installed (optional `dpdata` extra).
- Labels must be finite; units are fixed eV / eV/Å / Å and recorded in the
  conversion manifest.
- Configurations are grouped by exact symbol ordering (dpdata systems require
  a fixed atom layout); each group becomes one `deepmd/npy` system directory.
- Every produced file is SHA-256-hashed into a registered conversion manifest
  and individually registered; the manifest binds the source dataset content
  hash.
- Scientific status: `external_adapter_infrastructure_only`.

## Maturity and limits

- One direction only (first-party → deepmd/npy); no virials; no round-trip.
