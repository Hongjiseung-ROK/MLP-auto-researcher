# VESSL MLIP infrastructure replay runbook

Status: **locally implemented and dry-run verified; no VESSL resource has been
created.** This is not the same-session Colab contract.

## Topology

1. Job 1 checks out one exact 40-character commit detached, attests exactly one
   A100, verifies the bounded label view and immutable hashes, runs baseline and
   iteration 1, independently evaluates it, uploads a strict manifest, and exits.
2. The host pulls and hash-verifies Job 1, collects the three specialized reviews,
   seals their synthesis and proposal 2, and creates no idle GPU wait.
3. Job 2 uses the same commit, resource spec, image, label view, dataset, split,
   checkpoint, and environment lock. It verifies Job 1 and review receipts,
   executes iteration 2 only, uploads the final strict manifest, and exits.
4. The host verifies both Jobs are terminal, grades the combined trace, and
   reports every surviving storage or volume exposure.

An existing started operation without a completion receipt is ambiguous and is
never retried. One optimizer receipt is permitted per iteration.

## Transport boundary

Only `/runs/<run-id>/inputs/bounded_label_view.json`, its manifest, sealed Job 1
state, and the sealed review bundle may cross the boundary. Full datasets,
frozen-test IDs or labels, acquisition-pool labels, stress labels, secrets,
`.env` files, and unexpected files are rejected before upload. Pull-back uses a
fresh staging directory, exact manifest coverage, SHA-256 verification, and an
atomic publish.

Volume deletion remains a separate destructive action requiring explicit
confirmation after successful pull-back.

## Current blocker and cost gate

The current docs advertise `vesslctl job create --file`, but do not publish its
JSON file schema; the pinned cookbook uses flags. Because `vesslctl` is not
installed, the live export schema cannot be calibrated. Generated files are
therefore marked `normalized_dry_run_control` and
`vesslctl_file_schema_verified: false`. Submission fails closed.

Before any future Job creation, present one live card binding organization,
team, cluster, exact single-A100 resource spec, hourly price, credit, exact
image, duration, compute estimate, storage capacity/rate/duration/estimate,
mounts, timeout, cleanup, and approval artifact hash. Any changed field voids
approval.

The exact billable command remains blocked:

```bash
vesslctl job create --file <live-schema-verified-job-config.json>
```

Workspace, storage, and volume creation are not implemented by the provider.
