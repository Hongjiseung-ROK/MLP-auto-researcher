# Security boundary

- This bridge never executes a VESSL command and never handles credentials.
- Billable VESSL operations require a separately sealed, live cost approval.
- Protected label views, raw datasets, checkpoints, `.env` files, and token
  material are forbidden competition outputs.
- The optional W&B path is offline-only here. Online synchronization requires
  separate entity, project, visibility, allowlist, and directory approval.
- Outputs are infrastructure-only and cannot promote H2/H3/H5 or support a
  scientific claim.
