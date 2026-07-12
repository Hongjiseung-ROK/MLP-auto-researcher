# diversity_select

Deterministic farthest-point selection over explicit descriptor artifacts.

## Contract

- Inputs: pool candidate ids, label-free metadata, an explicit
  `DescriptorSet` (candidate → fixed-length vector), and an exact budget.
- Descriptor sources: `fixture` (deterministic CI path) or the reviewed
  `mace_descriptor_adapter`, which consumes the label-free acquisition view
  and a content-pinned MACE checkpoint. No SOAP dependency.
- Validation rejects: budget over pool, descriptors not matching the pool,
  wrong dimensions, non-finite components, and duplicate descriptors.
- Selection: farthest-point sampling seeded at the candidate farthest from
  the pool centroid. Tie-breaking is deterministic everywhere
  (lexicographic candidate id), so identical inputs yield identical
  selections with no RNG at all.
- Every record carries `diversity_distance` (min distance to the
  already-selected set at selection time) and `source_group`; the output
  reports the mean pairwise distance of the selected set.
- On a clustered synthetic fixture, FPS provably covers more clusters than
  head-of-list selection (enforced by the tests).

## Scientific and security status

- Maturity: deterministic infrastructure.
- Scientific status: infrastructure; makes no scientific claim.
- External dependencies: numpy only.
- Side effects: files only below `ctx.step_dir`.
- Known failures: unimplemented descriptor source, malformed/duplicate
  descriptors, over-budget request.
