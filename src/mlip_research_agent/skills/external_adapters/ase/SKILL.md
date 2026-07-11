# external_ase

Deterministic ASE bulk-structure builder. Adapter over the locked upstream
skill `atomistic-workflows/ase` (see
`external-skills/jinzhezenggroup/computational-chemistry-agent-skills/upstream.yaml`);
no upstream code is executed — the capability is implemented natively against
ASE, which is a core repository dependency.

## Contract

- Builds one conventional bulk crystal (`fcc`/`bcc`/`sc`/`hcp`/`diamond`) plus
  optional rattled copies; the rattle uses `np.random.default_rng(ctx.seed)`
  only, so identical inputs and seed reproduce identical bytes.
- Outputs the repository's first-party `StructureSet` JSON plus a provenance
  artifact carrying the upstream repo/commit/skill path and the input echo.
- Non-finite positions/cells fail closed; `hcp` with `cubic=true` is rejected
  at the schema.
- Scientific status: `external_adapter_infrastructure_only` — outputs are
  fixtures/infrastructure, never claim evidence.

## Maturity and limits

- Rattled copies are Gaussian displacements, not thermal sampling.
- No calculator attachment, no relaxation, no MD — structure generation only.
- Side effects: artifacts only below `ctx.step_dir`.
