# Ralphthon @ICML integration contract

The official plugin is vendored byte-for-byte from
`team-attention/ralphthon-icml` commit
`a9f4f2583648ef4ca54f980f951ae393d153473f`, plugin version `0.5.0`.
`third_party/ralphthon-icml/SOURCE_MANIFEST.json` records every imported file
hash. Upstream SKILL files are immutable.

Cu/MACE and other MLIP work uses the official **General Track 1** path:

```text
official auto-research + vessl-cloud-onboarding
→ first-party ralphthon_mlip_track1 bridge
→ project MLIP controller and independent evaluator
→ optional approval-gated VESSL Cloud provider
→ 2–4 page Track 1 template and self-review
```

The official Training path remains a separate, optional execution of
`vessl-ai/vessl-cloud-cookbook/autoresearch` at commit
`97a0af14b0acae042162b1f70f17fbe2d570afa2`. It preserves the unchanged
Karpathy benchmark and `val_bpb`. Karpathy metrics cannot enter the MLIP route,
and MLIP energy/force metrics cannot enter the Karpathy evidence ledger.

This implementation creates only schemas, a copied official template, and an
evidence-binding pipeline. It does not authorize a claim-bearing paper. H2,
H3, and H5 remain open.
