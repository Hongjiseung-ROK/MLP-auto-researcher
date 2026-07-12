# Reproducibility Statement

The reported pilot was executed from exact commit `50d09184021074e10985eee4c85951ab25e3ba9c` on one NVIDIA L4 with Torch 2.11.0+cu128 and CUDA 12.8. The dataset, split, foundation checkpoint, research specification, numerical metrics, three final models, figure source tables, and paper PDF are bound by SHA-256 receipts in `evidence_map.json`, `figure_manifest.json`, and `format_audit.json`.

Each adapted model restarted from identical MACE-MP-0 checkpoint bytes. Seeds 42, 43, and 44 used 20 full epochs and 160 optimizer steps. Same-calculator and checkpoint-reload drift were measured before the committee gate. The R figure script reads the evidence JSON with `jq`, asserts the headline floor and top-36 values, and emits both source CSVs and publication graphics.

The aborted pilot did not produce a completed campaign manifest; the paper therefore records a separate current-hash ledger and labels the evidence preliminary. No frozen-test labels or pool labels were accessed.
