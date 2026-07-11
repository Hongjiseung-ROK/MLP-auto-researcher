# MACE checkpoint comparison memo

Status: **evidence only; neither candidate is H2-selected.** This comparison
uses only the 32-record initial-labeled partition. Validation, frozen-test,
and stress-test labels were not accessed. No energy offset was fitted or
subtracted, and no scientific claim is minted.

## Fixed inputs

- Qualified dataset content SHA-256:
  `bc9b78ca5e3b95f91bf34bbc3641a3d6e3f92338b4e3d97065165157848cfc48`
- Normalized-manifest file SHA-256:
  `ac3655e41ce4327ba18e2a603866b97b00b839ff04f16bb2cb86a3d8ba8a70d3`
- Split-manifest file SHA-256:
  `80d9b95083ebba9d8f988a461525b326ba807c79da834b066d1917afc9d8a15e`
- Split semantic SHA-256:
  `4fb9397aeb3ec4ab3a52078b10663beff912d1bb209b5969b06406885ec6bfc9`
- Environment: `mace-torch 0.3.16`, torch 2.12.1, e3nn 0.4.4, local CPU,
  float64 inference.

## Candidate metadata

| Candidate | Exact bytes | Parameters | Training domain | Cu E0 stored in checkpoint | Status |
|---|---:|---:|---|---:|---|
| MACE-MP-0 small | 32,581,838; SHA-256 `2ddb079c…5736` | 3,847,696 | MPTrj, DFT PBE+U, 89 elements | −3.2520726959 eV | `candidate_only` |
| MACE-MPA-0 medium | 79,462,305; SHA-256 `75428afe…fb638` | 9,063,204 | MPTrj + sAlex, DFT PBE+U, 89 elements | −0.6002584600 eV | `candidate_only` |

Both assets are MIT licensed, explicitly support Cu, load through verified
local paths under `mace-torch 0.3.16`, and pass the same finite-output plus
translation/rotation/atom-order CPU fixture (`3 passed` for each). The MPA-0
asset is about 2.44 times larger; L4 optimizer-state and activation memory are
not inferred from file size and remain staging evidence.

Official metadata sources:

- <https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html>
- <https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mpa_0>
- <https://github.com/ACEsuit/mace-foundations/blob/main/LICENSE>
- <https://github.com/ACEsuit/mace/blob/v0.3.16/mace/calculators/foundations_models.py>

## D0 raw energy-offset diagnostic

The diagnostic reports `(model total energy − target total energy) / n_atoms`
without fitting an offset. Its artifact is
`artifacts/research/mace-checkpoint-comparison-4bdfa06/steps/e0-diagnostic/e0_diagnostic.json`
(gitignored), SHA-256
`cd588fa7d8ce6d681733c24aaee049d96f5059e80bbe82002acc1b5f5c235097`.

| Candidate | Mean (eV/atom) | Std. dev. | MAE | P95 absolute | Max absolute group mean |
|---|---:|---:|---:|---:|---:|
| MACE-MP-0 small | 0.00984363 | 0.01124825 | 0.01385895 | 0.02075376 | 0.02131393 |
| MACE-MPA-0 medium | −0.00313001 | 0.00357815 | 0.00365314 | 0.00920690 | 0.01116795 |

These values do not select a checkpoint. No pass/fail threshold was frozen;
the residual mixes atomic-reference alignment with model error; target
POTCAR/smearing metadata are incomplete; and MPA-0 may be more distributionally
familiar with the target through sAlex. Choosing the smaller observed D0
residual after seeing it would turn this diagnostic into an undeclared model
selection metric.

## Neutral disposition

Both checkpoints remain eligible comparison candidates. Before the next H2
freeze proposal, run the same explicit-E0, one-optimizer-step, checkpoint
round-trip, and bounded L4 memory/parity gates. The resulting packet may state
hard-gate eligibility and limitations, but the owner still selects the
checkpoint. Any fitted E0, changed reference policy, or dataset/model-family
pivot requires explicit approval.
