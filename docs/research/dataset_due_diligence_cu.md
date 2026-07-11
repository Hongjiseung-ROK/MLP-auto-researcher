# Dataset due diligence: mlearn fcc Cu (Zuo et al. 2019)

Subagent D report, 2026-07-11 (UTC). Candidate Phase 2 benchmark dataset: the Cu
portion of the benchmark suite from Zuo et al., *A Performance and Cost Assessment
of Machine Learning Interatomic Potentials*, arXiv:1906.08888 (published as
J. Phys. Chem. A 2020, 124, 731–745), distributed via the "mlearn" GitHub
repository. This document records evidence for the human H1 gate. **No promotion
decision is made here.**

Legend: **[V]** verified directly (fetched/parsed by this audit), **[P]** stated
in the paper but not independently reproducible from the data alone, **[U]**
unknown / not stated.

---

## 1. Canonical source

- **[V]** Historical URL `github.com/materialsvirtuallab/mlearn` now **redirects**
  (HTTP 301 at the GitHub API level, repo id 183467954) to
  `github.com/materialyzeai/mlearn`. The repository was transferred from the
  Materials Virtual Lab org to `materialyzeai` at some point; GitHub serves both
  names transparently (old-org raw URLs return HTTP 200).
- **[V]** Default branch: `master`.
- **[V]** Head commit inspected (pinned): `10c427a5480c6281c15c64efaf869b03be04818f`
  (committed 2022-02-22T04:04:46Z, message "add extended Mo data"). The repo is
  archived/deprecated per its README ("all code has been moved to the updated
  [maml] package ... retained for reference").
- **[V]** Cu data lives at `data/Cu/training.json` and `data/Cu/test.json`
  (siblings: Ge, Li, Mo, Ni, Si).
- **[V]** Pinned raw URLs (both org names resolve; pin by SHA, not branch):
  - `https://raw.githubusercontent.com/materialyzeai/mlearn/10c427a5480c6281c15c64efaf869b03be04818f/data/Cu/training.json`
  - `https://raw.githubusercontent.com/materialyzeai/mlearn/10c427a5480c6281c15c64efaf869b03be04818f/data/Cu/test.json`
  - Same paths under `materialsvirtuallab/mlearn/<sha>/...` also return 200 today.

Stability note: the org transfer means the "canonical" name in citations
(`materialsvirtuallab/mlearn`) no longer matches the owning org. Redirects are
GitHub-lifetime-of-the-old-name only; record both names and the SHA.

## 2. License

- **[V]** `LICENSE` at the pinned commit is the **BSD 3-Clause License**
  (SPDX: `BSD-3-Clause`), "Copyright (c) 2019, Materials Virtual Lab".
  SHA-256 of the fetched LICENSE file:
  `f82460fc65e8e5dc3d00bfee0ce2a656f8c9d5af0dfc5511e9875e88051a55e9`.
- **Redistribution verdict: permitted.** BSD-3-Clause allows redistribution
  (including of the data files and derived small test fixtures) provided the
  copyright notice, condition list, and disclaimer are retained, and the
  Materials Virtual Lab name is not used for endorsement. Any vendored fixture
  must ship the notice and cite Zuo et al. 2019.
- Nuance: BSD-3-Clause is a software license applied repo-wide; the data files
  carry no separate data license. Practically this is how the community
  redistributes mlearn data (e.g. via maml), and the copyright holder chose the
  license for the whole repo. Low residual ambiguity; note for H1.

## 3. Downloaded files (pinned commit)

Local directory: `data/raw/mlearn/10c427a5480c6281c15c64efaf869b03be04818f/`
(**[V]** confirmed gitignored via `git check-ignore` → `.gitignore:32:data/raw/`).

| file | size (bytes) | SHA-256 |
|---|---|---|
| `training.json` | 5,246,466 | `9a427b8e21c10d410bc139e98a5bf69c10ca1cdb208d383819cebf4f4be0992f` |
| `test.json` | 608,400 | `1805a315ce8d0ce4a18bb761df4c4f619909ce114d598d426015d0071edc98c2` |
| `LICENSE` | 1,529 | `f82460fc65e8e5dc3d00bfee0ce2a656f8c9d5af0dfc5511e9875e88051a55e9` |
| `README.md` | 2,469 | `551282de9ee6e88df624d2aa47a552f5f662a8d5dcd545d059d71e243b9f02a7` |

GitHub's content API reports blob sizes 5,246,466 / 608,400 for
training/test.json, matching the downloads byte-for-byte.

## 4. Parse and characterization **[V]**

Script: `characterize_mlearn_cu.py` (scratchpad; stdlib-json only, deterministic,
run under the `mlip-research-agent` conda env). Format: JSON list of documents,
each `{structure (pymatgen Structure dict: lattice matrix + fractional sites),
num_atoms, outputs {energy, forces, virial_stress}, element, group, tag,
description}`.

| property | training.json | test.json |
|---|---|---|
| configurations | **262** | **31** |
| tag | all `train` | all `test` |
| species | Cu only | Cu only |
| total atoms | 27,416 | 3,178 |
| atoms/config min–max | 6–108 | 20–108 |
| lattice present (3×3 matrix) | 262/262 | 31/31 |
| label fields | energy, forces, virial_stress (all 262) | same (all 31) |
| energy [eV] | −442.935 … −23.399 | −442.907 … −79.002 |
| energy/atom [eV] | −4.1012 … −3.5155 | −4.1010 … −3.5441 |
| ‖force‖ [eV/Å] | 0.0 … 11.819 (mean 0.814) | 0.0 … 9.186 (mean 0.904) |
| virial_stress | 6-component Voigt, every config | same |
| non-finite labels | **0** | **0** |

`num_atoms` matches `len(sites)` in every document (asserted). Combined total:
**293 configurations** (mlearn's own split is 262:31 ≈ 89.4:10.6, matching the
paper's stated 90:10).

Periodicity: pymatgen Structure dicts always carry a full lattice matrix and are
fully periodic by construction; there is no explicit pbc field. Surface slabs are
periodic cells with vacuum along one axis (verify vacuum thickness before use if
it matters). **[V]** for lattice presence; periodic interpretation is the format's
convention.

### Units

- **[P]** Energies eV, forces eV/Å: the paper reports test errors in meV/atom and
  eV/Å; per-atom energies here (−3.5 … −4.1 eV/atom) match PBE fcc Cu cohesive
  energetics, so eV / eV/Å is confirmed by physical consistency.
- **Inferred, flag for H1**: `virial_stress` units are **kbar** (VASP `Voigt`
  convention). Evidence: a 2% strained Elastic config shows diagonal components
  ~26–35; Cu bulk modulus ≈ 140 GPa → 2% volumetric-ish strain ≈ 2.8 GPa =
  28 kbar. Component range over all configs: −151.1 … +244.1 (i.e. up to
  ~24 GPa at 10% strain / 3000 K), physically sensible in kbar and nonsensical
  in GPa or eV/Å³. Neither the repo README nor `mlearn/data.py` documents the
  unit explicitly. If stresses are ever used as labels, re-verify against one
  VASP recomputation.

### Groups **[V]**

Every document carries `group` and a human-readable `description` encoding the
generation protocol. Coarse groups:

| group | train | test | total | protocol (from descriptions) |
|---|---|---|---|---|
| Elastic | 108 | 13 | 121 | 3×3×3 fcc cell; 6 strain modes × 20 strains (−10%…+10%, 1% steps, no 0) = 120, plus 1 "Ground state crystal" |
| AIMD-NVT | 108 | 12 | 120 | 108-atom bulk; 40 snapshots each at 300 K, 1000 K, 3000 K |
| Vacancy | 36 | 4 | 40 | 107-atom (single vacancy); 20 snapshots each at 300 K, 3000 K |
| Surface | 10 | 2 | 12 | slabs `Cu_surface_mp/Cu_mp-30_(hkl)`: (100),(110),(210),(211),(221),(310),(311),(320),(321),(322),(331),(332); 6–48 atoms |

Fine-grained (temperature-resolved) strata: Elastic(+ground state) ×1,
AIMD-NVT@{300,1000,3000 K} ×3, Vacancy@{300,3000 K} ×2, Surface ×1 → **7–8
usable strata**, but only **5 distinct AIMD trajectories** (3 bulk + 2 vacancy).

### Discrepancies vs the arXiv text (data is ground truth)

- Paper (ar5iv extraction) says AIMD sampled "20 snapshots" and vacancy "40
  snapshots"; the data descriptions say the reverse (AIMD "of 40", Vacancy
  "of 20"), and the actual counts match the data (3×40 AIMD, 2×20 vacancy).
- Paper says strains "−10% to 10% at 2% intervals"; Cu data has 1% intervals
  (20 strain values per mode).
- Paper describes AIMD temperatures as 300 K plus multiples of the melting
  point; Cu data has literal 300/1000/3000 K (Cu Tm ≈ 1358 K → ≈0.74× and
  ≈2.2× Tm). Minor wording mismatch; possibly revised in the published JPCA
  version. None of this affects usability, but cite counts from the data, not
  the paper.

## 5. DFT provenance

From arXiv:1906.08888 full text (ar5iv), quoted status:

| item | value | status |
|---|---|---|
| Code | VASP 5.4.1 | **[P]** stated |
| Method | Projector augmented wave (PAW) | **[P]** stated |
| XC functional | PBE (GGA) | **[P]** stated |
| Plane-wave cutoff | 520 eV | **[P]** stated |
| k-points | 4×4×4 for the Cu supercells; Γ-only for AIMD | **[P]** stated |
| Convergence | electronic energy 10⁻⁵ eV; forces 0.02 eV/Å | **[P]** stated |
| Smearing | not stated in the retrieved text | **[U]** |
| POTCAR variant (e.g. `Cu_pv` vs `Cu`) | not stated | **[U]** |
| Surface slab source | Crystalium database (Materials Virtual Lab), derived from MP `mp-30` Cu (confirmed by the descriptions in the data) | **[P]**/[V] |

Nothing in the JSON files records INCAR/KPOINTS/POTCAR, so none of the DFT
settings are independently verifiable from the data — they are paper statements.
Per-atom energies and stress magnitudes are consistent with PBE Cu, which is a
weak consistency check, not verification.

## 6. Pretraining-overlap risk (vs MPTrj / MACE-MP-0)

- **[V]** Provenance direction: mlearn Cu data was generated in 2019 (repo
  copyright 2019, paper June 2019) as **custom** strained-bulk, AIMD-NVT,
  vacancy-AIMD, and surface-slab configurations computed by Zuo et al. with
  their own VASP settings. It is **not** a set of Materials Project relaxation
  trajectories. MPTrj (the MACE-MP-0 / CHGNet training corpus) was compiled in
  2023 exclusively from MP relaxation/static task trajectories of bulk MP
  entries. Therefore the perturbed frames here (strained cells, AIMD snapshots
  at 300–3000 K, vacancy snapshots) are **not MPTrj members**.
- Residual risks, stated honestly:
  1. The equilibrium fcc Cu prototype (MP `mp-30`) is in the Materials Project,
     and its relaxation frames are in MPTrj. The single "Ground state crystal"
     config and the near-equilibrium end of the Elastic group are therefore
     *distributionally* inside MACE-MP-0's training set (same structure, nearly
     identical PBE settings — MP also uses 520 eV/PBE). Any foundation model
     will look artificially good near equilibrium.
  2. The 12 surface slabs derive geometrically from `mp-30` via Crystalium, but
     MPTrj contains no slabs; slab frames are not MPTrj members. Still,
     unrelaxed/ideal slab geometries of an elemental fcc metal are trivially
     "seen" in spirit by any broadly-trained potential.
  3. Elemental fcc Cu generally is among the most heavily represented chemistries
     in any universal-potential corpus; expect strong zero-shot baselines. This
     is a benchmark-design consideration, not data leakage.
- **Verdict: LOW direct-overlap risk** (no frame-level membership in MPTrj
  except conceptually the one equilibrium structure); **moderate distributional
  familiarity** for foundation-model baselines, concentrated in the equilibrium/
  small-strain region. Report per-group metrics (especially AIMD@3000 K,
  Vacancy, Surface) so active-learning gains aren't masked by the easy
  equilibrium region.

## 7. Suitability for the Phase 2 split design

Requirements: initial set 32, acquisition pool 128–512, validation ≥10%, frozen
test ≥20%, GROUP-aware splitting. Total available: **293** configs, 4 coarse /
7–8 fine strata.

Feasible arithmetic (group-proportional, counts rounded):

| partition | size | % of 293 | Elastic (121) | AIMD (120) | Vacancy (40) | Surface (12) |
|---|---|---|---|---|---|---|
| frozen test | 59 | 20.1% | 24 | 24 | 8 | 3 |
| validation | 30 | 10.2% | 12 | 12 | 4 | 2 |
| initial set | 32 | — | 13 | 13 | 4 | 2 |
| acquisition pool | 172 | — | 72 | 71 | 24 | 5 |

- Pool = 172 sits inside the required 128–512 band, at the low end: **feasible,
  but with no headroom** — an acquisition schedule that consumes >172 labels
  cannot be satisfied. If the design wants a pool near 512, this dataset alone
  is too small.
- mlearn's own test.json (31 = 10.6%) does **not** meet the ≥20% frozen-test
  requirement by itself; a project-defined re-split over all 293 configs is
  needed (fine, since we re-split group-aware anyway — but this deviates from
  the published split and must be preregistered).
- Group richness: 4 coarse groups with temperature substrata are enough for
  stratified assignment and per-group reporting. Caveats:
  - **Surface has only 12 configs** — 2–3 per partition; per-group Surface
    metrics will be statistically weak.
  - **AIMD correlation**: snapshots within a trajectory are 0.1 ps apart and
    temporally correlated; there are only 5 distinct trajectories. Splitting
    snapshots of one trajectory across train/test leaks correlation; splitting
    whole trajectories (group = trajectory) leaves too few units. Recommended
    compromise: split within trajectories by contiguous time blocks, and state
    the residual correlation in the prereg.
  - Single element, narrow per-atom energy range (−4.10…−3.52 eV/atom): ideal
    for a controlled vertical slice, not a claim about chemical generality.

**Suitability verdict: SUITABLE for the Phase 2 vertical slice**, with the
constraints above (pool capped at ~172; custom ≥20% frozen test must be re-cut;
Surface group weak; AIMD block-splitting needed).

## Open items for the human H1 gate

1. Confirm acceptance of BSD-3-Clause repo-wide license as covering the data
   files (notice retention + citation of Zuo et al. 2019 in any fixture).
2. Confirm the kbar interpretation of `virial_stress` (only needed if stresses
   become labels); optionally verify with a single VASP recomputation.
3. Accept the deviation from the published 262/31 split (required to meet the
   ≥20% frozen-test rule) and preregister the re-split seed + group scheme.
4. Accept pool size 172 (low end of 128–512) or approve a second element
   (e.g. mlearn Mo/Si) to enlarge the pool.
5. Decide the AIMD trajectory-correlation policy (time-block splitting vs
   trajectory-level grouping).
6. POTCAR variant and smearing are unknown; irrelevant for mock/benchmark use,
   relevant only if we ever extend labels with new DFT.

## Reproduction

```bash
PIN=10c427a5480c6281c15c64efaf869b03be04818f
curl -sL -o training.json "https://raw.githubusercontent.com/materialyzeai/mlearn/$PIN/data/Cu/training.json"
curl -sL -o test.json     "https://raw.githubusercontent.com/materialyzeai/mlearn/$PIN/data/Cu/test.json"
shasum -a 256 training.json test.json   # must match section 3
```

Characterization script (deterministic, stdlib only) was run as
`env -u PYTHONPATH conda run -n mlip-research-agent python characterize_mlearn_cu.py`.
Machine-readable core facts: `docs/research/dataset_due_diligence_cu.json`.
