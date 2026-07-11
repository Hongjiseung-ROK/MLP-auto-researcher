# Dataset card — cu_phase2

**Source:** mlearn benchmark (Zuo et al. 2019, arXiv:1906.08888), Cu subset.
Canonical repo `github.com/materialyzeai/mlearn` (transferred from
`materialsvirtuallab/mlearn`; archived, superseded by maml). Pinned commit
`10c427a5480c6281c15c64efaf869b03be04818f`.

**License:** BSD-3-Clause, Copyright (c) 2019 Materials Virtual Lab
(`license.txt`). Redistribution permitted with notice retention; cite the
paper. Small test fixtures are committed under this license; full data is
fetched by `scripts/data/fetch_cu_benchmark.py` (checksum-verified) and
never committed.

**Contents:** 293 periodic fcc-Cu configurations (262 train + 31 test in the
source's own split, which this project does NOT reuse — see below) with
total energy (eV), atomic forces (eV/Å), and 6-component virial stress
(unit inferred kbar — undocumented upstream, H1 open item; stress is
excluded from training exports until confirmed).

| Group | Configs | Fine-grained subgroups | Split units |
|---|---:|---|---:|
| Elastic | 121 | 6 strain modes + ground state | 7 |
| AIMD-NVT | 120 | 300/1000/3000 K trajectories | 12 (time blocks of 10) |
| Vacancy | 40 | 300/3000 K trajectories | 8 (time blocks of 5) |
| Surface | 12 | 12 individual mp-30 slabs | 12 |
| **Total** | **293** | | **42** |

**Level of theory:** VASP 5.4.1, PAW, PBE, 520 eV cutoff, 4×4×4 k-mesh
(Γ-only AIMD) — paper-stated; the JSONs carry no INCAR/POTCAR metadata.
Smearing and POTCAR variant unknown.

**Split policy:** the source's own 31-config test set (10.6%) does not meet
the preregistered ≥20% frozen-test rule; a group-aware project re-split over
the 42 split units is used instead (AIMD/vacancy trajectories split in
contiguous time blocks so temporally correlated frames never straddle a
partition boundary).

**Pretraining-overlap risk vs MACE-MP-0:** LOW direct overlap (custom 2019
strain/AIMD/vacancy/surface frames are not MPTrj members); MODERATE
distributional familiarity near equilibrium fcc Cu (mp-30 is in MPTrj).
Consequence: zero-shot numbers are contextual only; per-group metrics are
reported so the easy equilibrium region cannot mask differences.

**Qualification:** `qualification_report.json` (11 deterministic checks, all
passing as of 2026-07-11; dataset content SHA-256
`cff3ea7a346d4f07d6f349d734021395bb72ee1ac36ec0deff0f103528600433`).
Claim-eligible use requires the recorded H1 approval against that hash
(`src/mlip_research_agent/data/registry.py::load_qualified_dataset` enforces
this fail-closed).

**Known caveats for H1 review:**
- virial stress unit inferred, not documented;
- DFT settings are paper statements, not embedded metadata;
- paper-vs-data discrepancies recorded in
  `docs/research/dataset_due_diligence_cu.md` (snapshot counts, strain
  interval, temperature description);
- BSD-3 is a software license applied repo-wide to data.
