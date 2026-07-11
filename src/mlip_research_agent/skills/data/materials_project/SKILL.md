# SKILL: materials_project_recon

## Purpose
Typed reconnaissance against the Materials Project API: verify authentication,
survey summary metadata for a chemical system, and fetch canonical structures
into the repo's deterministic first-party structure format.

## Scientific status
**Reconnaissance/metadata only — outputs are NOT claim-eligible dataset
sources.** MPTrj/Materials Project data must not become the independent test
set for MACE-MP-style models (the model was trained on it); using MP data as
evaluation or training data requires an owner-approved PIVOT recorded in
`docs/OPEN_QUESTIONS.md`. This skill registers no `Claim`s.

## Supported use cases
- `auth_check`: cheapest possible authenticated probe (one material, one field)
  reporting authenticated true/false with a failure-class detail.
- `summary_search`: survey a chemsys, storing exactly the requested fields.
- `get_structures`: fetch canonical structures for explicit `mp-<n>` ids and
  convert them (pymatgen → ASE → `StructureRecord`) into a `StructureSet`.

## Non-goals
- Dataset construction, labeling, or anything feeding claim-eligible steps.
- Credential handling: the key is loaded by
  `secrets/api_env.py::mp_api_key_env` from repo-root `api.env` into the
  process environment for the duration of the call only. This skill never
  opens `api.env`, never sees the key value, and never writes key material,
  raw provider exception text, or HTTP headers into any artifact or message.

## Typed inputs — `schema.MaterialsProjectInput`
`operation` (`auth_check` | `summary_search` | `get_structures`), `chemsys`
(dash-separated element symbols, default `"Cu"`), `material_ids`
(`mp-<digits>`), `fields` (must include `material_id`; only these fields are
fetched/stored), `max_results` (1–500, default 100), `use_cache` (default
true). Dev knobs (tests only): `cache_root`, `api_env_file`.

## Typed outputs — `schema.MaterialsProjectOutput`
`operation`, `authenticated` (true iff a live authenticated call succeeded;
false on cache hits), `n_results`, `material_ids`, `response_artifact`,
`provenance_artifact`, `from_cache`, `detail` (never secrets).

## Required dependencies
`mp-api` + `python-dotenv` via the `materials-project` extra (lazy-imported;
missing extra raises a guided `TOOL_ERROR`). `get_structures` additionally
uses pymatgen's `AseAtomsAdaptor` (`atomistics` extra).

## Preconditions
Repo-root `api.env` defines one of `MP_API_KEY`, `MATERIALS_PROJECT_API_KEY`,
`MATERIALS_PROJECT_KEY` (network operations only; cache hits need no key).

## Validation rules (`validators.py`)
All raise `VALIDATION_ERROR`, non-retryable: chemsys pattern, `mp-\d+` id
pattern (non-empty for `get_structures`), fields non-empty snake_case and
containing `material_id`, `max_results` bounds.

## Failure taxonomy
- `VALIDATION_ERROR` — malformed inputs (non-retryable).
- `TOOL_ERROR` — missing `mp-api` extra or missing key (non-retryable);
  401/403 auth failures (non-retryable); transient network/provider errors
  (retryable).
- `RESOURCE_EXHAUSTED` — 429 rate limiting (retryable).
Error messages carry only the exception *type name* and a classification
label, never raw provider messages or headers.

## Retry and recovery policy
Inline deterministic retry loop: max 3 attempts, retryable classes only,
fixed short backoff (no jitter). `auth_check` converts terminal failures into
`authenticated=false` output instead of raising; the other operations raise
`SkillError` for the executor's bounded recovery.

## Caching
Sanitized payloads (requested fields only / structure records only) are keyed
by the sha256 of the canonical query JSON under
`data/cache/materials_project/` (gitignored, outside the run dir by design —
a sanctioned exception to the step_dir-only rule so identical recon queries
across runs skip the network). Cache hits perform no network call and still
write + register the step artifacts. `auth_check` is never cached (its whole
point is a live probe).

## Produced artifacts
- `summary_search`: `response.json` (kind `mp_response`) + `provenance.json`
  (kind `mp_provenance`).
- `get_structures`: `structures.json` (kind `mp_structures`, `StructureSet`
  with `perturbation_scale=0.0`) + `provenance.json`.
- `auth_check`: none.
Provenance carries the query params, `mp-api` client version, UTC retrieval
timestamp, material ids, and the sha256 of the response file. Because
provenance contains timestamps, do not wire this skill into
byte-reproducibility-gated scientific DAGs.

## Cost class / permission level
`cheap` / `auto` — bounded metadata queries; cached repeats are free.

## Minimal example
`examples/example_input.json`.

## Correctness tests
`tests/test_materials_project_skill.py` (offline: fake `mp_api.client` module,
synthetic key in a tmp env file, no network): missing-key error, auth_check
success/auth-failure/rate-limit mapping, sanitization + deterministic
ordering + artifact/provenance content, no key material in any written byte,
fcc-Cu pymatgen→StructureSet conversion, cache hit skips network,
retry-loop bounds.
