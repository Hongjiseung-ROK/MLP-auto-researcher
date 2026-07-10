# SKILL: mock_literature

## Purpose
Deterministic stand-in for literature retrieval: keyword-scores a fixed fixture
corpus and emits a citable reference-list artifact.

## Supported use cases
- Exercise the literature stage of the campaign DAG without network access.
- Provide stable reference fixtures for report generation tests.

## Non-goals
- Real search, PDF indexing, figure/table chunking (post-v0, PARNESS-style).
- Citation verification against external databases.

## Typed inputs — `schema.LiteratureInput`
`query` (3–500 chars), `max_results` (1–10).

## Typed outputs — `schema.LiteratureOutput`
`references_artifact`, `references_path`, `n_references`.

## Required dependencies
Standard library only.

## Preconditions
Writable `ctx.step_dir`.

## Validation rules
Reference keys must be unique (`TOOL_ERROR`, retryable).

## Failure taxonomy
`TOOL_ERROR` (corrupt retrieval result).

## Retry and recovery policy
Retryable within the executor's bounded budget; no skill-side repair params.

## Produced artifacts
`steps/<step_id>/references.json` (kind `reference_list`).

## Provenance fields
Query string embedded in the artifact; artifact hash in the manifest.

## Cost class / permission level
`trivial` / `auto`.

## Minimal example
`examples/example_input.json`.

## Correctness tests
`tests/test_mock_literature.py`: determinism, ranking by keyword overlap,
max_results bound, schema rejection of empty queries.
