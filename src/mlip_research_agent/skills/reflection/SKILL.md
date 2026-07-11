# SKILL: tea_time_with_reading_poem

## Purpose
A bounded, deterministic creative break for the research agent: pair the
current research question with a public-domain poem, a seeded set of
bias-resetting reframing techniques, and alternative numerical views of the
same data — so the next reasoning pass approaches the same boring data with
fresh eyes, wider framing, and fewer inherited assumptions.

## Supported use cases
- Mid-campaign perspective reset when iterations converge on one framing.
- Generating provocation prompts for the (future) LLM planner to consume.
- Re-viewing a labels/metrics artifact through non-habitual summaries
  (median-vs-mean, outlier-first, rank order) without recomputing science.

## Non-goals (read these twice)
- **Not evidence.** Output is agent-text class under the provenance policy:
  the skill registers `reflection` artifacts but never claims, and nothing it
  produces may back a numerical claim, table, or figure.
- Not a metric source, not an evaluation, never inserted into claim chains;
  `claim_verification` ignores `reflection` artifacts.
- Not an LLM call: v0 is deterministic; the poem and techniques are fixtures.

## Typed inputs — `schema.TeaTimeInput`
`focus_question` (5–500 chars), optional `data_path` (run-dir-relative JSON,
labels- or metrics-shaped), `n_provocations` (1–7), optional `poem_key`.

## Typed outputs — `schema.TeaTimeOutput`
`report_artifact`/`report_path` (`tea_time_report.md`),
`reframings_artifact`/`reframings_path` (`reframings.json`), `poem_key`,
`techniques`, `n_provocations`.

## Required dependencies
`numpy` (seeded rng + alternative views). Poem corpus is in-repo
(`poems.py`, all public domain).

## Preconditions
Writable `ctx.step_dir`; if `data_path` is given it must exist under the run
directory and parse as a JSON object.

## Validation rules (`validators.py`)
- Unknown `poem_key` → `VALIDATION_ERROR`, non-retryable.
- Missing/non-JSON/non-object data artifact → `VALIDATION_ERROR`.
- Post-condition: zero claims registered, enforced in-code
  (`UNSUPPORTED_CLAIM` if violated) and by tests.

## Failure taxonomy
`VALIDATION_ERROR`, `UNSUPPORTED_CLAIM` (invariant guard; unreachable in
correct code).

## Retry and recovery policy
All failures are non-retryable input problems; the executor escalates.

## Produced artifacts
`steps/<step_id>/reframings.json` and `steps/<step_id>/tea_time_report.md`,
both kind `reflection`, byte-deterministic for a fixed seed and inputs.

## Provenance fields
Seed (via executor), poem key, technique ids, `evidence_class: agent_text`
embedded in the JSON artifact.

## Cost class / permission level
`trivial` / `auto`.

## Minimal example
`examples/example_input.json`.

## Correctness tests
`tests/test_tea_time.py`: byte-determinism for equal seeds, seed sensitivity,
alternative-view numerics vs numpy, zero-claims invariant, unknown poem and
missing data rejection, n_provocations bounds.
