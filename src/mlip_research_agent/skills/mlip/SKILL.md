# SKILLs: mock_mlip_training, mock_evaluation

## Purpose
- `mock_mlip_training`: fit a deterministic mean-energy baseline on a seeded
  train split of a label set; the model artifact carries its own split.
- `mock_evaluation`: compute holdout energy MAE and register a backed
  scientific claim (`energy_mae_holdout`).

## Supported use cases
- Exercise train → evaluate → claim registration in the mock campaign.
- Provide the claim fixture that the verification skill audits.

## Non-goals
- Real MLIP fine-tuning (MACE/CHGNet/ORB/... — gated on backend approval).
- Force/stress metrics, calibration, MD-stability evaluation (post-v0).

## Typed inputs
`schema.TrainingInput` (`labels_path`, `model_name`, `holdout_fraction`);
`schema.EvaluationInput` (`model_path`, `labels_path`).

## Typed outputs
`schema.TrainingOutput` (`model_artifact`, `model_path`, split sizes);
`schema.EvaluationOutput` (`metrics_artifact`, `metrics_path`, `energy_mae`,
`n_holdout`).

## Required dependencies
`numpy`; metric math lives in `mlip_research_agent.evaluation.metrics`.

## Preconditions
Label set exists and holds >= 4 labels (train/holdout split needs both sides).

## Validation rules
- Unknown model name → `VALIDATION_ERROR`, non-retryable.
- Missing/short label set → `VALIDATION_ERROR`, non-retryable.

## Failure taxonomy
`VALIDATION_ERROR`; `OVERFITTING` is reserved for real backends.

## Retry and recovery policy
Validation failures escalate immediately (no retry budget consumed by design).

## Produced artifacts
`steps/<step_id>/model.json` (kind `model`),
`steps/<step_id>/metrics.json` (kind `metrics`).

## Provenance fields
Seed, split indices, source label path inside the model artifact; the claim
references the metrics (and model) artifact ids.

## Cost class / permission level
`cheap` / `auto` (real training: `expensive`, budget-gated).

## Minimal example
`examples/example_training_input.json`, `examples/example_evaluation_input.json`.

## Correctness tests
`tests/test_mock_mlip.py`: deterministic split/model bytes, honest holdout
(no train leakage), MAE correctness against a hand-computed value, claim
registration with artifact references, small-label-set rejection.
