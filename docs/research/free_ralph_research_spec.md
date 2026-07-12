# Frozen Free Ralph research specification

## Route and question

This is a Ralphthon General Track 1 research-system evaluation. It asks whether
independent semantic re-derivation detects repository-defined faults in an
autonomous MLIP research trace more completely than manifest-only integrity
verification, without rejecting clean traces.

The falsifiable hypothesis is that, across four fixed synthetic seeds and twelve
predeclared fault families, the semantic grader achieves at least 0.90 macro fault
recall, zero clean-trace false positives, and at least 0.25 absolute recall gain over
the baseline. Failure of any threshold supports the null.

## Fair comparison

Both methods inspect the same copied trace after the same single injected fault.
The baseline verifies the registered-file manifest, sizes, and SHA-256 values but
does not re-derive proposal lineage, mutation legality, evaluator independence,
decision consistency, rollback, protected-data policy, or scientific status. The
candidate is the repository's semantic trace grader. A fault operator may update
the manifest when modeling a semantically forged but internally consistent bundle;
that is intentional because the experiment tests what hashes alone cannot prove.

The primary metric is macro recall over the twelve fault families. Secondary
metrics are micro recall over 48 corrupted cases, clean false-positive rate over
four controls, absolute recall gain, violation category, and wall-clock time.

## Frozen cases and budgets

Seeds are `20260712` through `20260715`. Fault families are exactly those in
`configs/research/free_ralph_campaign.yaml`. The experiment uses local CPU only,
no labels, no protected validation queries, no frozen-test data, no acquisition
labels, no W&B sync, and no remote compute. Every case runs once with no retry.

The existing L4 replay at commit `ec2d804fb3b6dafa012620d83436007569de5e7c`
is a motivating infrastructure-only case study. It is not part of the fault-case
denominator and cannot support an MLIP performance claim.

## Failure modes and claim limit

Fault operators are white-box and may resemble the checks they target, so a pass
does not imply general security or universal scientific validity. Synthetic traces
do not establish Cu accuracy, active-learning sample efficiency, or cross-provider
reproducibility. The maximum paper claim is that the tested grader detected the
predeclared repository-defined faults under this protocol while preserving the
specified clean controls.

The preselected fallback is a negative-results paper showing that metric-only
selection would have promoted the second real-MACE candidate although the frozen
full policy rejected it for reproducibility failure.
