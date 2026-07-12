# Frozen protocol: Cu force-tail acquisition under a tiny label budget

Campaign: `cu-tail-risk-ralph-20260712`. The machine-readable source of truth is
`artifacts/research_spec/novel_mlip_ralph_spec.json`; this document explains it.

## Question and scope

The experiment asks whether a label-free score combining predicted force
magnitude and committee disagreement selects configurations that reduce P95
force-vector error and systematic force softening better than conventional
disagreement plus descriptor diversity. The claim is deliberately conditional
on the named mlearn Cu split. Each held-out stratum is one trajectory, so frames,
atoms, force components, and committee members are not policy replicates.

The confirmatory contrast is `tail_risk_fps - disagreement_fps`. The scientific
fallback is the established same-rule comparison between top disagreement and
top-36 disagreement followed by deterministic FPS. Random, zero-shot, and
static-initial models are contextual controls.

## Outcome-blind data isolation

The split uses complete `group_id` units. Initial labels contain 32 configurations;
the acquisition pool contains 141; AIMD-300 K is the 40-frame validation set;
AIMD-3000 K and vacancy-300 K form two separately reported frozen-test strata;
elastic mode 6 is a protected 20-configuration physical stress test. Pool labels
are revealed only after a selection artifact is frozen. Validation aggregates may
guide Ralph proposals, but frozen-test and stress-test labels are evaluated once,
after all final model and analysis hashes are frozen.

## End-to-end policies

All four active arms receive 12 labels in each of three rounds. `disagreement`
takes the top 12 remaining committee scores. `disagreement_fps` constructs the
top-36 disagreement window and runs deterministic FPS. `tail_risk_fps` replaces
the window score by the mean upper 5% atomic value of
`||mean committee force|| + population RMS vector disagreement`, then uses the
same FPS. Random selection uses three frozen round seeds. From round 2 onward,
the policies have different labeled sets and committees; the comparison is
therefore between complete adaptive policies using the same rule, not identical
candidate windows.

Descriptors are invariant node features from the pinned foundation MACE,
mean-pooled over atoms, standardized on the current remaining pool with
population variance, constant dimensions removed, and L2-normalized. FPS uses
Euclidean distance, a centroid-farthest seed, and candidate ID for every tie.

## Training and evaluation

Every fit restarts from the same MACE-MP-0-small checkpoint. Only the last
interaction block and readouts are trainable. Each fit runs 20 complete epochs
with AdamW, learning rate `5e-4`, energy/force weights `1/100`, batch size 4,
float32, and gradient clipping 10. The only outcome-blind fallback is `2.5e-4`
if D0 training becomes non-finite before any pool reveal; if invoked it applies
uniformly to the entire campaign.

Primary metrics are seed-specific P95 atomic force-vector errors, computed with
NumPy's linear quantile separately on the two final strata, then averaged over
seeds. Secondary metrics include energy and force aggregate errors, softening
slope, disagreement/error association, group coverage, learning curves, runtime,
memory, and failures. Elastic mode 6 supplies a protected relative energy-strain
curve and curvature diagnostic; it is not treated as an experimental elastic
constant.

The paired block-resampling sensitivity analysis resamples four existing
trajectory blocks with replacement, applies identical draws to every arm and
seed, and repeats 20,000 times. Its interval is conditional on these trajectories,
not a population confidence interval. The joint superiority rule must pass the
minimum-relevant-effect, paired-seed direction, and one-sided resampling bound in
both strata. Failure is reported as a negative or inconclusive result.

## Ralph and compute sequence

Colab performs numerical calibration, D0 baseline training, and later an
independent cross-provider confirmation. Three sequential VESSL A100 Jobs run
rounds 1–3; each is reviewed on the host before the next proposal freezes. A
fourth VESSL Job performs the single frozen final evaluation. Jobs terminate
before host review. The plan is at most 3 Colab GPU-hours and 6 VESSL GPU-hours;
hard caps remain 6, 9.67, and 22 hours respectively, and VESSL spend must remain
within the verified promotional credit.

Tea Time occurs before every oracle reveal, cloud submission, and frozen-test
authorization. It records reflection only and cannot grant approval. Rejected
models remain immutable evidence and never overwrite an accepted lineage head.
