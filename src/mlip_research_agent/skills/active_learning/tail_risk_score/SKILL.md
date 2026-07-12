# Force-tail proxy score

## Purpose

Rank an unlabeled configuration pool using only real force predictions from an
identified MACE committee. For atom `i` in configuration `c`, the skill computes
the committee-mean force magnitude plus population RMS committee disagreement.
The configuration score is the mean of the largest preregistered fraction of
atomic values.

## Scientific boundary

The score is a label-free acquisition proxy. It is not observed force error and
is not calibrated uncertainty. Oracle energies, forces, stresses, error metrics,
and test labels are forbidden from the input. Invalid member predictions exclude
the candidate rather than silently reducing the ensemble.

## Formula

For `M` committee members,

`u_ci = sqrt(M^-1 sum_m ||F_mci - mean_m(F_mci)||_2^2)`

and `z_ci = w_F ||mean_m(F_mci)||_2 + w_u u_ci`. The candidate score is the
mean of the largest `max(1, ceil(tail_fraction * N_c))` values of `z_ci`.
The population convention is fixed at `ddof=0`.

## Determinism and provenance

Candidates are ranked by descending score with lexicographic candidate-id ties.
The artifact records committee identities, provenance hashes, weights, tail
fraction, invalid-member flags, atomic summary statistics, and policy version.

## Cost class / permission level

`cheap` / `auto`. Upstream prediction and label reveal retain their own compute
and approval gates.
