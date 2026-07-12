# Limitations

- This is one real-GPU session, one element, one checkpoint, one 32-configuration training set, and one correlated 40-frame validation trajectory.
- The three seeds quantify optimizer variation conditional on the same labels; they are not independent acquisition-policy or data replicates.
- The final-epoch degradation does not establish that small-data transfer is generally harmful, and its mechanism was not isolated.
- The top-36 gate is a post-failure, pre-label protocol amendment, not a universally validated rule.
- No acquisition arm, frozen-test stratum, stress test, physical-property validation, VESSL job, or cross-provider replication has completed.
- The preregistered tail-risk hypothesis remains unanswered.
