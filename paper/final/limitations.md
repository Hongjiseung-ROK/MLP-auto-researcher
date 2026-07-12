# Limitations

- The twelve faults are white-box and repository-defined; perfect detection does not establish general adversarial robustness.
- Four clean controls are insufficient for a precise false-positive confidence bound.
- The four seeds are deterministic trace instantiations, not independent statistical replications.
- Manifest-only verification is a deliberately limited baseline.
- The preregistered `wall_seconds` secondary endpoint was under-specified; host-dependent combined-check timings are retained only as diagnostics and support no runtime claim.
- The synthetic traces evaluate evidence plumbing, not MLIP accuracy or active-learning sample efficiency.
- The real-MACE case study has two one-step iterations on one NVIDIA L4 and a binary exact-equality rerun flag. It does not quantify the physical magnitude of numerical drift.
- H2, H3, and H5 remain open. There is no checkpoint promotion, frozen-test release, publication-level Cu result, or cross-provider replication.
- No new VESSL or Colab compute was used because required approval artifacts were absent.
- Claude Fable reviewers were unavailable because the installed CLI reported a session limit; their review was not simulated.
