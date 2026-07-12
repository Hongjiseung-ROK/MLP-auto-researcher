# Feasibility matrix

| Candidate | Data boundary | Compute | Existing evidence | Feasibility |
|---|---|---|---|---|
| Fault-injected trace audit | Synthetic/public trace metadata only | Local CPU | Trace grader, synthetic traces, verified real-MACE infrastructure replay | High |
| Metric improvement is insufficient | Frozen aggregate decisions only | Local CPU | Two rejected L4 iterations | High |
| Decision-aware rerun stability | New public/synthetic MACE predictions | Local CPU plus optional GPU | One L4 replay, exact-equality metric | Medium; no new GPU authorization |
| Validation-query budgets | Synthetic response surfaces | Local CPU | Controller and acceptance contracts | High |
| Disagreement-diversity stability | Label-free synthetic descriptors/predictions | Local CPU | WP6 selectors | High, but chemistry claim ceiling is low |
| Exactly-once crash injection | Synthetic operation receipts | Local CPU | Receipt and VESSL replay contracts | High |
| E0-aware ranking | Aggregate diagnostics or synthetic shifts | Local CPU | Candidate checkpoint diagnostics | Medium; H2 remains open |
| World-model acquisition | Requires action-transition outcomes | GPU/labels likely | None | Rejected |
