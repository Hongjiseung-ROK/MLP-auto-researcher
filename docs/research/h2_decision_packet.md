# Phase 2 RALPHTON Auto-Research: H2 & H3 Authorization Packet

## Executive Summary
This packet requests formal authorization to advance the Auto Research pipeline past the local scaffolding phase and into bounded remote execution for the RALPHTON campaign. We have successfully implemented the core execution loop (Objective → Proposal → Tea Time → Execution → Evaluation → Decision → Lesson), complete with independent testing and explicitly defined boundaries.

## Scientific and Risk Boundaries Verified
1. **Secret Containment**: `api.env` remains strictly isolated and untracked. No agent code touches or parses it.
2. **Data Boundaries**: The Cu dataset is cleanly partitioned. The automated acquisition SKILLs (`random_select`, `ensemble_uq`, `diversity_select`, and `decision_gate`) operate entirely without accessing or depending upon ground-truth test labels or frozen test records.
3. **Immutability and Rollbacks**: The `MutationPolicy` explicitly sets boundaries around what can be altered (hyperparameters only for the demo). Any violations throw an error prior to execution.
4. **Independent Grading**: The `trace_grader.py` and `acceptance_policy.py` correctly enforce constraints (such as blocking a repeating sequence of identical proposals).

## H2 (Preregistration) Authorization Request
We request approval for the initial hypothesis config (`configs/research/ralphthon_demo.yaml`) and mutation policies.
- **Decision required**: `HumanGate.H2_PREREGISTRATION`
- **Owner action**: Please review the mutation policies and the generated preregistration hash to formally lock the experiment definition.

## H3 (Remote Execution) Authorization Request
Once H2 is granted, the auto-research agent will proceed to run its configured 2 iterations on Colab.
- **Decision required**: `HumanGate.H3_REMOTE_EXECUTION`
- **Owner action**: Please explicitly grant the pilot-level Colab resource allocation to execute the accepted proposals on the remote backend.
