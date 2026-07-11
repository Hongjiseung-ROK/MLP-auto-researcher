"""Atomic SKILL wrapper around :class:`SimulatedLabelOracle`."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel

from mlip_research_agent.data.manifests import NormalizedDataset
from mlip_research_agent.data.oracle import (
    BudgetLedger,
    LabelBatch,
    OracleError,
    OracleState,
    RevealDecision,
    RevealRequest,
    SimulatedLabelOracle,
    TrainingLineageManifest,
    write_json_atomic,
)
from mlip_research_agent.data.registry import NormalizedManifest, sha256_file
from mlip_research_agent.data.split import SplitManifest
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.active_learning.oracle_reveal.schema import (
    OracleRevealInput,
    OracleRevealOutput,
)
from mlip_research_agent.skills.active_learning.oracle_reveal.validators import (
    run_relative_file,
)
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)

TRANSACTION_DIR = "reveal_transaction"
REQUEST_FILE = "reveal_request.json"
DECISION_FILE = "reveal_decision.json"
LABEL_BATCH_FILE = "label_batch.json"
BUDGET_LEDGER_FILE = "budget_ledger.json"
TRAINING_LINEAGE_FILE = "training_lineage.json"
ORACLE_STATE_FILE = "oracle_state.json"


@register_skill
class OracleRevealSkill(Skill):
    name = "oracle_reveal"
    input_model = OracleRevealInput
    output_model = OracleRevealOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, OracleRevealInput)
        dataset_path = run_relative_file(ctx.run_dir, params.dataset_path, "qualified dataset")
        qualified_manifest_path = run_relative_file(
            ctx.run_dir, params.qualified_manifest_path, "qualified manifest"
        )
        split_path = run_relative_file(ctx.run_dir, params.split_manifest_path, "split manifest")
        previous_state = (
            run_relative_file(ctx.run_dir, params.previous_state_path, "previous oracle state")
            if params.previous_state_path is not None
            else None
        )
        try:
            dataset = NormalizedDataset.load(dataset_path)
            qualified_manifest = NormalizedManifest.load(qualified_manifest_path)
            split = SplitManifest.load(split_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SkillError(
                f"oracle input artifact is malformed: {exc}",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            ) from exc

        actual_split_sha = sha256_file(split_path)
        actual_qualified_sha = sha256_file(qualified_manifest_path)
        derived_manifest = NormalizedManifest.from_dataset(dataset, sha256_file(dataset_path))
        try:
            if actual_split_sha != params.expected_split_manifest_sha256:
                raise ValueError("split manifest does not match the pinned expected SHA-256")
            if split.qualified_manifest_sha256 != actual_qualified_sha:
                raise ValueError("split provenance does not match the qualified manifest bytes")
            if derived_manifest != qualified_manifest:
                raise ValueError("qualified dataset and manifest lineage do not match")
            split.validate_against(qualified_manifest)
        except ValueError as exc:
            raise SkillError(
                f"oracle provenance validation failed: {exc}",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.CRITICAL,
                retryable=False,
            ) from exc

        final_dir = ctx.step_dir / TRANSACTION_DIR
        if final_dir.is_dir():
            return self._resume_existing(final_dir, params, ctx, dataset, split)
        allow_new_campaign = self._validate_campaign_head(params, previous_state, ctx)
        temporary_dir = Path(tempfile.mkdtemp(prefix=".reveal-", dir=ctx.step_dir))
        try:
            oracle = SimulatedLabelOracle(
                dataset,
                split,
                state_path=temporary_dir / ORACLE_STATE_FILE,
                campaign_id=params.campaign_id,
                total_budget=params.total_budget,
                round_budgets=params.round_budgets,
                event_reference=f"{ctx.run_dir.name}:{ctx.step_id}:STEP_COMPLETED",
                previous_state_path=previous_state,
                allow_new_campaign=allow_new_campaign,
            )
            result = oracle.reveal_result(
                params.candidate_ids, params.campaign_id, params.round_id
            )
            # The public oracle keeps a persistent advisory-lock inode so a
            # stale file is harmless after process death.  This transaction is
            # immutable after reveal, so the lock file is not part of its six
            # evidence artifacts.
            (temporary_dir / f".{ORACLE_STATE_FILE}.lock").unlink(missing_ok=True)
            write_json_atomic(temporary_dir / REQUEST_FILE, result.request.model_dump(mode="json"))
            write_json_atomic(
                temporary_dir / DECISION_FILE, result.decision.model_dump(mode="json")
            )
            write_json_atomic(
                temporary_dir / LABEL_BATCH_FILE, result.label_batch.model_dump(mode="json")
            )
            write_json_atomic(
                temporary_dir / BUDGET_LEDGER_FILE, result.ledger.model_dump(mode="json")
            )
            write_json_atomic(
                temporary_dir / TRAINING_LINEAGE_FILE, result.lineage.model_dump(mode="json")
            )
            os.replace(temporary_dir, final_dir)
        except OracleError as exc:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise SkillError(
                str(exc),
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
                likely_causes=[
                    "protected-partition access",
                    "immutable request conflict",
                    "budget exceeded",
                ],
            ) from exc
        except BaseException:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
        return self._register_transaction(final_dir, ctx, idempotent=result.idempotent_replay)

    def _resume_existing(
        self,
        final_dir: Path,
        params: OracleRevealInput,
        ctx: SkillContext,
        dataset: NormalizedDataset,
        split: SplitManifest,
    ) -> OracleRevealOutput:
        expected = {
            REQUEST_FILE,
            DECISION_FILE,
            LABEL_BATCH_FILE,
            BUDGET_LEDGER_FILE,
            TRAINING_LINEAGE_FILE,
            ORACLE_STATE_FILE,
        }
        present = {path.name for path in final_dir.iterdir() if path.is_file()}
        if present != expected:
            raise SkillError(
                "incomplete oracle transaction: "
                f"expected {sorted(expected)}, found {sorted(present)}",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.CRITICAL,
                retryable=False,
            )
        state = OracleState.load(final_dir / ORACLE_STATE_FILE)
        if state.dataset_content_sha256 != dataset.content_hash():
            raise SkillError(
                "existing oracle transaction conflicts with the current dataset",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.CRITICAL,
                retryable=False,
            )
        if state.split_semantic_sha256 != split.semantic_hash():
            raise SkillError(
                "existing oracle transaction conflicts with the current split",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.CRITICAL,
                retryable=False,
            )
        request = state.requests_by_round.get(params.round_id)
        if request is None or request.campaign_id != params.campaign_id:
            raise SkillError(
                "existing oracle transaction belongs to another campaign or round",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )
        if request.candidate_ids != sorted(params.candidate_ids):
            raise SkillError(
                "existing oracle transaction conflicts with the repeated request",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )
        if state.ledger.total_budget != params.total_budget:
            raise SkillError(
                "existing oracle transaction conflicts with the repeated campaign budget",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )
        if state.ledger.round_budgets != dict(sorted(params.round_budgets.items())):
            raise SkillError(
                "existing oracle transaction conflicts with the repeated round budgets",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )
        try:
            request_file = RevealRequest.model_validate_json(
                (final_dir / REQUEST_FILE).read_text()
            )
            decision_file = RevealDecision.model_validate_json(
                (final_dir / DECISION_FILE).read_text()
            )
            batch_file = LabelBatch.model_validate_json(
                (final_dir / LABEL_BATCH_FILE).read_text()
            )
            ledger_file = BudgetLedger.model_validate_json(
                (final_dir / BUDGET_LEDGER_FILE).read_text()
            )
            lineage_file = TrainingLineageManifest.model_validate_json(
                (final_dir / TRAINING_LINEAGE_FILE).read_text()
            )
            if not (
                request_file == request
                and decision_file == state.decisions_by_round[params.round_id]
                and batch_file == state.batches_by_round[params.round_id]
                and ledger_file == state.ledger
                and lineage_file == state.lineage
            ):
                raise ValueError("standalone oracle artifacts do not match committed state")
        except ValueError as exc:
            raise SkillError(
                f"existing oracle transaction failed integrity validation: {exc}",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.CRITICAL,
                retryable=False,
            ) from exc
        return self._register_transaction(final_dir, ctx, idempotent=True)

    def _validate_campaign_head(
        self,
        params: OracleRevealInput,
        previous_state: Path | None,
        ctx: SkillContext,
    ) -> bool:
        campaign_states: list[tuple[Path, OracleState]] = []
        for artifact in ctx.registry.all():
            if artifact.kind != "oracle_state":
                continue
            if not ctx.registry.verify(artifact.artifact_id):
                raise SkillError(
                    f"registered campaign state failed artifact verification: "
                    f"{artifact.artifact_id}",
                    failure_class=FailureClass.VALIDATION_ERROR,
                    severity=Severity.CRITICAL,
                    retryable=False,
                )
            path = ctx.run_dir / artifact.relative_path
            state = OracleState.load(path)
            if state.campaign_id == params.campaign_id:
                campaign_states.append((path.resolve(), state))
        if not campaign_states:
            if previous_state is not None:
                raise SkillError(
                    "previous oracle state is not a registered state for this campaign",
                    failure_class=FailureClass.VALIDATION_ERROR,
                    severity=Severity.CRITICAL,
                    retryable=False,
                )
            return True
        if previous_state is None:
            raise SkillError(
                "campaign already has oracle state; continuation must supply its registered head",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.CRITICAL,
                retryable=False,
            )
        max_history = max(len(state.lineage.reveal_history) for _, state in campaign_states)
        heads = [
            path
            for path, state in campaign_states
            if len(state.lineage.reveal_history) == max_history
        ]
        if len(heads) != 1 or previous_state.resolve() != heads[0]:
            raise SkillError(
                "previous oracle state is not the unique registered campaign head",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.CRITICAL,
                retryable=False,
            )
        return False

    def _register_transaction(
        self, final_dir: Path, ctx: SkillContext, *, idempotent: bool
    ) -> OracleRevealOutput:
        kinds = {
            REQUEST_FILE: "oracle_reveal_request",
            DECISION_FILE: "oracle_reveal_decision",
            LABEL_BATCH_FILE: "revealed_label_batch",
            BUDGET_LEDGER_FILE: "label_budget_ledger",
            TRAINING_LINEAGE_FILE: "training_lineage_manifest",
            ORACLE_STATE_FILE: "oracle_state",
        }
        artifacts = {
            name: ctx.registry.register(final_dir / name, kind=kind, step_id=ctx.step_id)
            for name, kind in kinds.items()
        }
        state = OracleState.load(final_dir / ORACLE_STATE_FILE)
        request = json.loads((final_dir / REQUEST_FILE).read_text())
        decision = json.loads((final_dir / DECISION_FILE).read_text())
        return OracleRevealOutput(
            request_artifact=artifacts[REQUEST_FILE].artifact_id,
            request_path=artifacts[REQUEST_FILE].relative_path,
            decision_artifact=artifacts[DECISION_FILE].artifact_id,
            decision_path=artifacts[DECISION_FILE].relative_path,
            label_batch_artifact=artifacts[LABEL_BATCH_FILE].artifact_id,
            label_batch_path=artifacts[LABEL_BATCH_FILE].relative_path,
            budget_ledger_artifact=artifacts[BUDGET_LEDGER_FILE].artifact_id,
            budget_ledger_path=artifacts[BUDGET_LEDGER_FILE].relative_path,
            training_lineage_artifact=artifacts[TRAINING_LINEAGE_FILE].artifact_id,
            training_lineage_path=artifacts[TRAINING_LINEAGE_FILE].relative_path,
            oracle_state_artifact=artifacts[ORACLE_STATE_FILE].artifact_id,
            oracle_state_path=artifacts[ORACLE_STATE_FILE].relative_path,
            n_selected=len(request["candidate_ids"]),
            n_newly_charged=len(decision["charged_record_ids"]),
            remaining_budget=state.ledger.remaining_budget,
            idempotent_replay=idempotent,
        )
