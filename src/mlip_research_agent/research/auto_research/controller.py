"""The Ralphthon Auto Research controller: a resumable, audited loop.

One `step()` performs exactly one state-machine unit of work, writes its
artifacts, and checkpoints the round state — so interruption at *every* state
is recoverable and resume is deterministic. The controller:

- contains no benchmark-specific branches (execution goes through the typed
  :class:`BenchmarkAdapter`; judgement through the injected evaluator
  boundary whose identity must match ``objective.required_evaluator``);
- never computes metrics itself and never sees per-record test errors;
- never mutates configuration outside the transactional config store;
- never overwrites a produced artifact (append-only trace);
- pauses for Tea Time at the declared boundaries, whose output remains
  agent-text and can satisfy no evidence gate.

It reuses the existing executor-era primitives — ``ArtifactRegistry``,
``EventLog`` — instead of growing parallel ones.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Protocol, TypeVar

from mlip_research_agent.artifacts.registry import Artifact, ArtifactRegistry
from mlip_research_agent.research.auto_research.acceptance_policy import (
    AcceptanceConstraints,
    decide,
)
from mlip_research_agent.research.auto_research.adapters.base import (
    AdapterRunResult,
    BenchmarkAdapter,
)
from mlip_research_agent.research.auto_research.decision import (
    DecisionValue,
    ExperimentDecision,
)
from mlip_research_agent.research.auto_research.evaluation import (
    BaselineReference,
    EvaluationOutcome,
)
from mlip_research_agent.research.auto_research.execution import (
    EventLogRange,
    ExperimentExecution,
    ResourceUsage,
)
from mlip_research_agent.research.auto_research.lesson import ResearchLesson
from mlip_research_agent.research.auto_research.lineage import (
    ExperimentLineage,
    IterationLineage,
    LineageNode,
    LineageNodeKind,
)
from mlip_research_agent.research.auto_research.mutation import (
    LegalityResult,
    Mutation,
    MutationClass,
    MutationDiff,
    MutationPolicy,
    TransactionalConfigStore,
)
from mlip_research_agent.research.auto_research.objective import ResearchObjective
from mlip_research_agent.research.auto_research.proposal import ExperimentProposal
from mlip_research_agent.research.auto_research.proposal_policy import (
    PriorOutcome,
    generate_proposal,
)
from mlip_research_agent.research.auto_research.review import (
    AgentReview,
    ReviewPacket,
    ReviewSynthesis,
    verify_review_packet,
)
from mlip_research_agent.research.auto_research.round_state import (
    TERMINAL_STATES,
    CompletionStatus,
    IterationRecord,
    LoopState,
    RoundState,
)
from mlip_research_agent.research.auto_research.tea_time_boundary import (
    TeaTimeReviewRecord,
    run_tea_time_review,
    select_trigger,
)
from mlip_research_agent.research.auto_research.validators import (
    ConfigValue,
    ContentAddressedModel,
    require_sealed,
)
from mlip_research_agent.runtime.events import EventLog
from mlip_research_agent.schemas.events import EventType
from mlip_research_agent.skills.base import SkillContext

_CAM = TypeVar("_CAM", bound=ContentAddressedModel)


class EvaluatorBoundary(Protocol):
    """The independent judge. Its identity is pinned by the objective."""

    @property
    def name(self) -> str: ...

    def evaluate(
        self,
        *,
        evaluation_id: str,
        execution_id: str,
        predictions_path: Path,
        predictions_rerun_path: Path,
        prediction_artifact_ids: list[str],
        metric_artifact_ids: list[str],
        legality: LegalityResult,
        baseline: BaselineReference,
        resource_metrics: dict[str, float],
        constraints: dict[str, float],
        metrics_out_path: Path,
    ) -> EvaluationOutcome: ...


class ControllerError(Exception):
    """A controller invariant was violated."""


class AutoResearchController:
    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        objective: ResearchObjective,
        mutation_policy: MutationPolicy,
        adapter: BenchmarkAdapter,
        evaluator: EvaluatorBoundary,
        acceptance: AcceptanceConstraints,
        base_config: dict[str, ConfigValue],
        fixture_seed: int,
        git_commit: str,
        external_review_after_iteration: int | None = None,
        initial_proposal: ExperimentProposal | None = None,
    ) -> None:
        require_sealed(objective, "research objective")
        if evaluator.name != objective.required_evaluator:
            raise ControllerError(
                f"evaluator identity {evaluator.name!r} does not match the objective's "
                f"required evaluator {objective.required_evaluator!r}"
            )
        if adapter.remote and objective.budget.max_remote_jobs == 0:
            raise ControllerError("the objective permits no remote jobs but the adapter is remote")
        self.run_id = run_id
        self.run_dir = run_dir
        self.objective = objective
        self.policy = mutation_policy
        self.adapter = adapter
        self.evaluator = evaluator
        self.acceptance = acceptance
        self.base_config = dict(base_config)
        self.fixture_seed = fixture_seed
        self.git_commit = git_commit
        self.external_review_after_iteration = external_review_after_iteration
        self.initial_proposal = initial_proposal
        if initial_proposal is not None:
            require_sealed(initial_proposal, "initial proposal")
            if initial_proposal.objective_id != objective.objective_id:
                raise ControllerError("initial proposal targets a different objective")
            if initial_proposal.parent_iteration_id is not None:
                raise ControllerError("initial proposal cannot have a parent iteration")

        run_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ArtifactRegistry.load(run_dir)
        self.events = EventLog(run_dir, run_id)
        self.store = TransactionalConfigStore(run_dir / "config_store")
        if (run_dir / "round_state.json").is_file():
            self.state = RoundState.load(run_dir)
            if self.state.objective_sha256 != objective.content_sha256:
                raise ControllerError(
                    "resume refused: the objective on disk differs from the one provided"
                )
        else:
            self.state = RoundState(run_id=run_id, objective_sha256=objective.content_sha256)
            self.state.save(run_dir)
            self.events.emit(EventType.RUN_STARTED, payload={"run_id": run_id})

    # ------------------------------------------------------------------ api

    def run(self) -> LoopState:
        """Drive the loop to a terminal state."""
        while self.state.state not in TERMINAL_STATES | {LoopState.AWAITING_EXTERNAL_REVIEW}:
            self.step()
        return self.state.state

    def resume_with_external_review(
        self,
        *,
        reviews: list[AgentReview],
        review_packet: ReviewPacket,
        synthesis: ReviewSynthesis,
        proposal: ExperimentProposal,
    ) -> None:
        """Seal the host review boundary and make iteration 2 runnable."""
        if self.state.state is not LoopState.AWAITING_EXTERNAL_REVIEW:
            raise ControllerError("controller is not awaiting an external review")
        by_role = {review.role: review for review in reviews}
        require_sealed(review_packet, "review packet")
        verify_review_packet(review_packet, self.run_dir)
        if len(by_role) != 3 or len(reviews) != 3:
            raise ControllerError("exactly three unique specialized reviews are required")
        for review in reviews:
            require_sealed(review, f"{review.role} review")
            if review.review_status in {"stop", "escalate"}:
                raise ControllerError(f"review role {review.role} requested {review.review_status}")
            if synthesis.review_hashes.get(review.role) != review.content_sha256:
                raise ControllerError(f"synthesis hash mismatch for {review.role}")
            if review.review_packet_sha256 != review_packet.content_sha256:
                raise ControllerError(f"review packet hash mismatch for {review.role}")
        require_sealed(synthesis, "review synthesis")
        if synthesis.review_packet_sha256 != review_packet.content_sha256:
            raise ControllerError("synthesis review-packet hash mismatch")
        require_sealed(proposal, "external proposal")
        if proposal.parent_iteration_id != "iteration-001":
            raise ControllerError("external proposal must descend from iteration-001")
        synthesis_reference = f"review_synthesis:{synthesis.content_sha256}"
        if synthesis_reference not in proposal.required_skills:
            raise ControllerError("external proposal does not reference the review synthesis")
        review_dir = self.run_dir / "agent_reviews" / "iteration-001"
        review_dir.mkdir(parents=True, exist_ok=True)
        for role, review in sorted(by_role.items()):
            self._write_node(review_dir, f"{role}.json", review, "agent_text")
        self._write_node(review_dir, "review_packet.json", review_packet, "agent_text")
        self._write_node(review_dir, "synthesis.json", synthesis, "agent_text")
        self._write_node(self._iteration_dir(), "proposal.json", proposal, "auto_research")
        self.state.iterations.append(
            IterationRecord(
                iteration_id=self.state.current_iteration_id,
                proposal_id=proposal.proposal_id,
            )
        )
        self.state.transition(LoopState.NEXT_PROPOSAL_READY)
        self._checkpoint()

    def step(self) -> LoopState:
        """Execute exactly one state-machine unit of work and checkpoint."""
        state = self.state.state
        handler = {
            LoopState.INITIALIZED: self._handle_initialized,
            LoopState.PROPOSAL_READY: self._handle_tea_time,
            LoopState.NEXT_PROPOSAL_READY: self._handle_tea_time,
            LoopState.TEA_TIME_REVIEWED: self._handle_legality,
            LoopState.LEGALITY_APPROVED: self._handle_apply_mutation,
            LoopState.MUTATION_APPLIED: self._handle_start_execution,
            LoopState.EXECUTING: self._handle_executing,
            LoopState.EXECUTED: self._handle_post_execution,
            LoopState.EVALUATED: self._handle_decide,
            LoopState.DECIDED: self._handle_post_decision,
            LoopState.ROLLED_BACK: self._handle_lesson,
            LoopState.LESSON_RECORDED: self._handle_iteration_boundary,
        }.get(state)
        if handler is None:
            raise ControllerError(f"no work is possible in terminal state {state.value}")
        self.events.emit(
            EventType.STEP_STARTED,
            step_id=f"{self.state.current_iteration_id}:{state.value}",
        )
        handler()
        self.events.emit(
            EventType.STEP_COMPLETED,
            step_id=f"{self.state.current_iteration_id}:{state.value}",
            payload={"new_state": self.state.state.value},
        )
        self._checkpoint()
        if self.state.state in TERMINAL_STATES:
            self._finalize()
        return self.state.state

    # ------------------------------------------------------------- plumbing

    def _checkpoint(self) -> None:
        self.state.save(self.run_dir)
        self.registry.save()

    def _iteration_dir(self) -> Path:
        d = self.run_dir / self.state.current_iteration_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_node(
        self, directory: Path, filename: str, model: ContentAddressedModel, kind: str
    ) -> Artifact:
        """Write one sealed lineage node.

        Append-only: an existing file with *different* content is a defect. An
        existing file with identical content is a resume re-entry (the handler
        re-ran deterministically after an interruption) and is accepted as-is.
        """
        require_sealed(model, filename)
        path = directory / filename
        if path.exists():
            if path.read_text() != model.canonical_text():
                raise ControllerError(f"trace is append-only; {path} exists with different content")
        else:
            path.write_text(model.canonical_text())
        return self.registry.register(path, kind=kind, step_id=directory.name)

    def _load_node(self, directory: Path, filename: str, model_cls: type[_CAM]) -> _CAM:
        model = model_cls.model_validate_json((directory / filename).read_text())
        require_sealed(model, filename)
        return model

    def _prior_outcome(self) -> PriorOutcome | None:
        if not self.state.iterations:
            return None
        last = self.state.iterations[-1]
        it_dir = self.run_dir / last.iteration_id
        decision = self._load_node(it_dir, "decision.json", ExperimentDecision)
        lesson = self._load_node(it_dir, "lesson.json", ResearchLesson)
        diff_path = it_dir / "mutation_diff.json"
        mutated_keys: list[str] = []
        if diff_path.is_file():
            diff = self._load_node(it_dir, "mutation_diff.json", MutationDiff)
            mutated_keys = [m.target_key for m in diff.changes]
        constraint_failed = (
            decision.decision is DecisionValue.REJECT
            and decision.metric_result.passed
            and not (
                decision.tail_risk.passed
                and decision.compute_budget.passed
                and decision.reproducibility.passed
            )
        )
        return PriorOutcome(
            iteration_id=last.iteration_id,
            decision=decision,
            lesson=lesson,
            constraint_failed=constraint_failed,
            mutated_keys=mutated_keys,
        )

    # ------------------------------------------------------------- handlers

    def _handle_initialized(self) -> None:
        # Run-level artifacts.
        for filename, kind in (
            ("compute_attestation.json", "compute_attestation"),
            ("dataset_verification.json", "dataset_verification"),
            ("split_verification.json", "split_verification"),
            ("checkpoint_verification.json", "checkpoint_verification"),
        ):
            path = self.run_dir / filename
            if path.is_file():
                self.registry.register(path, kind=kind, step_id="run")
        objective_path = self.run_dir / "objective.json"
        if not objective_path.is_file():
            objective_path.write_text(self.objective.canonical_text())
            self.registry.register(objective_path, kind="auto_research", step_id="run")
        policy_path = self.run_dir / "mutation_policy.json"
        if not policy_path.is_file():
            payload = self.policy.model_dump(mode="json")
            payload["content_sha256"] = self.policy.content_hash()
            policy_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            self.registry.register(policy_path, kind="auto_research", step_id="run")
        environment_path = self.run_dir / "environment.json"
        if not environment_path.is_file():
            environment_path.write_text(
                json.dumps(
                    {
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "git_commit": self.git_commit,
                        "adapter": self.adapter.name,
                        "evaluator": self.evaluator.name,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        self.registry.register(environment_path, kind="auto_research", step_id="run")
        freeze_path = self.run_dir / "pip_freeze.txt"
        if freeze_path.is_file():
            self.registry.register(freeze_path, kind="environment_lock", step_id="run")

        self.store.initialize(self.base_config)

        # Baseline execution + evaluation on the accepted base configuration.
        baseline_dir = self.run_dir / "baseline"
        result = self.adapter.execute(self.store.current_config(), self.fixture_seed, baseline_dir)
        if not result.succeeded:
            self.state.partial_reason = "baseline execution failed"
            self.state.transition(LoopState.BLOCKED)
            return
        baseline_metrics = self._evaluate_baseline(baseline_dir, result)
        self.state.baseline_metrics = baseline_metrics

        proposal = self.initial_proposal or generate_proposal(
            objective=self.objective,
            policy=self.policy,
            current_config=self.store.current_config(),
            iteration_index=self.state.iteration_index,
            prior=None,
            seed=self.fixture_seed,
        )
        self._write_node(self._iteration_dir(), "proposal.json", proposal, "auto_research")
        self.state.iterations.append(
            IterationRecord(
                iteration_id=self.state.current_iteration_id,
                proposal_id=proposal.proposal_id,
            )
        )
        self.state.transition(LoopState.PROPOSAL_READY)

    def _evaluate_baseline(self, baseline_dir: Path, result: AdapterRunResult) -> dict[str, float]:
        assert result.predictions_path and result.predictions_rerun_path
        self.registry.register(
            Path(result.fixture_definition_path),
            kind="auto_research",
            step_id="baseline",
        )
        predictions = self.registry.register(
            Path(result.predictions_path), kind="auto_research", step_id="baseline"
        )
        rerun = self.registry.register(
            Path(result.predictions_rerun_path), kind="auto_research", step_id="baseline"
        )
        for path_str in result.additional_artifact_paths:
            self.registry.register(Path(path_str), kind="auto_research", step_id="baseline")
        legality = LegalityResult(
            proposal_id="baseline",
            policy_id=self.policy.policy_id,
            policy_content_sha256=self.policy.content_hash(),
            legal=True,
            per_mutation_class=[],
            violations=[],
        ).sealed()
        outcome = self.evaluator.evaluate(
            evaluation_id="baseline-evaluation",
            execution_id="baseline",
            predictions_path=Path(result.predictions_path),
            predictions_rerun_path=Path(result.predictions_rerun_path),
            prediction_artifact_ids=[predictions.artifact_id, rerun.artifact_id],
            metric_artifact_ids=["baseline:metrics.json"],
            legality=legality,
            baseline=BaselineReference(baseline_id="none", metrics={}),
            resource_metrics={
                "wall_seconds": result.simulated_wall_seconds,
                "peak_memory_mb": result.simulated_peak_memory_mb,
            },
            constraints=self._constraint_limits(),
            metrics_out_path=baseline_dir / "metrics.json",
        )
        self.registry.register(
            baseline_dir / "metrics.json", kind="auto_research", step_id="baseline"
        )
        self._write_node(baseline_dir, "evaluation.json", outcome, "auto_research")
        return dict(outcome.aggregate_metrics)

    def _constraint_limits(self) -> dict[str, float]:
        return {
            "tail_max": self.acceptance.tail_max,
            "runtime_max_seconds": self.acceptance.runtime_max_seconds,
            "memory_max_mb": self.acceptance.memory_max_mb,
        }

    def _handle_tea_time(self) -> None:
        it_dir = self._iteration_dir()
        proposal = self._load_node(it_dir, "proposal.json", ExperimentProposal)
        if proposal.estimated_compute.remote and self.objective.budget.max_remote_jobs == 0:
            self.state.partial_reason = "proposal requires remote compute with no remote budget"
            self.state.transition(LoopState.BLOCKED)
            return
        prior_classes: list[MutationClass] = []
        for record in self.state.iterations[:-1]:
            legality_path = self.run_dir / record.iteration_id / "legality.json"
            if not legality_path.is_file():
                continue
            legality = self._load_node(
                self.run_dir / record.iteration_id,
                "legality.json",
                LegalityResult,
            )
            prior_classes.extend(legality.per_mutation_class)
        trigger = select_trigger(
            is_first_proposal=self.state.iteration_index == 0,
            proposal_is_remote=proposal.estimated_compute.remote,
            prior_mutation_classes=prior_classes,
            consecutive_failures=self.state.consecutive_failures,
            is_pivot_request=False,
            changes_checkpoint_family=False,
            renders_claims=False,
        )
        ctx = SkillContext(
            run_dir=self.run_dir,
            step_id=f"{self.state.current_iteration_id}-tea-time",
            seed=proposal.seed,
            attempt=1,
            registry=self.registry,
        )
        decisions_summary: dict[str, int] = {}
        for record in self.state.iterations[:-1]:
            if record.decision:
                decisions_summary[record.decision] = decisions_summary.get(record.decision, 0) + 1
        review = run_tea_time_review(
            ctx=ctx,
            objective=self.objective,
            proposal=proposal,
            trigger=trigger,
            prior_decisions_summary=decisions_summary,
            failure_category=self.state.last_failure_category,
            remaining_iterations=(self.objective.maximum_iterations - self.state.iteration_index),
            remaining_compute_seconds=(
                self.objective.budget.max_compute_seconds - self.state.compute_seconds_used
            ),
        )
        self._write_node(it_dir, "tea_time.json", review, "reflection")
        self.state.transition(LoopState.TEA_TIME_REVIEWED)

    def _handle_legality(self) -> None:
        it_dir = self._iteration_dir()
        proposal = self._load_node(it_dir, "proposal.json", ExperimentProposal)
        legality = self.policy.check_mutations(
            proposal.proposal_id,
            list(proposal.proposed_mutations),
            self.store.current_config(),
            allowed_classes=list(self.objective.allowed_mutation_classes),
        )
        self._write_node(it_dir, "legality.json", legality, "auto_research")
        if legality.legal:
            self.state.transition(LoopState.LEGALITY_APPROVED)
            return
        decision = decide(
            decision_id=f"{proposal.proposal_id}-decision"[:64],
            proposal_id=proposal.proposal_id,
            legality=legality,
            execution=None,
            evaluation=None,
            constraints=self.acceptance,
            baseline_metrics=self.state.baseline_metrics,
            repair_available=False,
            compute_budget_ok=self._budget_ok(),
        )
        self._write_node(it_dir, "decision.json", decision, "auto_research")
        self.state.iterations[-1].decision = decision.decision.value
        self.state.transition(LoopState.DECIDED)

    def _handle_apply_mutation(self) -> None:
        it_dir = self._iteration_dir()
        proposal = self._load_node(it_dir, "proposal.json", ExperimentProposal)
        if (it_dir / "mutation_diff.json").is_file() and self.store.has_pending_candidate():
            # Resume re-entry after an interruption between apply and transition.
            self.state.transition(LoopState.MUTATION_APPLIED)
            return
        diff = self.store.apply(proposal.proposal_id, list(proposal.proposed_mutations))
        self._write_node(it_dir, "mutation_diff.json", diff, "auto_research")
        self.state.transition(LoopState.MUTATION_APPLIED)

    def _handle_start_execution(self) -> None:
        self.state.transition(LoopState.EXECUTING)

    def _handle_executing(self) -> None:
        it_dir = self._iteration_dir()
        if (it_dir / "execution.json").is_file():
            # Resume after an interruption mid-execution: the completed record
            # exists, never execute twice.
            self.state.transition(LoopState.EXECUTED)
            return
        proposal = self._load_node(it_dir, "proposal.json", ExperimentProposal)
        first_sequence = self.events.emit(
            EventType.STEP_STARTED,
            step_id=f"{self.state.current_iteration_id}:adapter",
        ).sequence
        result = self.adapter.execute(
            self.store.candidate_config(), proposal.seed, it_dir / "workdir"
        )
        last_sequence = self.events.emit(
            EventType.STEP_COMPLETED,
            step_id=f"{self.state.current_iteration_id}:adapter",
            payload={"succeeded": result.succeeded},
        ).sequence

        produced: list[str] = []
        definition = self.registry.register(
            Path(result.fixture_definition_path),
            kind="auto_research",
            step_id=self.state.current_iteration_id,
        )
        produced.append(definition.artifact_id)
        failure_artifact_id: str | None = None
        for path_str in (result.predictions_path, result.predictions_rerun_path):
            if path_str:
                artifact = self.registry.register(
                    Path(path_str),
                    kind="auto_research",
                    step_id=self.state.current_iteration_id,
                )
                produced.append(artifact.artifact_id)
        for path_str in result.additional_artifact_paths:
            artifact = self.registry.register(
                Path(path_str), kind="auto_research", step_id=self.state.current_iteration_id
            )
            produced.append(artifact.artifact_id)
        if result.failure_path:
            artifact = self.registry.register(
                Path(result.failure_path),
                kind="failure",
                step_id=self.state.current_iteration_id,
            )
            failure_artifact_id = artifact.artifact_id
            produced.append(artifact.artifact_id)
        model_artifact_id: str | None = None
        if result.model_path:
            artifact = self.registry.register(
                Path(result.model_path),
                kind="fine_tuned_mace_model",
                step_id=self.state.current_iteration_id,
            )
            model_artifact_id = artifact.artifact_id
            produced.append(artifact.artifact_id)
        split_artifact_id: str | None = None
        if result.split_path:
            artifact = self.registry.register(
                Path(result.split_path),
                kind="split_verification",
                step_id="run",
            )
            split_artifact_id = artifact.artifact_id
        if result.operation_receipt_path:
            artifact = self.registry.register(
                Path(result.operation_receipt_path),
                kind="optimizer_operation_receipt",
                step_id=self.state.current_iteration_id,
            )
            produced.append(artifact.artifact_id)

        config_path = it_dir / "candidate_config.json"
        config_path.write_text(
            json.dumps(self.store.candidate_config(), indent=2, sort_keys=True) + "\n"
        )
        config_artifact = self.registry.register(
            config_path, kind="auto_research", step_id=self.state.current_iteration_id
        )
        execution = ExperimentExecution(
            execution_id=f"{proposal.proposal_id}-exec"[:64],
            proposal_artifact_sha256=proposal.content_sha256,
            git_commit=self.git_commit,
            environment_artifact="run:environment.json",
            config_artifact=config_artifact.artifact_id,
            dataset_artifact=definition.artifact_id,
            split_artifact=split_artifact_id,
            model_artifact=model_artifact_id,
            compute_attestation=result.compute_attestation,
            event_log_range=EventLogRange(
                first_sequence=first_sequence, last_sequence=last_sequence
            ),
            produced_artifacts=produced,
            failure_artifact=failure_artifact_id,
            succeeded=result.succeeded,
            resource_usage=ResourceUsage(
                wall_seconds=result.simulated_wall_seconds,
                peak_memory_mb=result.simulated_peak_memory_mb,
                device=result.device,
            ),
        ).sealed()
        self._write_node(it_dir, "execution.json", execution, "auto_research")
        self.state.compute_seconds_used += result.simulated_wall_seconds
        self.state.last_failure_category = result.failure_category
        self.state.transition(LoopState.EXECUTED)

    def _handle_post_execution(self) -> None:
        it_dir = self._iteration_dir()
        execution = self._load_node(it_dir, "execution.json", ExperimentExecution)
        proposal = self._load_node(it_dir, "proposal.json", ExperimentProposal)
        legality = self._load_node(it_dir, "legality.json", LegalityResult)
        if not execution.succeeded:
            repair_available = False
            if execution.failure_artifact is not None:
                failure_payload = json.loads((it_dir / "workdir" / "failure.json").read_text())
                repair_available = bool(failure_payload.get("repair_available", False))
            decision = decide(
                decision_id=f"{proposal.proposal_id}-decision"[:64],
                proposal_id=proposal.proposal_id,
                legality=legality,
                execution=execution,
                evaluation=None,
                constraints=self.acceptance,
                baseline_metrics=self.state.baseline_metrics,
                repair_available=repair_available,
                compute_budget_ok=self._budget_ok(),
            )
            self._write_node(it_dir, "decision.json", decision, "auto_research")
            self.state.iterations[-1].decision = decision.decision.value
            self.state.transition(LoopState.DECIDED)
            return

        prediction_ids = [a for a in execution.produced_artifacts if "predictions" in a]
        metrics_path = it_dir / "metrics.json"
        outcome = self.evaluator.evaluate(
            evaluation_id=f"{proposal.proposal_id}-eval"[:64],
            execution_id=execution.execution_id,
            predictions_path=it_dir / "workdir" / "predictions.json",
            predictions_rerun_path=it_dir / "workdir" / "predictions_rerun.json",
            prediction_artifact_ids=prediction_ids,
            metric_artifact_ids=[f"{self.state.current_iteration_id}:metrics.json"],
            legality=legality,
            baseline=BaselineReference(
                baseline_id="accepted-config-baseline",
                metrics=dict(self.state.baseline_metrics),
            ),
            resource_metrics={
                "wall_seconds": execution.resource_usage.wall_seconds,
                "peak_memory_mb": execution.resource_usage.peak_memory_mb,
            },
            constraints=self._constraint_limits(),
            metrics_out_path=metrics_path,
        )
        if outcome.evaluator_name != self.objective.required_evaluator:
            raise ControllerError(
                "evaluation was not produced by the objective's required evaluator"
            )
        self.registry.register(
            metrics_path, kind="auto_research", step_id=self.state.current_iteration_id
        )
        self._write_node(it_dir, "evaluation.json", outcome, "auto_research")
        self.state.transition(LoopState.EVALUATED)

    def _handle_decide(self) -> None:
        it_dir = self._iteration_dir()
        proposal = self._load_node(it_dir, "proposal.json", ExperimentProposal)
        legality = self._load_node(it_dir, "legality.json", LegalityResult)
        execution = self._load_node(it_dir, "execution.json", ExperimentExecution)
        evaluation = self._load_node(it_dir, "evaluation.json", EvaluationOutcome)
        decision = decide(
            decision_id=f"{proposal.proposal_id}-decision"[:64],
            proposal_id=proposal.proposal_id,
            legality=legality,
            execution=execution,
            evaluation=evaluation,
            constraints=self.acceptance,
            baseline_metrics=self.state.baseline_metrics,
            repair_available=False,
            compute_budget_ok=self._budget_ok(),
        )
        self._write_node(it_dir, "decision.json", decision, "auto_research")
        self.state.iterations[-1].decision = decision.decision.value
        self.state.transition(LoopState.DECIDED)

    def _budget_ok(self) -> bool:
        return self.state.compute_seconds_used <= self.objective.budget.max_compute_seconds

    def _handle_post_decision(self) -> None:
        it_dir = self._iteration_dir()
        decision = self._load_node(it_dir, "decision.json", ExperimentDecision)
        mutation_applied = (it_dir / "mutation_diff.json").is_file()
        if decision.decision is DecisionValue.ACCEPT:
            if self.store.has_pending_candidate():
                self.store.commit()
            evaluation = self._load_node(it_dir, "evaluation.json", EvaluationOutcome)
            self.state.baseline_metrics = dict(evaluation.aggregate_metrics)
            self.state.iterations[-1].accepted_config_index = self.store.current_index
            self.state.consecutive_failures = 0
            self._record_lesson(it_dir)
            self.state.transition(LoopState.LESSON_RECORDED)
            return
        if decision.decision in {DecisionValue.REFINE, DecisionValue.ESCALATE}:
            self.state.consecutive_failures += 1
        else:
            self.state.consecutive_failures = 0
        if mutation_applied:
            if self.store.has_pending_candidate():
                self.store.rollback()
            rollback_path = it_dir / "rollback.json"
            rollback_path.write_text(
                json.dumps(
                    {
                        "rolled_back": True,
                        "restored_config_index": self.store.current_index,
                        "decision": decision.decision.value,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            self.registry.register(
                rollback_path, kind="auto_research", step_id=self.state.current_iteration_id
            )
            self.state.transition(LoopState.ROLLED_BACK)
            return
        # No mutation was ever applied (illegal proposal): straight to lesson.
        self._record_lesson(it_dir)
        self.state.transition(LoopState.LESSON_RECORDED)

    def _handle_lesson(self) -> None:
        it_dir = self._iteration_dir()
        self._record_lesson(it_dir)
        self.state.transition(LoopState.LESSON_RECORDED)

    def _record_lesson(self, it_dir: Path) -> None:
        proposal = self._load_node(it_dir, "proposal.json", ExperimentProposal)
        decision = self._load_node(it_dir, "decision.json", ExperimentDecision)
        mutation = proposal.proposed_mutations[0]
        evidence = [f"{it_dir.name}:decision.json", f"{it_dir.name}:proposal.json"]
        evaluation_path = it_dir / "evaluation.json"
        observed: str
        if evaluation_path.is_file():
            evaluation = self._load_node(it_dir, "evaluation.json", EvaluationOutcome)
            evidence.append(f"{it_dir.name}:evaluation.json")
            primary = self.acceptance.primary_metric
            observed_value = evaluation.aggregate_metrics.get(primary)
            direction = (
                "decrease"
                if decision.decision is DecisionValue.ACCEPT
                else "no qualifying decrease"
            )
            observed = (
                f"{primary} = {observed_value:.6f} against baseline "
                f"{decision.metric_result.detail}; outcome: {direction}."
            )
        else:
            observed = (
                f"No evaluation was produced; decision {decision.decision.value} "
                f"({decision.rationale})"
            )
        if (it_dir / "execution.json").is_file():
            evidence.append(f"{it_dir.name}:execution.json")

        is_reduction = _is_reduction(mutation)
        lesson = ResearchLesson(
            lesson_id=f"{proposal.proposal_id}-lesson"[:64],
            iteration_id=it_dir.name,
            attempt_summary=(
                f"{'Reducing' if is_reduction else 'Increasing'} {mutation.target_key} from "
                f"{mutation.old_value!r} to {mutation.new_value!r} under hypothesis: "
                f"{proposal.hypothesis}"
            ),
            observed_result=observed,
            evidence_references=evidence,
            likely_explanation=(
                mutation.scientific_effect
                if decision.decision is DecisionValue.ACCEPT
                else "The mutation did not produce the hypothesized effect within bounds; "
                "the bounded response surface or step size is the likely reason."
            ),
            alternative_explanations=[
                "The aggregate metric may mask a smaller configuration-specific effect.",
                "The step size was too large or too small to reveal the mechanism.",
            ],
            limits=(
                "Bounded infrastructure iteration; aggregate validation metrics support no "
                "scientific or checkpoint-selection claim."
            ),
            applicability_conditions=(
                "Applies only to this adapter, exact inputs, and bounded mutation policy."
            ),
            anti_pattern=(
                f"decision={decision.decision.value}: do not repeat an identical "
                f"{mutation.target_key} move without changing the step or direction."
            ),
            recommended_next_mutation_class=MutationClass.BOUNDED_MUTABLE,
        ).sealed()
        self._write_node(it_dir, "lesson.json", lesson, "auto_research")

    def _handle_iteration_boundary(self) -> None:
        it_dir = self._iteration_dir()
        record = self.state.iterations[-1]
        record.completed = True
        self._write_iteration_manifest(it_dir)
        self._write_lineage()

        decision = self._load_node(it_dir, "decision.json", ExperimentDecision)
        prior = self._prior_outcome()
        self.state.iteration_index += 1

        if decision.decision is DecisionValue.ESCALATE:
            self.state.partial_reason = "escalated: owner input required"
            self.state.transition(LoopState.BLOCKED)
            return
        if self.state.iteration_index >= self.objective.maximum_iterations:
            self.state.transition(LoopState.COMPLETE)
            return
        if not self._budget_ok():
            self.state.partial_reason = "compute budget exhausted"
            self.state.transition(LoopState.PARTIAL)
            return

        if self.external_review_after_iteration == self.state.iteration_index:
            self.state.transition(LoopState.AWAITING_EXTERNAL_REVIEW)
            return

        proposal = generate_proposal(
            objective=self.objective,
            policy=self.policy,
            current_config=self.store.current_config(),
            iteration_index=self.state.iteration_index,
            prior=prior,
            seed=self.fixture_seed + self.state.iteration_index,
        )
        self._write_node(self._iteration_dir(), "proposal.json", proposal, "auto_research")
        self.state.iterations.append(
            IterationRecord(
                iteration_id=self.state.current_iteration_id,
                proposal_id=proposal.proposal_id,
            )
        )
        self.state.transition(LoopState.NEXT_PROPOSAL_READY)

    # ------------------------------------------------------- trace assembly

    def _write_iteration_manifest(self, it_dir: Path) -> None:
        prefix = f"{it_dir.name}/"
        entries = [
            a.model_dump(mode="json")
            for a in self.registry.all()
            if a.relative_path.startswith(prefix)
            or a.relative_path.startswith(f"steps/{it_dir.name}-tea-time/")
        ]
        manifest_path = it_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
        self.registry.register(manifest_path, kind="manifest", step_id=it_dir.name)

    def _lineage_nodes_for(self, it_dir: Path) -> list[LineageNode]:
        mapping: list[tuple[str, LineageNodeKind, type[ContentAddressedModel]]] = [
            ("proposal.json", LineageNodeKind.PROPOSAL, ExperimentProposal),
            ("tea_time.json", LineageNodeKind.TEA_TIME, TeaTimeReviewRecord),
            ("legality.json", LineageNodeKind.LEGALITY, LegalityResult),
            ("mutation_diff.json", LineageNodeKind.MUTATION, MutationDiff),
            ("execution.json", LineageNodeKind.EXECUTION, ExperimentExecution),
            ("evaluation.json", LineageNodeKind.EVALUATION, EvaluationOutcome),
            ("decision.json", LineageNodeKind.DECISION, ExperimentDecision),
            ("lesson.json", LineageNodeKind.LESSON, ResearchLesson),
        ]
        nodes: list[LineageNode] = []
        for filename, kind, model_cls in mapping:
            path = it_dir / filename
            if not path.is_file():
                continue
            model = self._load_node(it_dir, filename, model_cls)
            nodes.append(
                LineageNode(
                    kind=kind,
                    node_id=f"{it_dir.name}:{filename}",
                    artifact_id=f"{it_dir.name}:{filename}",
                    content_sha256=model.content_sha256,
                )
            )
        return nodes

    def _write_lineage(self) -> None:
        iterations: list[IterationLineage] = []
        previous: str | None = None
        for record in self.state.iterations:
            it_dir = self.run_dir / record.iteration_id
            if not (it_dir / "lesson.json").is_file():
                continue
            iterations.append(
                IterationLineage(
                    iteration_id=record.iteration_id,
                    parent_iteration_id=previous,
                    nodes=self._lineage_nodes_for(it_dir),
                )
            )
            previous = record.iteration_id
        lineage = ExperimentLineage(
            run_id=self.run_id,
            objective_node=LineageNode(
                kind=LineageNodeKind.OBJECTIVE,
                node_id="run:objective.json",
                artifact_id="run:objective.json",
                content_sha256=self.objective.content_sha256,
            ),
            iterations=iterations,
        ).sealed()
        lineage_path = self.run_dir / "lineage.json"
        lineage_path.write_text(lineage.canonical_text())
        # lineage.json is a growing index over immutable content-addressed
        # nodes and is rewritten at every iteration boundary, so it is not
        # itself registered; the trace grader verifies its nodes directly.

    def _finalize(self) -> None:
        reason = {
            LoopState.COMPLETE: "maximum iterations reached with a complete audited chain",
            LoopState.PARTIAL: self.state.partial_reason or "partial completion",
            LoopState.BLOCKED: self.state.partial_reason or "owner input required",
            LoopState.FAILED: self.state.partial_reason or "controller failure",
        }[self.state.state]
        CompletionStatus(
            run_id=self.run_id,
            status=self.state.state,
            iterations_completed=sum(1 for r in self.state.iterations if r.completed),
            iterations_planned=self.objective.maximum_iterations,
            reason=reason,
            scientific_status=self.objective.scientific_status_ceiling.value,
            claim_eligible=False,
        ).save(self.run_dir)
        self._write_trace_report()
        event = (
            EventType.RUN_COMPLETED
            if self.state.state is LoopState.COMPLETE
            else EventType.RUN_FAILED
        )
        self.events.emit(event, payload={"terminal_state": self.state.state.value})
        self.registry.save()

    def _write_trace_report(self) -> None:
        lines = [
            "# Auto Research trace report",
            "",
            f"- run_id: `{self.run_id}`",
            f"- objective: `{self.objective.objective_id}` — {self.objective.title}",
            f"- terminal state: `{self.state.state.value}`",
            f"- iterations completed: "
            f"{sum(1 for r in self.state.iterations if r.completed)} / "
            f"{self.objective.maximum_iterations}",
            f"- compute used (recorded execution time): {self.state.compute_seconds_used:.1f}s "
            f"of {self.objective.budget.max_compute_seconds:.1f}s",
            "",
            f"> Scientific status: {self.objective.scientific_status_ceiling.value}. "
            "Nothing in this trace is claim-eligible.",
            "",
        ]
        for record in self.state.iterations:
            it_dir = self.run_dir / record.iteration_id
            lines.append(f"## {record.iteration_id}")
            lines.append("")
            lines.append(f"- proposal: `{record.proposal_id}`")
            lines.append(f"- decision: `{record.decision or 'not reached'}`")
            if (it_dir / "evaluation.json").is_file():
                evaluation = self._load_node(it_dir, "evaluation.json", EvaluationOutcome)
                for key, value in sorted(evaluation.aggregate_metrics.items()):
                    lines.append(f"- {key}: {value:.6f}")
            if (it_dir / "rollback.json").is_file():
                lines.append("- rolled back: yes")
            lines.append("")
        report_path = self.run_dir / "trace_report.md"
        report_path.write_text("\n".join(lines))
        self.registry.register(report_path, kind="report", step_id="run")


def _is_reduction(mutation: Mutation) -> bool:
    old, new = mutation.old_value, mutation.new_value
    if isinstance(old, int | float) and isinstance(new, int | float):
        return float(new) < float(old)
    return False
