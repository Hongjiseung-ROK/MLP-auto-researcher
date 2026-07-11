"""Hidden-label acquisition view and atomic simulated label oracle.

The acquisition controller receives :class:`AcquisitionView` only.  The full
qualified dataset is held behind :class:`SimulatedLabelOracle`, which validates
the complete request and all budgets before committing one atomic state file.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.data.manifests import (
    ENERGY_UNIT,
    FORCE_UNIT,
    STRESS_UNIT,
    LabeledConfiguration,
    NormalizedDataset,
)
from mlip_research_agent.data.registry import NormalizedManifest, sha256_file
from mlip_research_agent.data.split import (
    GroupingField,
    PartitionName,
    SplitManifest,
    derive_structural_fingerprint_clusters,
)

ORACLE_SCHEMA_VERSION = "2.0.0"


class OracleError(ValueError):
    """Base class for fail-closed oracle policy violations."""


class OracleAccessError(OracleError):
    """A request includes a protected, unknown, or otherwise ineligible id."""


class OracleBudgetError(OracleError):
    """A reveal would exceed a declared campaign or round budget."""


class OracleConflictError(OracleError):
    """An immutable campaign/round request was replayed with different ids."""


class AcquisitionMetadata(BaseModel):
    """Explicit non-label metadata allowlist available to selectors."""

    model_config = ConfigDict(extra="forbid")

    top_group: str
    group_id: str
    n_atoms: int = Field(gt=0)
    structure_fingerprint_sha256: str = Field(min_length=64, max_length=64)


class AcquisitionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    symbols: list[str] = Field(min_length=1)
    positions: list[list[float]]
    cell: list[list[float]]
    pbc: list[bool]
    metadata: AcquisitionMetadata


class AcquisitionView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ORACLE_SCHEMA_VERSION
    dataset_id: str
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    partition: str = PartitionName.ACQUISITION_POOL.value
    candidates: list[AcquisitionCandidate]

    def save(self, path: Path) -> str:
        assert_no_hidden_labels(self.model_dump(mode="json"))
        return write_json_atomic(path, self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: Path) -> AcquisitionView:
        view = cls.model_validate(json.loads(path.read_text()))
        assert_no_hidden_labels(view.model_dump(mode="json"))
        return view


class RevealRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ORACLE_SCHEMA_VERSION
    request_id: str = Field(min_length=64, max_length=64)
    campaign_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)
    candidate_ids: list[str] = Field(min_length=1)
    dataset_id: str
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    event_reference: str

    @model_validator(mode="after")
    def _canonical_ids(self) -> RevealRequest:
        if self.candidate_ids != sorted(set(self.candidate_ids)):
            raise ValueError("reveal candidate ids must be sorted and unique")
        return self


class RevealDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    approved: bool = True
    charged_record_ids: list[str]
    previously_revealed_record_ids: list[str]
    campaign_budget_before: int = Field(ge=0)
    campaign_budget_after: int = Field(ge=0)
    round_budget_limit: int = Field(ge=0)
    round_budget_after: int = Field(ge=0)
    reason: str
    event_reference: str


class RevealedLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    energy_ev: float
    forces_ev_per_a: list[list[float]]
    virial_stress_kbar: list[float] | None
    energy_unit: str = ENERGY_UNIT
    force_unit: str = FORCE_UNIT
    stress_unit: str = STRESS_UNIT
    level_of_theory: str
    source_dataset_id: str
    source_id: str
    source_record_sha256: str = Field(min_length=64, max_length=64)
    normalized_record_sha256: str = Field(min_length=64, max_length=64)


class LabelBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ORACLE_SCHEMA_VERSION
    batch_id: str = Field(min_length=64, max_length=64)
    request_id: str
    campaign_id: str
    round_id: str
    source_dataset_id: str
    source_dataset_content_sha256: str
    split_semantic_sha256: str
    records: list[RevealedLabel]


class BudgetLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ORACLE_SCHEMA_VERSION
    campaign_id: str
    total_budget: int = Field(ge=0)
    charged_total: int = Field(ge=0)
    remaining_budget: int = Field(ge=0)
    round_budgets: dict[str, int]
    round_charged: dict[str, int]
    revealed_record_ids: list[str]
    request_id_by_round: dict[str, str]
    prior_ledger_sha256: str | None = None

    @model_validator(mode="after")
    def _budget_consistency(self) -> BudgetLedger:
        if self.charged_total + self.remaining_budget != self.total_budget:
            raise ValueError("campaign budget ledger is inconsistent")
        if self.charged_total != len(self.revealed_record_ids):
            raise ValueError("charged_total must equal unique revealed records")
        if any(value < 0 for value in self.round_budgets.values()):
            raise ValueError("round budgets must be non-negative")
        if any(value < 0 for value in self.round_charged.values()):
            raise ValueError("round charges must be non-negative")
        if self.revealed_record_ids != sorted(set(self.revealed_record_ids)):
            raise ValueError("revealed record ids must be sorted and unique")
        if sum(self.round_charged.values()) != self.charged_total:
            raise ValueError("round charges must reconcile to the campaign charge")
        return self


class TrainingLineageEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    round_id: str
    request_id: str
    label_batch_id: str
    selected_record_ids: list[str]
    newly_appended_record_ids: list[str]


class TrainingLineageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ORACLE_SCHEMA_VERSION
    campaign_id: str
    dataset_id: str
    dataset_content_sha256: str
    split_semantic_sha256: str
    initial_record_ids: list[str]
    reveal_history: list[TrainingLineageEntry]
    training_record_ids: list[str]
    prior_lineage_sha256: str | None = None

    @model_validator(mode="after")
    def _lineage_ids(self) -> TrainingLineageManifest:
        if self.initial_record_ids != sorted(set(self.initial_record_ids)):
            raise ValueError("initial lineage ids must be sorted and unique")
        if self.training_record_ids != sorted(set(self.training_record_ids)):
            raise ValueError("training lineage ids must be sorted and unique")
        if not set(self.initial_record_ids).issubset(self.training_record_ids):
            raise ValueError("training lineage must retain every initial record")
        return self


class OracleState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ORACLE_SCHEMA_VERSION
    campaign_id: str
    dataset_id: str
    dataset_content_sha256: str
    split_semantic_sha256: str
    ledger: BudgetLedger
    lineage: TrainingLineageManifest
    requests_by_round: dict[str, RevealRequest]
    decisions_by_round: dict[str, RevealDecision]
    batches_by_round: dict[str, LabelBatch]

    @model_validator(mode="after")
    def _cross_invariants(self) -> OracleState:
        rounds = set(self.requests_by_round)
        if rounds != set(self.decisions_by_round) or rounds != set(self.batches_by_round):
            raise ValueError("oracle request/decision/batch round keys must match")
        if rounds != set(self.ledger.request_id_by_round):
            raise ValueError("budget ledger request rounds must match oracle state")
        for round_id in rounds:
            request = self.requests_by_round[round_id]
            decision = self.decisions_by_round[round_id]
            batch = self.batches_by_round[round_id]
            if request.round_id != round_id or batch.round_id != round_id:
                raise ValueError("oracle state round key does not match embedded round id")
            if not (
                request.request_id
                == decision.request_id
                == batch.request_id
                == self.ledger.request_id_by_round[round_id]
            ):
                raise ValueError("oracle request ids do not reconcile")
            if [record.record_id for record in batch.records] != request.candidate_ids:
                raise ValueError("label batch records do not match the reveal request")
        history_by_round = {entry.round_id: entry for entry in self.lineage.reveal_history}
        if set(history_by_round) != rounds:
            raise ValueError("training lineage history does not match reveal rounds")
        for round_id, entry in history_by_round.items():
            if entry.request_id != self.requests_by_round[round_id].request_id:
                raise ValueError("training lineage request id mismatch")
            if entry.label_batch_id != self.batches_by_round[round_id].batch_id:
                raise ValueError("training lineage label batch id mismatch")
        expected_training = sorted(
            set(self.lineage.initial_record_ids).union(self.ledger.revealed_record_ids)
        )
        if self.lineage.training_record_ids != expected_training:
            raise ValueError("training lineage does not match initial plus revealed ids")
        return self

    @classmethod
    def load(cls, path: Path) -> OracleState:
        return cls.model_validate(json.loads(path.read_text()))


class OracleRevealResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: RevealRequest
    decision: RevealDecision
    label_batch: LabelBatch
    ledger: BudgetLedger
    lineage: TrainingLineageManifest
    state: OracleState
    idempotent_replay: bool = False


class LabelOracle(Protocol):
    def reveal(
        self, candidate_ids: list[str], campaign_id: str, round_id: str
    ) -> LabelBatch: ...


_FORBIDDEN_ACQUISITION_KEYS = {
    "energy_ev",
    "forces_ev_per_a",
    "virial_stress_kbar",
    "source_record_sha256",
    "normalized_record_sha256",
    "source_id",
    "level_of_theory",
}


def assert_no_hidden_labels(payload: object) -> None:
    """Recursively reject label-bearing keys in serialized acquisition data."""
    if isinstance(payload, dict):
        present = _FORBIDDEN_ACQUISITION_KEYS.intersection(payload)
        if present:
            raise OracleAccessError(
                f"acquisition artifact exposes forbidden label fields: {sorted(present)}"
            )
        for value in payload.values():
            assert_no_hidden_labels(value)
    elif isinstance(payload, list):
        for value in payload:
            assert_no_hidden_labels(value)


def _canonical_hash(model: BaseModel) -> str:
    canonical = json.dumps(model.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _structure_fingerprint(record: LabeledConfiguration) -> str:
    payload = {
        "symbols": record.symbols,
        "positions": record.positions,
        "cell": record.cell,
        "pbc": record.pbc,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_acquisition_view(
    dataset: NormalizedDataset, split: SplitManifest
) -> AcquisitionView:
    if dataset.dataset_id != split.dataset_id:
        raise OracleAccessError("dataset id does not match split")
    if dataset.content_hash() != split.qualified_dataset_content_sha256:
        raise OracleAccessError("dataset content hash does not match split provenance")
    derived_manifest = NormalizedManifest.from_dataset(dataset, "0" * 64)
    structural_fingerprints = (
        derive_structural_fingerprint_clusters(dataset)
        if split.grouping_fields_used == [GroupingField.STRUCTURAL_FINGERPRINT.value]
        else None
    )
    split.validate_against(
        derived_manifest, structural_fingerprints=structural_fingerprints
    )
    records = dataset.by_id()
    pool_ids = split.record_ids[PartitionName.ACQUISITION_POOL.value]
    unknown = sorted(set(pool_ids) - set(records))
    if unknown:
        raise OracleAccessError(f"split acquisition pool contains unknown ids: {unknown[:5]}")
    candidates = [
        AcquisitionCandidate(
            record_id=record_id,
            symbols=records[record_id].symbols,
            positions=records[record_id].positions,
            cell=records[record_id].cell,
            pbc=records[record_id].pbc,
            metadata=AcquisitionMetadata(
                top_group=records[record_id].top_group,
                group_id=records[record_id].group_id,
                n_atoms=records[record_id].n_atoms,
                structure_fingerprint_sha256=_structure_fingerprint(records[record_id]),
            ),
        )
        for record_id in sorted(pool_ids)
    ]
    view = AcquisitionView(
        dataset_id=dataset.dataset_id,
        split_semantic_sha256=split.semantic_hash(),
        candidates=candidates,
    )
    assert_no_hidden_labels(view.model_dump(mode="json"))
    return view


def write_json_atomic(path: Path, payload: object) -> str:
    """Write deterministic JSON and atomically replace the destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(text.encode()).hexdigest()


class SimulatedLabelOracle:
    """Persistent, deterministic oracle with immutable per-round requests."""

    def __init__(
        self,
        dataset: NormalizedDataset,
        split: SplitManifest,
        *,
        state_path: Path,
        campaign_id: str,
        total_budget: int,
        round_budgets: dict[str, int] | None = None,
        event_reference: str = "local:oracle",
        previous_state_path: Path | None = None,
        allow_new_campaign: bool = False,
    ) -> None:
        if total_budget < 0:
            raise OracleBudgetError("total budget must be non-negative")
        if dataset.dataset_id != split.dataset_id:
            raise OracleAccessError("dataset id does not match split")
        if dataset.content_hash() != split.qualified_dataset_content_sha256:
            raise OracleAccessError("dataset content hash does not match split provenance")
        self.dataset = dataset
        self.split = split
        self.state_path = state_path
        self.campaign_id = campaign_id
        self.total_budget = total_budget
        self.round_budgets = dict(sorted((round_budgets or {}).items()))
        self.event_reference = event_reference
        self._state_path_snapshot = sha256_file(state_path) if state_path.is_file() else None
        self._records = dataset.by_id()
        self._partition_by_id = {
            record_id: partition
            for partition, record_ids in split.record_ids.items()
            for record_id in record_ids
        }

        if state_path.is_file():
            self.state = OracleState.load(state_path)
        elif previous_state_path is not None:
            self.state = OracleState.load(previous_state_path)
        else:
            if not allow_new_campaign:
                raise OracleConflictError(
                    "creating a new campaign ledger requires explicit authority; "
                    "supply the prior state for continuation"
                )
            initial_ids = sorted(split.record_ids[PartitionName.INITIAL_LABELED.value])
            ledger = BudgetLedger(
                campaign_id=campaign_id,
                total_budget=total_budget,
                charged_total=0,
                remaining_budget=total_budget,
                round_budgets=self.round_budgets,
                round_charged={},
                revealed_record_ids=[],
                request_id_by_round={},
            )
            lineage = TrainingLineageManifest(
                campaign_id=campaign_id,
                dataset_id=dataset.dataset_id,
                dataset_content_sha256=dataset.content_hash(),
                split_semantic_sha256=split.semantic_hash(),
                initial_record_ids=initial_ids,
                reveal_history=[],
                training_record_ids=initial_ids,
            )
            self.state = OracleState(
                campaign_id=campaign_id,
                dataset_id=dataset.dataset_id,
                dataset_content_sha256=dataset.content_hash(),
                split_semantic_sha256=split.semantic_hash(),
                ledger=ledger,
                lineage=lineage,
                requests_by_round={},
                decisions_by_round={},
                batches_by_round={},
            )
        self._validate_state_contract()
        derived_manifest = NormalizedManifest.from_dataset(dataset, "0" * 64)
        structural_fingerprints = (
            derive_structural_fingerprint_clusters(dataset)
            if split.grouping_fields_used == [GroupingField.STRUCTURAL_FINGERPRINT.value]
            else None
        )
        split.validate_against(
            derived_manifest, structural_fingerprints=structural_fingerprints
        )

    def _validate_state_contract(self) -> None:
        state = self.state
        if state.campaign_id != self.campaign_id:
            raise OracleConflictError("oracle state belongs to a different campaign")
        if state.dataset_content_sha256 != self.dataset.content_hash():
            raise OracleConflictError("oracle state belongs to different dataset content")
        if state.split_semantic_sha256 != self.split.semantic_hash():
            raise OracleConflictError("oracle state belongs to a different split")
        if state.ledger.total_budget != self.total_budget:
            raise OracleConflictError("campaign budget cannot change during resume")
        if state.ledger.round_budgets != self.round_budgets:
            raise OracleConflictError("round budgets cannot change during resume")

    def _make_request(
        self, candidate_ids: list[str], campaign_id: str, round_id: str
    ) -> RevealRequest:
        if campaign_id != self.campaign_id:
            raise OracleConflictError("reveal campaign id does not match oracle state")
        if not round_id:
            raise OracleConflictError("round id must be non-empty")
        if not candidate_ids:
            raise OracleAccessError("reveal request must contain at least one candidate")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise OracleAccessError("duplicate candidate ids in one reveal request")
        sorted_ids = sorted(candidate_ids)
        identity = {
            "campaign_id": campaign_id,
            "round_id": round_id,
            "candidate_ids": sorted_ids,
            "dataset_content_sha256": self.dataset.content_hash(),
            "split_semantic_sha256": self.split.semantic_hash(),
        }
        request_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        return RevealRequest(
            request_id=request_id,
            campaign_id=campaign_id,
            round_id=round_id,
            candidate_ids=sorted_ids,
            dataset_id=self.dataset.dataset_id,
            dataset_content_sha256=self.dataset.content_hash(),
            split_semantic_sha256=self.split.semantic_hash(),
            event_reference=self.event_reference,
        )

    def reveal_result(
        self, candidate_ids: list[str], campaign_id: str, round_id: str
    ) -> OracleRevealResult:
        request = self._make_request(candidate_ids, campaign_id, round_id)
        existing = self.state.requests_by_round.get(round_id)
        if existing is not None:
            if existing != request:
                raise OracleConflictError(
                    f"round {round_id!r} already has a different immutable reveal request"
                )
            if not self.state_path.is_file():
                self._commit_state(self.state)
            return OracleRevealResult(
                request=existing,
                decision=self.state.decisions_by_round[round_id],
                label_batch=self.state.batches_by_round[round_id],
                ledger=self.state.ledger,
                lineage=self.state.lineage,
                state=self.state,
                idempotent_replay=True,
            )

        invalid: dict[str, list[str]] = {}
        for record_id in request.candidate_ids:
            partition = self._partition_by_id.get(record_id, "unknown")
            if partition != PartitionName.ACQUISITION_POOL.value:
                invalid.setdefault(partition, []).append(record_id)
        if invalid:
            detail = "; ".join(
                f"{partition}={sorted(ids)}" for partition, ids in sorted(invalid.items())
            )
            raise OracleAccessError(f"only acquisition-pool ids may be revealed: {detail}")

        already_revealed = set(self.state.ledger.revealed_record_ids)
        charged_ids = sorted(set(request.candidate_ids) - already_revealed)
        reused_ids = sorted(set(request.candidate_ids).intersection(already_revealed))
        round_limit = self.round_budgets.get(round_id, self.total_budget)
        round_before = self.state.ledger.round_charged.get(round_id, 0)
        if round_before + len(charged_ids) > round_limit:
            raise OracleBudgetError(
                f"round {round_id!r} budget exceeded: {round_before} + "
                f"{len(charged_ids)} > {round_limit}"
            )
        if self.state.ledger.charged_total + len(charged_ids) > self.total_budget:
            raise OracleBudgetError(
                "campaign label budget exceeded: "
                f"{self.state.ledger.charged_total} + {len(charged_ids)} > {self.total_budget}"
            )

        # Resolve and validate the entire label batch before any state write.
        missing_sources = sorted(set(request.candidate_ids) - set(self._records))
        if missing_sources:
            raise OracleAccessError(f"source dataset lacks selected ids: {missing_sources}")
        revealed = [
            self._revealed_label(self._records[record_id])
            for record_id in request.candidate_ids
        ]
        batch_id = hashlib.sha256(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "records": [record.model_dump(mode="json") for record in revealed],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        batch = LabelBatch(
            batch_id=batch_id,
            request_id=request.request_id,
            campaign_id=campaign_id,
            round_id=round_id,
            source_dataset_id=self.dataset.dataset_id,
            source_dataset_content_sha256=self.dataset.content_hash(),
            split_semantic_sha256=self.split.semantic_hash(),
            records=revealed,
        )

        ledger = BudgetLedger(
            campaign_id=campaign_id,
            total_budget=self.total_budget,
            charged_total=self.state.ledger.charged_total + len(charged_ids),
            remaining_budget=(
                self.total_budget - self.state.ledger.charged_total - len(charged_ids)
            ),
            round_budgets=self.round_budgets,
            round_charged={
                **self.state.ledger.round_charged,
                round_id: round_before + len(charged_ids),
            },
            revealed_record_ids=sorted(already_revealed.union(charged_ids)),
            request_id_by_round={
                **self.state.ledger.request_id_by_round,
                round_id: request.request_id,
            },
            prior_ledger_sha256=_canonical_hash(self.state.ledger),
        )
        decision = RevealDecision(
            request_id=request.request_id,
            charged_record_ids=charged_ids,
            previously_revealed_record_ids=reused_ids,
            campaign_budget_before=self.state.ledger.remaining_budget,
            campaign_budget_after=ledger.remaining_budget,
            round_budget_limit=round_limit,
            round_budget_after=round_limit - ledger.round_charged[round_id],
            reason=(
                "all selected ids are in the acquisition pool and declared budgets permit reveal"
            ),
            event_reference=self.event_reference,
        )
        training_ids = sorted(set(self.state.lineage.training_record_ids).union(charged_ids))
        lineage_entry = TrainingLineageEntry(
            round_id=round_id,
            request_id=request.request_id,
            label_batch_id=batch.batch_id,
            selected_record_ids=request.candidate_ids,
            newly_appended_record_ids=charged_ids,
        )
        lineage = TrainingLineageManifest(
            campaign_id=campaign_id,
            dataset_id=self.dataset.dataset_id,
            dataset_content_sha256=self.dataset.content_hash(),
            split_semantic_sha256=self.split.semantic_hash(),
            initial_record_ids=self.state.lineage.initial_record_ids,
            reveal_history=[*self.state.lineage.reveal_history, lineage_entry],
            training_record_ids=training_ids,
            prior_lineage_sha256=_canonical_hash(self.state.lineage),
        )
        next_state = OracleState(
            campaign_id=campaign_id,
            dataset_id=self.dataset.dataset_id,
            dataset_content_sha256=self.dataset.content_hash(),
            split_semantic_sha256=self.split.semantic_hash(),
            ledger=ledger,
            lineage=lineage,
            requests_by_round={**self.state.requests_by_round, round_id: request},
            decisions_by_round={**self.state.decisions_by_round, round_id: decision},
            batches_by_round={**self.state.batches_by_round, round_id: batch},
        )
        self._commit_state(next_state)
        self.state = next_state
        return OracleRevealResult(
            request=request,
            decision=decision,
            label_batch=batch,
            ledger=ledger,
            lineage=lineage,
            state=next_state,
        )

    def reveal(
        self, candidate_ids: list[str], campaign_id: str, round_id: str
    ) -> LabelBatch:
        return self.reveal_result(candidate_ids, campaign_id, round_id).label_batch

    def _revealed_label(self, record: LabeledConfiguration) -> RevealedLabel:
        return RevealedLabel(
            record_id=record.config_id,
            energy_ev=record.energy_ev,
            forces_ev_per_a=record.forces_ev_per_a,
            virial_stress_kbar=record.virial_stress_kbar,
            level_of_theory=record.level_of_theory,
            source_dataset_id=self.dataset.dataset_id,
            source_id=record.source_id,
            source_record_sha256=record.source_record_sha256,
            normalized_record_sha256=record.content_hash(),
        )

    def _commit_state(self, state: OracleState) -> None:
        """Locked compare-and-swap prevents stale writers from losing reveals."""
        lock_path = self.state_path.with_name(f".{self.state_path.name}.lock")
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise OracleConflictError(
                    "another oracle writer holds the campaign state lock"
                ) from exc
            current = sha256_file(self.state_path) if self.state_path.is_file() else None
            if current != self._state_path_snapshot:
                raise OracleConflictError(
                    "oracle state changed since it was loaded; stale reveal refused"
                )
            new_hash = write_json_atomic(self.state_path, state.model_dump(mode="json"))
            self._state_path_snapshot = new_hash
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
