"""D0-only energy-reference diagnostics for MACE checkpoint due diligence."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlip_research_agent.data.manifests import NormalizedDataset
from mlip_research_agent.data.split import PartitionName, SplitManifest
from mlip_research_agent.schemas.predictions import PredictionBatch


class GroupEnergyOffset(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    group_id: str
    n_structures: int = Field(gt=0)
    mean_offset_ev_per_atom: float
    std_offset_ev_per_atom: float = Field(ge=0)


class CandidateE0Diagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model_id: str
    checkpoint_sha256: str = Field(min_length=64, max_length=64)
    cu_atomic_reference_energy_ev: float
    n_structures: int = Field(gt=0)
    mean_offset_ev_per_atom: float
    median_offset_ev_per_atom: float
    std_offset_ev_per_atom: float = Field(ge=0)
    mae_offset_ev_per_atom: float = Field(ge=0)
    p95_abs_offset_ev_per_atom: float = Field(ge=0)
    max_abs_group_mean_offset_ev_per_atom: float = Field(ge=0)
    groups: list[GroupEnergyOffset] = Field(min_length=1)
    no_offset_fit_applied: Literal[True] = True
    eligibility_status: Literal["threshold_not_frozen"] = "threshold_not_frozen"


class E0DiagnosticReport(BaseModel):
    """Evidence-only report; it deliberately cannot choose a checkpoint."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["1.0.0"] = "1.0.0"
    diagnostic_algorithm: Literal["d0_raw_energy_offset_v1"] = "d0_raw_energy_offset_v1"
    dataset_id: str
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    partition: Literal["initial_labeled"] = "initial_labeled"
    candidates: list[CandidateE0Diagnostic] = Field(min_length=1)
    selection_status: Literal["evidence_only_not_h2_selected"] = (
        "evidence_only_not_h2_selected"
    )
    limitations: list[str] = Field(min_length=1)

    def save(self, path: Path) -> str:
        text = json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return text


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _candidate_summary(
    batch: PredictionBatch,
    dataset: NormalizedDataset,
    expected_ids: set[str],
    cu_atomic_reference_energy_ev: float,
) -> CandidateE0Diagnostic:
    prediction_by_id = {record.record_id: record for record in batch.predictions}
    actual_ids = set(prediction_by_id)
    if actual_ids != expected_ids:
        raise ValueError(
            "D0 prediction coverage mismatch; "
            f"missing={sorted(expected_ids - actual_ids)[:5]}, "
            f"extra={sorted(actual_ids - expected_ids)[:5]}"
        )
    dataset_by_id = dataset.by_id()
    offsets: list[float] = []
    by_group: dict[str, list[float]] = defaultdict(list)
    for record_id in sorted(expected_ids):
        target = dataset_by_id[record_id]
        prediction = prediction_by_id[record_id]
        offset = (prediction.energy_ev - target.energy_ev) / target.n_atoms
        if not math.isfinite(offset):
            raise ValueError(f"non-finite D0 energy offset for {record_id}")
        offsets.append(offset)
        by_group[target.group_id].append(offset)
    groups = [
        GroupEnergyOffset(
            group_id=group_id,
            n_structures=len(values),
            mean_offset_ev_per_atom=statistics.fmean(values),
            std_offset_ev_per_atom=statistics.pstdev(values),
        )
        for group_id, values in sorted(by_group.items())
    ]
    abs_offsets = [abs(value) for value in offsets]
    return CandidateE0Diagnostic(
        model_id=batch.model_id,
        checkpoint_sha256=batch.checkpoint_sha256,
        cu_atomic_reference_energy_ev=cu_atomic_reference_energy_ev,
        n_structures=len(offsets),
        mean_offset_ev_per_atom=statistics.fmean(offsets),
        median_offset_ev_per_atom=statistics.median(offsets),
        std_offset_ev_per_atom=statistics.pstdev(offsets),
        mae_offset_ev_per_atom=statistics.fmean(abs_offsets),
        p95_abs_offset_ev_per_atom=_percentile(abs_offsets, 0.95),
        max_abs_group_mean_offset_ev_per_atom=max(
            abs(group.mean_offset_ev_per_atom) for group in groups
        ),
        groups=groups,
    )


def compute_e0_diagnostic(
    d0_label_view: NormalizedDataset,
    qualified_dataset_content_sha256: str,
    split: SplitManifest,
    prediction_batches: list[PredictionBatch],
    cu_atomic_reference_energies_ev: dict[str, float],
) -> E0DiagnosticReport:
    """Compare raw D0 offsets without fitting, shifting, ranking, or selecting."""
    if split.dataset_id != d0_label_view.dataset_id:
        raise ValueError("D0 diagnostic dataset/split id mismatch")
    if split.qualified_dataset_content_sha256 != qualified_dataset_content_sha256:
        raise ValueError("D0 diagnostic dataset/split content hash mismatch")
    expected_ids = set(split.record_ids[PartitionName.INITIAL_LABELED.value])
    if not expected_ids:
        raise ValueError("D0 diagnostic requires a non-empty initial_labeled partition")
    if set(d0_label_view.by_id()) != expected_ids:
        raise ValueError("D0 label view must contain exactly the initial_labeled records")
    model_ids = [batch.model_id for batch in prediction_batches]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("D0 diagnostic candidate model ids must be unique")
    candidates: list[CandidateE0Diagnostic] = []
    for batch in prediction_batches:
        if batch.dataset_id != d0_label_view.dataset_id:
            raise ValueError(f"{batch.model_id}: prediction dataset id mismatch")
        if batch.dataset_content_sha256 != qualified_dataset_content_sha256:
            raise ValueError(f"{batch.model_id}: prediction dataset content hash mismatch")
        if batch.split_semantic_sha256 != split.semantic_hash():
            raise ValueError(f"{batch.model_id}: prediction split identity mismatch")
        try:
            cu_e0 = cu_atomic_reference_energies_ev[batch.model_id]
        except KeyError as exc:
            raise ValueError(f"{batch.model_id}: missing serialized Cu atomic reference") from exc
        candidates.append(_candidate_summary(batch, d0_label_view, expected_ids, cu_e0))
    return E0DiagnosticReport(
        dataset_id=d0_label_view.dataset_id,
        dataset_content_sha256=qualified_dataset_content_sha256,
        split_semantic_sha256=split.semantic_hash(),
        candidates=sorted(candidates, key=lambda candidate: candidate.model_id),
        limitations=[
            "No pass/fail threshold is frozen; this report cannot select a checkpoint.",
            "Raw per-atom energy residuals mix reference-energy offset and model error.",
            "Target POTCAR variant and smearing are unknown; exact reference parity is unproven.",
            "No offset was fitted, subtracted, or optimized from D0 labels.",
            "Validation, frozen-test, and stress-test labels were not accessed.",
        ],
    )
