"""Fail-closed protected-pool and model checks for MACE descriptors."""

from __future__ import annotations

from mlip_research_agent.data.oracle import AcquisitionView
from mlip_research_agent.data.split import PartitionName


def validate_view(view: AcquisitionView, supported_species: list[str]) -> None:
    if view.partition != PartitionName.ACQUISITION_POOL.value:
        raise ValueError("MACE descriptors require the acquisition_pool partition")
    record_ids = [candidate.record_id for candidate in view.candidates]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("MACE descriptor acquisition view has duplicate candidate ids")
    supported = set(supported_species)
    for candidate in view.candidates:
        unknown = sorted(set(candidate.symbols) - supported)
        if unknown:
            raise ValueError(
                f"candidate {candidate.record_id}: unsupported checkpoint species {unknown}"
            )


def validate_num_layers(num_layers: int, num_interactions: int) -> None:
    if num_interactions < 1:
        raise ValueError("MACE checkpoint exposes no interaction layers")
    if num_layers != -1 and not 1 <= num_layers <= num_interactions:
        raise ValueError(
            f"num_layers must be -1 or between 1 and {num_interactions}; got {num_layers}"
        )
