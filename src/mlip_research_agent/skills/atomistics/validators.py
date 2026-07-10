"""Domain validation for generated structures."""

from __future__ import annotations

import numpy as np

from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.atomistics.schema import (
    SUPPORTED_LATTICES,
    StructureGenerationInput,
)
from mlip_research_agent.skills.atomistics.structures_io import StructureSet, record_to_atoms
from mlip_research_agent.skills.base import SkillError

MIN_INTERATOMIC_DISTANCE = 0.5  # Angstrom; below this the geometry is unphysical


def validate_inputs(inputs: StructureGenerationInput) -> None:
    if inputs.crystal_structure not in SUPPORTED_LATTICES:
        raise SkillError(
            f"unsupported crystal structure {inputs.crystal_structure!r}; "
            f"supported: {SUPPORTED_LATTICES}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )


def validate_structure_set(structures: StructureSet) -> None:
    for record in structures.systems:
        atoms = record_to_atoms(record)
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        min_distance = float(distances.min())
        if min_distance < MIN_INTERATOMIC_DISTANCE:
            raise SkillError(
                f"structure {record.index} has interatomic distance {min_distance:.3f} A "
                f"< {MIN_INTERATOMIC_DISTANCE} A",
                failure_class=FailureClass.SIMULATION_INSTABILITY,
                severity=Severity.MEDIUM,
                retryable=True,
                likely_causes=["perturbation amplitude too large for lattice"],
                repair_params={"max_perturbation": 0.05},
            )
