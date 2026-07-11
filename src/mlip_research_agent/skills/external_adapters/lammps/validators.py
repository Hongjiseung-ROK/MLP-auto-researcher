"""Validation for the prepare-only LAMMPS adapter."""

from mlip_research_agent.skills.external_adapters.common import adapter_error
from mlip_research_agent.skills.external_adapters.validators_shared import (
    finite_structure_records,
    structure_index,
)


def require_orthorhombic(cell: list[list[float]], tolerance: float = 1e-10) -> None:
    """The v1 data writer supports orthorhombic cells only; fail closed otherwise."""
    for i in range(3):
        for j in range(3):
            if i != j and abs(cell[i][j]) > tolerance:
                raise adapter_error(
                    "prepare-only LAMMPS adapter supports orthorhombic cells; "
                    f"cell[{i}][{j}] = {cell[i][j]!r}"
                )


__all__ = ["finite_structure_records", "require_orthorhombic", "structure_index"]
