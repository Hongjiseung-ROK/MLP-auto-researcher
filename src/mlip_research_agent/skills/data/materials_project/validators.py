"""Validation for the materials_project_recon skill."""

from __future__ import annotations

import re

from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError
from mlip_research_agent.skills.data.materials_project.schema import MaterialsProjectInput

#: Dash-separated element symbols, e.g. "Cu" or "Li-Fe-O".
CHEMSYS_PATTERN = re.compile(r"^[A-Z][a-z]?(-[A-Z][a-z]?)*$")
MP_ID_PATTERN = re.compile(r"^mp-\d+$")
#: Summary field names are plain snake_case identifiers.
FIELD_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

MAX_RESULTS_MIN = 1
MAX_RESULTS_MAX = 500


def _invalid(message: str) -> SkillError:
    return SkillError(
        message,
        failure_class=FailureClass.VALIDATION_ERROR,
        severity=Severity.HIGH,
        retryable=False,
    )


def validate_chemsys(chemsys: str) -> None:
    if not CHEMSYS_PATTERN.match(chemsys):
        raise _invalid(
            f"invalid chemsys {chemsys!r}: expected dash-separated element symbols like "
            "'Cu' or 'Li-Fe-O'"
        )


def validate_material_ids(material_ids: list[str], *, require_nonempty: bool) -> None:
    if require_nonempty and not material_ids:
        raise _invalid("material_ids must be non-empty for this operation")
    for mid in material_ids:
        if not MP_ID_PATTERN.match(mid):
            raise _invalid(f"invalid material id {mid!r}: expected the form 'mp-<digits>'")


def validate_fields(fields: list[str]) -> None:
    if not fields:
        raise _invalid("fields must be non-empty: list the summary fields to fetch")
    for name in fields:
        if not FIELD_PATTERN.match(name):
            raise _invalid(f"invalid field name {name!r}: expected snake_case identifiers")
    if "material_id" not in fields:
        raise _invalid(
            "fields must include 'material_id': results are sorted deterministically by it"
        )


def validate_max_results(max_results: int) -> None:
    if not MAX_RESULTS_MIN <= max_results <= MAX_RESULTS_MAX:
        raise _invalid(
            f"max_results must be within [{MAX_RESULTS_MIN}, {MAX_RESULTS_MAX}]; "
            f"got {max_results}"
        )


def validate_inputs(inputs: MaterialsProjectInput) -> None:
    validate_max_results(inputs.max_results)
    if inputs.operation == "summary_search":
        validate_chemsys(inputs.chemsys)
        validate_fields(inputs.fields)
    elif inputs.operation == "get_structures":
        validate_material_ids(inputs.material_ids, require_nonempty=True)
    # auth_check needs no further validation: it uses a fixed cheap probe query.
