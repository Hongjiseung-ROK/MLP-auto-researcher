"""Research program guardrail keeps Cu in a bounded scaffold role."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_research_agent.research.preregistration import HumanGate
from mlip_research_agent.research.program_guardrail import ResearchProgramGuardrail
from mlip_research_agent.schemas.claims import ScientificEvidenceTier

ROOT = Path(__file__).resolve().parents[1]
GUARDRAIL_PATH = ROOT / "configs/research/program_guardrail.yaml"


def test_committed_guardrail_is_machine_readable_and_bounded() -> None:
    guardrail = ResearchProgramGuardrail.load(GUARDRAIL_PATH)
    assert guardrail.allowed_claim_tier is ScientificEvidenceTier.PILOT_ONLY
    assert guardrail.next_human_gate is HumanGate.H2_PREREGISTRATION
    assert "scaffold" in guardrail.current_benchmark_role.lower()
    assert len(guardrail.criteria_to_exit_benchmark_mode) >= 4
    assert len(guardrail.content_hash()) == 64


def test_guardrail_hash_is_semantic_and_stable(tmp_path: Path) -> None:
    guardrail = ResearchProgramGuardrail.load(GUARDRAIL_PATH)
    rewritten = tmp_path / "guardrail.yaml"
    rewritten.write_text(GUARDRAIL_PATH.read_text().replace("schema_version:", "schema_version: "))
    assert ResearchProgramGuardrail.load(rewritten).content_hash() == guardrail.content_hash()


def test_elevated_claim_tiers_are_rejected() -> None:
    guardrail = ResearchProgramGuardrail.load(GUARDRAIL_PATH)
    guardrail.assert_claim_tier_allowed(ScientificEvidenceTier.PILOT_ONLY)
    with pytest.raises(ValueError, match="exceeds program limit"):
        guardrail.assert_claim_tier_allowed(ScientificEvidenceTier.REPLICATED)

    payload = guardrail.model_dump(mode="json")
    payload["allowed_claim_tier"] = ScientificEvidenceTier.PUBLICATION_ELIGIBLE.value
    with pytest.raises(ValidationError, match="cannot authorize"):
        ResearchProgramGuardrail.model_validate(payload)
