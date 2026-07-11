"""Claim verification: every claim must trace to registered, hash-intact artifacts.

Agent text is never evidence. A claim is VERIFIED only if every referenced
artifact exists in the run manifest and its on-disk sha256 still matches.
"""

from __future__ import annotations

import json
from collections import Counter

from pydantic import BaseModel

from mlip_research_agent.schemas.claims import Claim, ClaimStatus
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.verification.schema import VerificationInput, VerificationOutput
from mlip_research_agent.skills.verification.validators import load_claims

REPORT_FILENAME = "verification_report.json"
VERIFIED_CLAIMS_FILENAME = "verified_claims.json"

# Narrative/agent-text artifacts may explain a result but can never verify a
# numerical claim (plan.md data-provenance policy).
FORBIDDEN_EVIDENCE_KINDS = frozenset({"agent_text", "reflection", "report", "run_report"})


@register_skill
class ClaimVerificationSkill(Skill):
    name = "claim_verification"
    input_model = VerificationInput
    output_model = VerificationOutput
    cost_class = "trivial"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, VerificationInput)
        claims = load_claims(ctx.run_dir, params.claims_path)

        audited: list[Claim] = []
        for claim in claims:
            missing = [ref for ref in claim.artifact_references if ctx.registry.get(ref) is None]
            corrupted = [
                ref
                for ref in claim.artifact_references
                if ctx.registry.get(ref) is not None and not ctx.registry.verify(ref)
            ]
            prose = [
                ref
                for ref in claim.artifact_references
                if (artifact := ctx.registry.get(ref)) is not None
                and artifact.kind in FORBIDDEN_EVIDENCE_KINDS
            ]
            if missing or corrupted or prose:
                if missing:
                    category = "unregistered_artifact"
                elif corrupted:
                    category = "corrupted_artifact"
                else:
                    category = "agent_prose_as_evidence"
                audited.append(
                    claim.model_copy(
                        update={
                            "status": ClaimStatus.REJECTED,
                            "rejection_reason": (
                                f"unregistered artifacts: {missing}; "
                                f"corrupted artifacts: {corrupted}; "
                                f"forbidden agent prose artifacts: {prose}"
                            ),
                            "rejection_reason_category": category,
                            "rejection_evidence": [
                                *[f"unregistered:{ref}" for ref in missing],
                                *[f"corrupted:{ref}" for ref in corrupted],
                                *[f"agent_prose:{ref}" for ref in prose],
                            ],
                        }
                    )
                )
            else:
                audited.append(
                    claim.model_copy(
                        update={
                            "status": ClaimStatus.VERIFIED,
                            "rejection_reason": None,
                            "rejection_reason_category": None,
                            "rejection_evidence": [],
                        }
                    )
                )

        n_verified = sum(1 for c in audited if c.status is ClaimStatus.VERIFIED)
        n_rejected = sum(1 for c in audited if c.status is ClaimStatus.REJECTED)
        counts_by_claim_class = dict(
            sorted(Counter(c.claim_class.value for c in audited).items())
        )
        counts_by_evidence_tier = dict(
            sorted(Counter(c.scientific_evidence_tier.value for c in audited).items())
        )
        counts_by_verification_status = dict(
            sorted(Counter(c.status.value for c in audited).items())
        )
        rejection_reason_categories = dict(
            sorted(
                Counter(
                    c.rejection_reason_category or "unspecified"
                    for c in audited
                    if c.status is ClaimStatus.REJECTED
                ).items()
            )
        )

        verified_path = ctx.step_dir / VERIFIED_CLAIMS_FILENAME
        verified_path.write_text(
            json.dumps([c.model_dump(mode="json") for c in audited], indent=2, sort_keys=True)
            + "\n"
        )
        report_path = ctx.step_dir / REPORT_FILENAME
        report_path.write_text(
            json.dumps(
                {
                    "n_claims": len(audited),
                    "n_verified": n_verified,
                    "n_rejected": n_rejected,
                    "artifact_integrity": {
                        "verified": n_verified,
                        "rejected": n_rejected,
                    },
                    "counts_by_claim_class": counts_by_claim_class,
                    "counts_by_evidence_tier": counts_by_evidence_tier,
                    "counts_by_verification_status": counts_by_verification_status,
                    "rejection_reason_categories": rejection_reason_categories,
                    "rejected_claim_ids": sorted(
                        c.claim_id for c in audited if c.status is ClaimStatus.REJECTED
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        report_artifact = ctx.registry.register(
            report_path, kind="verification_report", step_id=ctx.step_id
        )
        verified_artifact = ctx.registry.register(
            verified_path, kind="verified_claims", step_id=ctx.step_id
        )

        if params.strict and n_rejected > 0:
            # Artifacts above are already registered, so the evidence of the
            # rejection is preserved before the step fails.
            raise SkillError(
                f"{n_rejected} claim(s) lack verifiable backing artifacts",
                failure_class=FailureClass.UNSUPPORTED_CLAIM,
                severity=Severity.HIGH,
                retryable=False,
                likely_causes=[
                    "claim registered against unregistered artifact ids",
                    "artifact files modified or deleted after registration",
                ],
            )

        return VerificationOutput(
            report_artifact=report_artifact.artifact_id,
            report_path=report_artifact.relative_path,
            verified_claims_artifact=verified_artifact.artifact_id,
            verified_claims_path=verified_artifact.relative_path,
            n_claims=len(audited),
            n_verified=n_verified,
            n_rejected=n_rejected,
            counts_by_claim_class=counts_by_claim_class,
            counts_by_evidence_tier=counts_by_evidence_tier,
            counts_by_verification_status=counts_by_verification_status,
            rejection_reason_categories=rejection_reason_categories,
        )
