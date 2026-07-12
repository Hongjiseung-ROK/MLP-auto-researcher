"""Repository bridge from official Ralphthon General Track 1 to MLIP contracts."""

from __future__ import annotations

import json

from pydantic import BaseModel

from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.competition.ralphthon_mlip_track1.schema import (
    RalphthonMLIPTrack1Input,
    RalphthonMLIPTrack1Output,
)
from mlip_research_agent.skills.competition.ralphthon_mlip_track1.validators import (
    OFFICIAL_SKILL_DEPENDENCIES,
    validate_mlip_route,
)


@register_skill
class RalphthonMLIPTrack1Skill(Skill):
    name = "ralphthon_mlip_track1"
    input_model = RalphthonMLIPTrack1Input
    output_model = RalphthonMLIPTrack1Output
    cost_class = "gated"
    permission_level = "human_approval"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, RalphthonMLIPTrack1Input)
        validate_mlip_route(params)
        route = [
            "official_ralphthon_general_track_1",
            "project_mlip_research_controller",
            "optional_vessl_cloud_provider" if params.use_vessl_compute else "local_dry_run",
            "track_1_short_paper_template",
            "self_review",
        ]
        path = ctx.step_dir / "ralphthon_mlip_track1_contract.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "route": route,
                    "required_official_skills": list(OFFICIAL_SKILL_DEPENDENCIES),
                    "evidence_metrics": params.metrics,
                    "paper_page_range": [2, 4],
                    "self_review_required": True,
                    "scientific_status": "infrastructure_only",
                    "claim_eligible": False,
                    "karpathy_training_used": False,
                    "wandb_mode": params.wandb_mode,
                    "wandb_online_enabled": False,
                    "h2_open": True,
                    "h3_open": True,
                    "h5_open": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        artifact = ctx.registry.register(path, "ralphthon_track1_contract", ctx.step_id)
        return RalphthonMLIPTrack1Output(
            route=route,
            required_official_skills=list(OFFICIAL_SKILL_DEPENDENCIES),
            evidence_metrics=params.metrics,
            paper_page_range=(2, 4),
            self_review_required=True,
            track1_contract_artifact=artifact.artifact_id,
        )
