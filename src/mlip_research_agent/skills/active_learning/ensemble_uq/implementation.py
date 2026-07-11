"""Ensemble force-disagreement ranking (label-free acquisition signal)."""

from __future__ import annotations

import json

import numpy as np
from pydantic import BaseModel

from mlip_research_agent.skills.active_learning.ensemble_uq.schema import (
    POLICY_VERSION,
    EnsembleUQInput,
    EnsembleUQOutput,
)
from mlip_research_agent.skills.active_learning.ensemble_uq.validators import (
    force_array,
    validate_inputs,
)
from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill

UQ_FILENAME = "ensemble_disagreement.json"


@register_skill
class EnsembleUQSkill(Skill):
    name = "ensemble_uq"
    input_model = EnsembleUQInput
    output_model = EnsembleUQOutput
    cost_class = "cheap"
    permission_level = "auto"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, EnsembleUQInput)
        validate_inputs(params)

        ranked: list[dict[str, object]] = []
        invalid_member_flags: dict[str, list[str]] = {}
        for candidate_id in sorted(params.pool_candidate_ids):
            n_atoms = params.metadata[candidate_id].n_atoms
            stacks: list[np.ndarray] = []
            bad_members: list[str] = []
            energies: list[float] = []
            for member in params.members:
                arr = force_array(
                    member.member_id,
                    candidate_id,
                    member.predicted_forces[candidate_id],
                    n_atoms,
                )
                if not np.all(np.isfinite(arr)):
                    bad_members.append(member.member_id)
                    continue
                stacks.append(arr)
                if params.include_energy_disagreement:
                    energy = member.predicted_energy_per_atom[candidate_id]
                    if not np.isfinite(energy):
                        bad_members.append(member.member_id)
                        stacks.pop()
                        continue
                    energies.append(energy)
            if bad_members:
                invalid_member_flags[candidate_id] = sorted(set(bad_members))
                continue  # a candidate with any invalid member is not ranked

            # Disagreement: per-atom std of force components across members,
            # aggregated as the mean per-atom disagreement norm.
            forces = np.stack(stacks)  # (n_members, n_atoms, 3)
            per_component_std = np.std(forces, axis=0, ddof=0)  # (n_atoms, 3)
            per_atom_norm = np.linalg.norm(per_component_std, axis=1)  # (n_atoms,)
            force_disagreement = float(np.mean(per_atom_norm))
            record: dict[str, object] = {
                "candidate_id": candidate_id,
                "force_disagreement": force_disagreement,
                "max_atom_disagreement": float(np.max(per_atom_norm)),
                "source_group": params.metadata[candidate_id].source_group,
            }
            if params.include_energy_disagreement:
                record["energy_per_atom_disagreement"] = float(np.std(energies, ddof=0))
            ranked.append(record)

        # Stable ranking: disagreement descending, candidate id as tiebreak.
        def sort_key(r: dict[str, object]) -> tuple[float, str]:
            disagreement = r["force_disagreement"]
            candidate = r["candidate_id"]
            assert isinstance(disagreement, float) and isinstance(candidate, str)
            return (-disagreement, candidate)

        ranked.sort(key=sort_key)
        for rank, record in enumerate(ranked):
            record["rank"] = rank

        payload = {
            "schema_version": "1.0.0",
            "campaign_id": params.campaign_id,
            "round_id": params.round_id,
            "policy_version": POLICY_VERSION,
            "signal_kind": "ensemble_disagreement",
            "signal_caveat": (
                "Member disagreement is an acquisition ranking signal, not "
                "calibrated predictive uncertainty."
            ),
            "member_identities": [
                {"member_id": m.member_id, "provenance_sha256": m.provenance_sha256}
                for m in params.members
            ],
            "invalid_member_flags": invalid_member_flags,
            "records": ranked,
        }
        path = ctx.step_dir / UQ_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        artifact = ctx.registry.register(path, kind="selection_signal", step_id=ctx.step_id)
        return EnsembleUQOutput(
            uq_artifact=artifact.artifact_id,
            uq_path=artifact.relative_path,
            n_candidates_ranked=len(ranked),
            n_candidates_invalid=len(invalid_member_flags),
            invalid_member_flags=invalid_member_flags,
            policy_version=POLICY_VERSION,
        )
