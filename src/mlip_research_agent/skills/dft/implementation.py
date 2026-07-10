"""Mock DFT labeling with a deterministic Lennard-Jones stand-in.

Energies are not physically meaningful; they are stable, cheap labels that
exercise the full labeling contract (selection handling, settings provenance,
convergence-failure taxonomy, bounded recovery).
"""

from __future__ import annotations

import json

from ase.calculators.lj import LennardJones
from pydantic import BaseModel

from mlip_research_agent.schemas.failure import FailureClass, RecoveryDecision, Severity
from mlip_research_agent.skills.atomistics.structures_io import StructureSet, record_to_atoms
from mlip_research_agent.skills.base import (
    Skill,
    SkillContext,
    SkillError,
    expect_inputs,
    register_skill,
)
from mlip_research_agent.skills.dft.schema import LabelingInput, LabelingOutput, LabelRecord
from mlip_research_agent.skills.dft.validators import validate_inputs, validate_labels

LABELS_FILENAME = "labels.json"
LJ_SETTINGS = {"sigma": 2.3, "epsilon": 0.1, "rc": 6.0}


@register_skill
class MockDFTLabelingSkill(Skill):
    name = "mock_dft_labeling"
    input_model = LabelingInput
    output_model = LabelingOutput
    cost_class = "cheap"  # the real skill will be 'gated' (human approval, licenses)

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, LabelingInput)
        validate_inputs(params)

        if ctx.attempt < params.inject_failure_times:
            raise SkillError(
                "mock SCF did not converge within iteration cap (injected failure "
                f"{ctx.attempt + 1}/{params.inject_failure_times})",
                failure_class=FailureClass.CONVERGENCE_FAILURE,
                severity=Severity.MEDIUM,
                retryable=True,
                likely_causes=["SCF oscillation (mock)", "poor initial guess (mock)"],
                recommended_action=RecoveryDecision.REFINE,
                repair_params={"scf_damping": 0.7},
            )

        structures = StructureSet.load(ctx.run_dir / params.structures_path)
        indices = list(range(len(structures.systems)))
        if params.selection_path is not None:
            selection = json.loads((ctx.run_dir / params.selection_path).read_text())
            indices = [int(i) for i in selection["selected_indices"]]

        calculator = LennardJones(**LJ_SETTINGS)
        labels: list[LabelRecord] = []
        by_index = {record.index: record for record in structures.systems}
        for i in indices:
            atoms = record_to_atoms(by_index[i])
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
            fmax = float((atoms.get_forces() ** 2).sum(axis=1).max() ** 0.5)
            labels.append(LabelRecord(index=i, energy=round(energy, 10), fmax=round(fmax, 10)))
        validate_labels(labels)

        path = ctx.step_dir / LABELS_FILENAME
        payload = {
            "method": params.method,
            "settings": {**LJ_SETTINGS, "scf_damping": params.scf_damping},
            "labels": [rec.model_dump() for rec in labels],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        artifact = ctx.registry.register(path, kind="label_set", step_id=ctx.step_id)
        return LabelingOutput(
            labels_artifact=artifact.artifact_id,
            labels_path=artifact.relative_path,
            n_labeled=len(labels),
            method=params.method,
        )
