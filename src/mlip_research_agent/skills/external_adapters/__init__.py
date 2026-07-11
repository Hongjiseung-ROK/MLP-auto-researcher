"""Adapters over the locked external computational-chemistry skill pack.

Registration is pinned in
``external-skills/jinzhezenggroup/computational-chemistry-agent-skills/upstream.yaml``.
Every adapter fails closed when its optional dependency is absent and emits
infrastructure artifacts only — never claim evidence.
"""

from mlip_research_agent.skills.external_adapters.ase import ExternalASEBuildSkill
from mlip_research_agent.skills.external_adapters.deepmd_inference import (
    ExternalDeepMDInferenceSkill,
)
from mlip_research_agent.skills.external_adapters.dft_qe import ExternalQEPrepareSkill
from mlip_research_agent.skills.external_adapters.dpdata_cli import ExternalDpdataConvertSkill
from mlip_research_agent.skills.external_adapters.lammps import ExternalLAMMPSPrepareSkill
from mlip_research_agent.skills.external_adapters.phonopy import ExternalPhonopySkill
from mlip_research_agent.skills.external_adapters.pymatgen_structure import (
    ExternalPymatgenSymmetrySkill,
)
from mlip_research_agent.skills.external_adapters.rdkit import ExternalRDKitConformerSkill

__all__ = [
    "ExternalASEBuildSkill",
    "ExternalDeepMDInferenceSkill",
    "ExternalDpdataConvertSkill",
    "ExternalLAMMPSPrepareSkill",
    "ExternalPhonopySkill",
    "ExternalPymatgenSymmetrySkill",
    "ExternalQEPrepareSkill",
    "ExternalRDKitConformerSkill",
]
