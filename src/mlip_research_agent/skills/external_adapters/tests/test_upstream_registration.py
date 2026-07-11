"""The code-side upstream pin must match the committed upstream.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from mlip_research_agent.skills.base import registered_skills
from mlip_research_agent.skills.external_adapters.common import (
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
UPSTREAM_YAML = (
    REPO_ROOT
    / "external-skills"
    / "jinzhezenggroup"
    / "computational-chemistry-agent-skills"
    / "upstream.yaml"
)

EXPECTED_SKILLS = {
    "external_ase",
    "external_pymatgen_structure",
    "external_dpdata_cli",
    "external_dft_qe",
    "external_phonopy",
    "external_deepmd_inference",
    "external_lammps",
    "external_rdkit",
}


def test_upstream_yaml_matches_code_pin() -> None:
    payload = yaml.safe_load(UPSTREAM_YAML.read_text())
    assert payload["repository"] == UPSTREAM_REPOSITORY
    assert payload["commit"] == UPSTREAM_COMMIT
    adopted = {entry["skill_name"] for entry in payload["adopted_skills"]}
    assert adopted == EXPECTED_SKILLS


def test_all_adapters_are_registered() -> None:
    names = set(registered_skills())
    missing = EXPECTED_SKILLS - names
    assert not missing, f"unregistered external adapters: {sorted(missing)}"
