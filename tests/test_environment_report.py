"""Environment report generation tests."""

import json
from pathlib import Path

from mlip_research_agent.provenance.environment_report import generate_environment_report

REQUIRED_KEYS = {
    "environment_name",
    "python_version",
    "operating_system",
    "architecture",
    "cuda",
    "installed_direct_dependencies",
    "failed_or_unavailable_tools",
    "commands_executed",
    "generated_at",
    "environment_file_hash",
}


def test_report_contains_required_fields(tmp_path: Path) -> None:
    env_file = tmp_path / "environment.yml"
    env_file.write_text("name: mlip-research-agent\n")
    path = generate_environment_report(
        tmp_path / "report.json",
        environment_file=env_file,
        commands_executed=["conda env create -f environment.yml"],
    )
    report = json.loads(path.read_text())
    assert set(report) >= REQUIRED_KEYS
    assert report["environment_name"] == "mlip-research-agent"
    assert isinstance(report["cuda"]["available"], bool)
    assert report["environment_file_hash"] is not None
    assert report["installed_direct_dependencies"]["numpy"] != "not-installed"


def test_missing_environment_file_yields_null_hash(tmp_path: Path) -> None:
    path = generate_environment_report(tmp_path / "report.json", environment_file=None)
    assert json.loads(path.read_text())["environment_file_hash"] is None
