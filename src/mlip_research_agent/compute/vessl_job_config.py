"""Dry-run Job control records pending live vesslctl file-schema calibration."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from mlip_research_agent.compute.vessl_schemas import (
    COMMIT_PATTERN,
    SHA256_PATTERN,
    VesslRecord,
    VesslVolumeMount,
)


class VesslGeneratedJobConfig(VesslRecord):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_kind: Literal["normalized_dry_run_control"] = "normalized_dry_run_control"
    vesslctl_file_schema_verified: Literal[False] = False
    phase: Literal["iteration_1", "iteration_2"]
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    git_commit: str = Field(pattern=COMMIT_PATTERN)
    resource_spec_slug: str = Field(min_length=1)
    image: str = Field(min_length=1)
    volume_mounts: list[VesslVolumeMount]
    bounded_label_view_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    split_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_lock_sha256: str = Field(pattern=SHA256_PATTERN)
    remote_steps: list[list[str]] = Field(min_length=7)
    submission_argv_template: list[str]
    scientific_status: Literal["infrastructure_only"] = "infrastructure_only"
    claim_eligible: Literal[False] = False

    @model_validator(mode="after")
    def _required_steps(self) -> VesslGeneratedJobConfig:
        rendered = [" ".join(step) for step in self.remote_steps]
        required = (
            "git clone",
            f"checkout --detach {self.git_commit}",
            "rev-parse HEAD",
            "pip install",
            "verify-environment",
            "attest-gpu",
            f"run-phase {self.phase}",
            "write-artifact-manifest",
        )
        joined = "\n".join(rendered)
        missing = [needle for needle in required if needle not in joined]
        if missing:
            raise ValueError(f"generated VESSL Job control misses steps: {missing}")
        if self.submission_argv_template != [
            "vesslctl",
            "job",
            "create",
            "--file",
            "<live-schema-verified-job-config.json>",
        ]:
            raise ValueError("VESSL submission template must use job create --file")
        return self


def build_dry_run_job_config(
    *,
    phase: Literal["iteration_1", "iteration_2"],
    name: str,
    git_commit: str,
    resource_spec_slug: str,
    image: str,
    volume_mounts: list[VesslVolumeMount],
    bounded_label_view_sha256: str,
    dataset_sha256: str,
    split_sha256: str,
    checkpoint_sha256: str,
    environment_lock_sha256: str,
) -> VesslGeneratedJobConfig:
    repository = "https://github.com/Hongjiseung-ROK/MLP-auto-researcher.git"
    steps = [
        ["git", "clone", repository, "/workspace/repo"],
        ["git", "-C", "/workspace/repo", "checkout", "--detach", git_commit],
        ["git", "-C", "/workspace/repo", "rev-parse", "HEAD"],
        [
            "python",
            "-m",
            "pip",
            "install",
            "-r",
            "scripts/colab/ralphthon_mace_replay_requirements.txt",
        ],
        ["python", "scripts/vessl/remote_entrypoint.py", "verify-environment"],
        ["python", "scripts/vessl/remote_entrypoint.py", "attest-gpu"],
        ["python", "scripts/vessl/remote_entrypoint.py", "verify-input-hashes"],
        ["python", "scripts/vessl/remote_entrypoint.py", "run-phase", phase],
        ["python", "scripts/vessl/remote_entrypoint.py", "write-artifact-manifest"],
    ]
    return VesslGeneratedJobConfig(
        phase=phase,
        name=name,
        git_commit=git_commit,
        resource_spec_slug=resource_spec_slug,
        image=image,
        volume_mounts=volume_mounts,
        bounded_label_view_sha256=bounded_label_view_sha256,
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        checkpoint_sha256=checkpoint_sha256,
        environment_lock_sha256=environment_lock_sha256,
        remote_steps=steps,
        submission_argv_template=[
            "vesslctl",
            "job",
            "create",
            "--file",
            "<live-schema-verified-job-config.json>",
        ],
    ).sealed()
