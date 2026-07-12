from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_research_agent.compute.vessl_job_config import build_dry_run_job_config
from mlip_research_agent.compute.vessl_replay import (
    VesslReplayError,
    VesslReplayLedger,
    atomic_publish,
    build_artifact_manifest,
    validate_cleanup,
    validate_job_pair,
    validate_review_ordering,
    validate_transport_inputs,
    verify_artifact_manifest,
)
from mlip_research_agent.compute.vessl_schemas import (
    VesslCleanupRecord,
    VesslJobRequest,
    VesslJobStatus,
    VesslOptimizerReceipt,
    VesslReviewBundleReceipt,
    VesslStorageExposure,
    VesslVolumeMount,
)


def _mount() -> VesslVolumeMount:
    return VesslVolumeMount(
        volume_type="object", volume_slug="runs", mount_path="/runs"
    ).sealed()


def _job1(**updates: Any) -> VesslJobRequest:
    payload: dict[str, Any] = {
        "phase": "iteration_1",
        "organization": "org",
        "team": "team",
        "cluster": "cluster",
        "resource_spec_slug": "a100-1",
        "gpu_type": "A100 SXM 80GB",
        "gpu_count": 1,
        "image": "image@sha256:" + "1" * 64,
        "expected_max_duration_minutes": 30,
        "volume_mounts": [_mount()],
        "timeout_behavior": "Fail and inspect the terminal Job state.",
        "cleanup_action": "Report storage after both Jobs become terminal.",
        "git_commit": "2" * 40,
        "job_config_sha256": "3" * 64,
        "job_config_schema_verified": False,
        "bounded_label_view_sha256": "4" * 64,
        "dataset_sha256": "5" * 64,
        "split_sha256": "6" * 64,
        "checkpoint_sha256": "7" * 64,
        "environment_lock_sha256": "8" * 64,
    }
    payload.update(updates)
    return VesslJobRequest.model_validate(payload).sealed()


def _receipt(job1: VesslJobRequest) -> VesslOptimizerReceipt:
    return VesslOptimizerReceipt(
        phase="iteration_1",
        job_request_sha256=job1.content_sha256,
        operation_id="iteration-001",
        artifact_manifest_sha256="9" * 64,
        bounded_label_view_sha256=job1.bounded_label_view_sha256,
        dataset_sha256=job1.dataset_sha256,
        split_sha256=job1.split_sha256,
        checkpoint_sha256=job1.checkpoint_sha256,
    ).sealed()


def _reviews(receipt: VesslOptimizerReceipt) -> VesslReviewBundleReceipt:
    return VesslReviewBundleReceipt(
        iteration_1_receipt_sha256=receipt.content_sha256,
        mlip_scientist_sha256="a" * 64,
        active_learning_scientist_sha256="b" * 64,
        scientific_auditor_sha256="c" * 64,
        synthesis_sha256="d" * 64,
        proposal_2_sha256="e" * 64,
    ).sealed()


def _job2(
    job1: VesslJobRequest,
    receipt: VesslOptimizerReceipt,
    reviews: VesslReviewBundleReceipt,
    **updates: Any,
) -> VesslJobRequest:
    payload = job1.model_dump(mode="json")
    payload.pop("content_sha256")
    payload.update(
        {
            "phase": "iteration_2",
            "job_config_sha256": "f" * 64,
            "iteration_1_receipt_sha256": receipt.content_sha256,
            "review_bundle_sha256": reviews.content_sha256,
        }
    )
    payload.update(updates)
    return VesslJobRequest.model_validate(payload).sealed()


def test_job_pair_and_review_ordering_bind_same_identity() -> None:
    job1 = _job1()
    receipt = _receipt(job1)
    reviews = _reviews(receipt)
    job2 = _job2(job1, receipt, reviews)
    validate_job_pair(job1, job2)
    validate_review_ordering(iteration_1=receipt, reviews=reviews, job2=job2)
    with pytest.raises(VesslReplayError, match="image"):
        validate_job_pair(job1, _job2(job1, receipt, reviews, image="changed"))


def test_generated_job_control_has_required_argument_array_steps() -> None:
    job = _job1()
    config = build_dry_run_job_config(
        phase="iteration_1",
        name="mlip-replay-1",
        git_commit=job.git_commit,
        resource_spec_slug=job.resource_spec_slug,
        image=job.image,
        volume_mounts=job.volume_mounts,
        bounded_label_view_sha256=job.bounded_label_view_sha256,
        dataset_sha256=job.dataset_sha256,
        split_sha256=job.split_sha256,
        checkpoint_sha256=job.checkpoint_sha256,
        environment_lock_sha256=job.environment_lock_sha256,
    )
    assert config.verify_seal()
    assert config.vesslctl_file_schema_verified is False
    assert all(isinstance(step, list) and step for step in config.remote_steps)
    assert config.submission_argv_template[:4] == [
        "vesslctl",
        "job",
        "create",
        "--file",
    ]


def test_iteration1_duplicate_and_ambiguous_execution_are_denied(tmp_path: Path) -> None:
    job1 = _job1()
    ledger = VesslReplayLedger(tmp_path)
    ledger.begin(job1)
    with pytest.raises(VesslReplayError, match="ambiguous"):
        ledger.begin(job1)
    ledger.complete(_receipt(job1))
    with pytest.raises(VesslReplayError, match="duplicate"):
        ledger.begin(job1)


def test_review_bundle_must_follow_and_bind_iteration1() -> None:
    job1 = _job1()
    receipt = _receipt(job1)
    reviews = _reviews(receipt)
    job2 = _job2(job1, receipt, reviews)
    wrong = reviews.model_copy(update={"iteration_1_receipt_sha256": "0" * 64}).sealed()
    with pytest.raises(VesslReplayError, match="does not bind iteration 1"):
        validate_review_ordering(iteration_1=receipt, reviews=wrong, job2=job2)


def test_protected_data_and_unexpected_files_are_rejected(tmp_path: Path) -> None:
    allowed = tmp_path / "bounded_label_view.json"
    allowed.write_text('{"initial_labeled": [], "validation": []}\n')
    validate_transport_inputs(tmp_path, {allowed.name})
    leak = tmp_path / "frozen_test.json"
    leak.write_text("{}\n")
    with pytest.raises(VesslReplayError, match="allowlist mismatch"):
        validate_transport_inputs(tmp_path, {allowed.name})
    leak.unlink()
    allowed.write_text('{"acquisition_pool_labels": [1]}\n')
    with pytest.raises(VesslReplayError, match="protected or secret"):
        validate_transport_inputs(tmp_path, {allowed.name})


def test_artifact_manifest_tampering_and_incomplete_pullback_fail(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    result = staging / "result.json"
    result.write_text('{"claim_eligible": false}\n')
    manifest = build_artifact_manifest(staging, staging / "artifact_manifest.json")
    verify_artifact_manifest(staging, manifest)
    result.write_text("tampered\n")
    with pytest.raises(VesslReplayError, match="hash mismatch"):
        verify_artifact_manifest(staging, manifest)
    result.unlink()
    with pytest.raises(VesslReplayError, match="incomplete"):
        verify_artifact_manifest(staging, manifest)


def test_atomic_publish_only_after_hash_verification(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "result.json").write_text("{}\n")
    build_artifact_manifest(staging, staging / "artifact_manifest.json")
    destination = tmp_path / "published"
    atomic_publish(staging, destination, "artifact_manifest.json")
    assert destination.is_dir() and not staging.exists()


def _status(state: str) -> VesslJobStatus:
    return VesslJobStatus(
        job_slug=f"job-{state}", state=state, raw_payload_sha256="a" * 64
    ).sealed()


def test_cleanup_requires_terminal_jobs_and_reports_surviving_storage() -> None:
    with pytest.raises(ValidationError, match="terminal"):
        VesslCleanupRecord(
            job_statuses=[_status("running")],
            surviving_storage=[],
            checked_at=datetime.now(UTC),
        )
    exposure = VesslStorageExposure(
        storage_type="object",
        storage_slug="object-us",
        volume_slug="run-volume",
        capacity_gb=10,
        hourly_rate=0.1,
        currency="USD",
        active=True,
    ).sealed()
    record = VesslCleanupRecord(
        job_statuses=[_status("succeeded"), _status("failed")],
        surviving_storage=[exposure],
        checked_at=datetime.now(UTC),
    ).sealed()
    assert validate_cleanup(record) == ["run-volume"]
