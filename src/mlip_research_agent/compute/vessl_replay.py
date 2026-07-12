"""Two-sequential-Job, exactly-once VESSL MLIP replay contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.compute.vessl_schemas import (
    VesslCleanupRecord,
    VesslJobRequest,
    VesslOptimizerReceipt,
    VesslReviewBundleReceipt,
)
from mlip_research_agent.research.auto_research.validators import require_sealed

FORBIDDEN_PATH_MARKERS = (
    ".env",
    "api.env",
    "normalized_dataset",
    "frozen_test",
    "stress_test",
    "acquisition_pool_labels",
)
FORBIDDEN_JSON_KEYS = {
    "frozen_test",
    "frozen_test_ids",
    "stress_labels",
    "acquisition_pool_labels",
    "vesslctl_access_token",
    "sidecar_initial_access_token",
}


class VesslReplayError(RuntimeError):
    pass


def validate_job_pair(job1: VesslJobRequest, job2: VesslJobRequest) -> None:
    require_sealed(job1, "VESSL Job 1 request")
    require_sealed(job2, "VESSL Job 2 request")
    if job1.phase != "iteration_1" or job2.phase != "iteration_2":
        raise VesslReplayError("VESSL replay requires Job 1 then Job 2")
    for field in (
        "organization",
        "team",
        "cluster",
        "resource_spec_slug",
        "gpu_type",
        "gpu_count",
        "image",
        "git_commit",
        "bounded_label_view_sha256",
        "dataset_sha256",
        "split_sha256",
        "checkpoint_sha256",
        "environment_lock_sha256",
    ):
        if getattr(job1, field) != getattr(job2, field):
            raise VesslReplayError(f"VESSL Job identity mismatch: {field}")


def validate_review_ordering(
    *,
    iteration_1: VesslOptimizerReceipt,
    reviews: VesslReviewBundleReceipt,
    job2: VesslJobRequest,
) -> None:
    require_sealed(iteration_1, "iteration-1 optimizer receipt")
    require_sealed(reviews, "review bundle receipt")
    require_sealed(job2, "VESSL Job 2 request")
    if iteration_1.phase != "iteration_1":
        raise VesslReplayError("review bundle must follow iteration 1")
    if reviews.iteration_1_receipt_sha256 != iteration_1.content_sha256:
        raise VesslReplayError("review bundle does not bind iteration 1")
    if job2.iteration_1_receipt_sha256 != iteration_1.content_sha256:
        raise VesslReplayError("Job 2 does not bind iteration 1")
    if job2.review_bundle_sha256 != reviews.content_sha256:
        raise VesslReplayError("Job 2 does not bind the sealed review bundle")


class VesslReplayLedger:
    """Append-only local receipt ledger; ambiguous starts are never retried."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _started(self, phase: str) -> Path:
        return self.root / f"{phase}.started.json"

    def _completed(self, phase: str) -> Path:
        return self.root / f"{phase}.completed.json"

    def begin(self, request: VesslJobRequest) -> None:
        require_sealed(request, "VESSL Job request")
        started = self._started(request.phase)
        completed = self._completed(request.phase)
        if completed.exists():
            raise VesslReplayError(f"duplicate {request.phase} execution denied")
        if started.exists():
            raise VesslReplayError(
                f"ambiguous {request.phase} optimizer execution; retry denied"
            )
        started.write_text(request.canonical_text())

    def complete(self, receipt: VesslOptimizerReceipt) -> None:
        require_sealed(receipt, "VESSL optimizer receipt")
        started = self._started(receipt.phase)
        completed = self._completed(receipt.phase)
        if not started.is_file() or completed.exists():
            raise VesslReplayError("optimizer completion has no unique started operation")
        request = VesslJobRequest.model_validate_json(started.read_text())
        if request.content_sha256 != receipt.job_request_sha256:
            raise VesslReplayError("optimizer receipt does not bind its Job request")
        completed.write_text(receipt.canonical_text())


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        found = {str(key).casefold() for key in value}
        for item in value.values():
            found.update(_walk_keys(item))
        return found
    if isinstance(value, list):
        list_found: set[str] = set()
        for item in value:
            list_found.update(_walk_keys(item))
        return list_found
    return set()


def validate_transport_inputs(root: Path, allowed_relative_paths: set[str]) -> None:
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != allowed_relative_paths:
        unexpected = sorted(actual - allowed_relative_paths)
        missing = sorted(allowed_relative_paths - actual)
        raise VesslReplayError(
            f"transport input allowlist mismatch: unexpected={unexpected} missing={missing}"
        )
    for relative in sorted(actual):
        folded = relative.casefold()
        if any(marker in folded for marker in FORBIDDEN_PATH_MARKERS):
            raise VesslReplayError(f"forbidden transport input path: {relative}")
        path = root / relative
        if path.suffix == ".json":
            try:
                payload: object = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise VesslReplayError(f"invalid JSON transport input: {relative}") from exc
            overlap = _walk_keys(payload) & FORBIDDEN_JSON_KEYS
            if overlap:
                raise VesslReplayError(
                    f"protected or secret JSON keys in {relative}: {sorted(overlap)}"
                )


def build_artifact_manifest(root: Path, output: Path) -> Path:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != output:
            entries.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    output.write_text(
        json.dumps(
            {"schema_version": "1.0.0", "files": entries},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output


def verify_artifact_manifest(root: Path, manifest: Path) -> None:
    payload = json.loads(manifest.read_text())
    entries = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise VesslReplayError("artifact manifest has no file list")
    expected: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise VesslReplayError("artifact manifest entry is not an object")
        relative = entry.get("relative_path")
        digest = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise VesslReplayError("artifact manifest entry is malformed")
        expected[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    if actual != set(expected):
        raise VesslReplayError("artifact pull-back is incomplete or contains extra files")
    for relative, digest in expected.items():
        if sha256_file(root / relative) != digest:
            raise VesslReplayError(f"artifact hash mismatch: {relative}")


def atomic_publish(staging: Path, destination: Path, manifest_name: str) -> None:
    if destination.exists():
        raise VesslReplayError("artifact publication destination already exists")
    verify_artifact_manifest(staging, staging / manifest_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)


def validate_cleanup(record: VesslCleanupRecord) -> list[str]:
    require_sealed(record, "VESSL cleanup record")
    return [
        exposure.volume_slug or exposure.storage_slug or "unidentified-storage"
        for exposure in record.surviving_storage
        if exposure.active
    ]
