"""Build the cu_phase2 qualified-dataset registry entry from pinned raw data.

Usage:
    python scripts/data/qualify_cu_benchmark.py

Pipeline (plan_phase_2.md §4.3): verify raw checksums → parse → normalize →
qualify → write registry artifacts. Promotion to claim-eligible additionally
requires the H1 human approval (recorded separately in
docs/research/approvals.jsonl); this script never grants it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlip_research_agent.data.manifests import NormalizedDataset  # noqa: E402
from mlip_research_agent.data.mlearn import (  # noqa: E402
    MLEARN_LEVEL_OF_THEORY,
    parse_mlearn_file,
)
from mlip_research_agent.data.qualification import (  # noqa: E402
    QUALIFICATION_REPORT_NAME,
    qualify_dataset,
)
from mlip_research_agent.data.registry import (  # noqa: E402
    NORMALIZED_DATASET_NAME,
    NORMALIZED_MANIFEST_NAME,
    RAW_MANIFEST_NAME,
    SOURCE_NAME,
    NormalizedManifest,
    RawFileRecord,
    SourceDescriptor,
    verify_raw_files,
)

DATASET_ID = "cu_phase2"
DUE_DILIGENCE = REPO_ROOT / "docs" / "research" / "dataset_due_diligence_cu.json"
DATASET_DIR = REPO_ROOT / "data_registry" / "datasets" / DATASET_ID


def main() -> int:
    facts = json.loads(DUE_DILIGENCE.read_text())
    source = SourceDescriptor(
        dataset_id=DATASET_ID,
        citation="Zuo et al. 2019, arXiv:1906.08888 (mlearn benchmark, Cu subset)",
        canonical_repo=facts["source_repo"]["current_url"],
        pinned_commit=facts["pinned_commit"],
        license_spdx=facts["license_spdx"],
        redistribution_allowed=facts["redistribution_allowed"],
        files=[
            RawFileRecord(
                url=f["url"],
                local_path=f["path"],
                sha256=f["sha256"],
                size_bytes=f["size_bytes"],
            )
            for f in facts["files"]
        ],
    )
    verify_raw_files(source, REPO_ROOT)
    print("raw checksums: OK (fail-closed)")

    raw_dir = REPO_ROOT / f"data/raw/mlearn/{source.pinned_commit}"
    configs = parse_mlearn_file(raw_dir / "training.json", dataset_id=DATASET_ID)
    configs += parse_mlearn_file(raw_dir / "test.json", dataset_id=DATASET_ID)
    dataset = NormalizedDataset(
        dataset_id=DATASET_ID,
        level_of_theory=MLEARN_LEVEL_OF_THEORY,
        configurations=configs,
    )

    expected_hist = {g["name"]: g["count"] for g in facts["groups"]}
    report = qualify_dataset(
        dataset,
        expected_total=facts["n_configurations"]["total"],
        expected_group_histogram=expected_hist,
    )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / SOURCE_NAME).write_text(
        json.dumps(source.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
    )
    (DATASET_DIR / RAW_MANIFEST_NAME).write_text(
        json.dumps(
            [f.model_dump(mode="json") for f in source.files], sort_keys=True, indent=2
        )
        + "\n"
    )
    (DATASET_DIR / "license.txt").write_bytes((raw_dir / "LICENSE").read_bytes())
    file_sha = dataset.save(DATASET_DIR / NORMALIZED_DATASET_NAME)
    manifest = NormalizedManifest.from_dataset(dataset, file_sha)
    manifest.save(DATASET_DIR / NORMALIZED_MANIFEST_NAME)
    report.save(DATASET_DIR / QUALIFICATION_REPORT_NAME)

    for check in report.checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"  [{mark}] {check.name}" + (f" — {check.detail}" if check.detail else ""))
    print(f"configurations: {report.n_configurations}")
    print(f"split units:    {report.split_unit_count}")
    print(f"groups:         {report.group_histogram}")
    print(f"dataset content sha256: {manifest.dataset_content_sha256}")
    print(f"qualification passed:   {report.passed}")
    print("NOTE: claim-eligible use still requires the H1 approval record.")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
