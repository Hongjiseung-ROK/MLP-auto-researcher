#!/usr/bin/env python3
"""Build the self-excluding manifest for the anonymous Ralphthon submission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL = REPO_ROOT / "paper/final"
INTERNAL = REPO_ROOT / "paper/internal"
OUTPUT = FINAL / "submission_manifest.json"
REQUIRED_FILES = (
    "paper.tex",
    "paper.pdf",
    "references.bib",
    "appendix.tex",
    "research_spec.json",
    "agent_workflow.md",
    "experiment_ledger.jsonl",
    "evidence_map.json",
    "claims.json",
    "figure_manifest.json",
    "citation_audit.json",
    "self_review.json",
    "external_review.json",
    "claim_audit.json",
    "format_audit.json",
    "reproducibility_statement.md",
    "limitations.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (FINAL / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required submission files missing: {missing}")
    claim_audit = json.loads((FINAL / "claim_audit.json").read_text())
    format_audit = json.loads((FINAL / "format_audit.json").read_text())
    if claim_audit["status"] != "pass" or format_audit["status"] != "pass":
        raise ValueError("claim and format audits must pass before manifest freeze")

    public_files = sorted(
        path
        for path in FINAL.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and path.suffix not in {".aux", ".bbl", ".blg", ".log", ".out"}
    )
    internal_files = [
        INTERNAL / "full_paper.tex",
        INTERNAL / "full_paper.pdf",
        INTERNAL / "technical_appendix.tex",
        INTERNAL / "technical_appendix.pdf",
        INTERNAL / "private_evidence_map.json",
    ]
    if not all(path.is_file() for path in internal_files):
        raise FileNotFoundError("internal paper bundle is incomplete")

    code_files = [
        REPO_ROOT / "src/mlip_research_agent/research/trace_fault_audit.py",
        REPO_ROOT / "scripts/research/run_trace_fault_audit.py",
        REPO_ROOT / "scripts/research/build_trace_audit_analysis.py",
        REPO_ROOT / "scripts/research/audit_paper_bundle.py",
        REPO_ROOT / "scripts/research/build_submission_manifest.py",
        REPO_ROOT / "paper/figures/render_figures.R",
        REPO_ROOT / "paper/figures/renv.lock",
        REPO_ROOT / "configs/research/free_ralph_campaign.yaml",
    ]
    manifest = {
        "schema_version": "1.0.0",
        "submission": {
            "event": "Ralphthon @ ICML",
            "track": "General Track 1",
            "artifact": "anonymous 2-4 page short paper plus self-review",
            "paper_pages": format_audit["checks"]["page_count"],
            "scientific_status": "infrastructure_only",
            "claim_ceiling": "T2",
        },
        "status": "ready_with_disclosed_scope_limitations",
        "required_files_present": True,
        "required_files": [entry(FINAL / name, FINAL) for name in REQUIRED_FILES],
        "public_bundle": [entry(path, FINAL) for path in public_files],
        "code_and_configuration_identities": [
            entry(path, REPO_ROOT) for path in code_files
        ],
        "internal_bundle_not_for_anonymous_submission": [
            entry(path, REPO_ROOT) for path in internal_files
        ],
        "audit_status": {
            "claim_audit": claim_audit["status"],
            "format_audit": format_audit["status"],
            "citation_audit": json.loads((FINAL / "citation_audit.json").read_text())[
                "status"
            ],
            "external_adversarial_recommendation_preserved": True,
        },
        "manifest_self_hash_policy": (
            "submission_manifest.json is excluded from its own file list and hash "
            "to avoid a recursive digest."
        ),
        "release_notes": [
            "The paper is anonymous, but publishing the work branch or pull request "
            "can reveal authorship; repository publication and anonymous conference "
            "submission are distinct disclosure channels.",
            "The remote evidence packet is a sanitized extract. Original source hashes "
            "are verified through the internal private evidence map.",
            "No general robustness, MLIP improvement, active-learning efficiency, or "
            "publication-level scientific claim is released."
        ],
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "files": len(public_files)}))


if __name__ == "__main__":
    main()
