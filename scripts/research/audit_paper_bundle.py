#!/usr/bin/env python3
"""Audit the anonymous Ralphthon paper bundle and rebuild its PDF in isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = REPO_ROOT / "paper/final"
PRIVATE_MAP = REPO_ROOT / "paper/internal/private_evidence_map.json"
TIER_ORDER = {f"T{index}": index for index in range(6)}
BANNED_PUBLIC_TOKENS = (
    "/Users/",
    "hongjiseung",
    "ec2d804",
    "13ecd09",
    "1f3feba",
    "research/free-ralph-paper",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit_claims(output: Path) -> dict[str, Any]:
    claims = _json(FINAL_DIR / "claims.json")
    evidence = _json(FINAL_DIR / "evidence_map.json")
    private = _json(PRIVATE_MAP)
    remote_packet_path = FINAL_DIR / "evidence/remote_trace_A.json"
    remote_packet_text = remote_packet_path.read_text()

    entries = {entry["artifact_id"]: entry for entry in evidence["entries"]}
    claim_ids = [claim["claim_id"] for claim in claims["claims"]]
    checks: dict[str, Any] = {}
    checks["unique_claim_ids"] = len(claim_ids) == len(set(claim_ids))
    checks["unique_evidence_ids"] = len(entries) == len(evidence["entries"])
    checks["claim_evidence_resolves"] = all(
        artifact_id in entries
        for claim in claims["claims"]
        for artifact_id in claim["evidence"]
    )
    ceiling = TIER_ORDER[str(claims["claim_ceiling"])]
    checks["claim_tiers_within_ceiling"] = all(
        TIER_ORDER[str(claim["tier"])] <= ceiling for claim in claims["claims"]
    )

    local_hash_checks: dict[str, bool] = {}
    for artifact_id, entry in entries.items():
        if "sha256" not in entry:
            continue
        locator = str(entry["public_locator"])
        path = FINAL_DIR / locator
        local_hash_checks[artifact_id] = path.is_file() and _sha256(path) == entry["sha256"]
    checks["local_public_hashes"] = local_hash_checks

    private_paths = private["remote_artifacts"]
    source_map = {
        "remote-h1-approval": "h1_approval",
        "remote-attestation": "compute_attestation",
        "remote-environment": "environment",
        "remote-checkpoint-verification": "checkpoint_verification",
        "remote-dataset-verification": "dataset_verification",
        "remote-iteration-001-evaluation": "iteration_001_evaluation",
        "remote-iteration-001-decision": "iteration_001_decision",
        "remote-op-001": "iteration_001_operation",
        "remote-iteration-002-evaluation": "iteration_002_evaluation",
        "remote-iteration-002-decision": "iteration_002_decision",
        "remote-op-002": "iteration_002_operation",
        "remote-trace-grade": "trace_grade",
        "remote-completion-status": "completion_status",
        "remote-round-state": "round_state",
        "remote-cleanup-confirmation": "cleanup_confirmation",
    }
    remote_hash_checks: dict[str, bool] = {}
    for artifact_id, private_key in source_map.items():
        entry = entries[artifact_id]
        source_hash = str(entry["source_sha256"])
        source_path = REPO_ROOT / str(private_paths[private_key])
        remote_hash_checks[artifact_id] = (
            source_path.is_file()
            and _sha256(source_path) == source_hash
            and source_hash in remote_packet_text
        )
    checks["remote_private_hashes_and_public_bindings"] = remote_hash_checks
    checks["remote_packet_hash"] = (
        _sha256(remote_packet_path) == evidence["resolver"]["remote_packet_sha256"]
    )

    ledger_records = [
        json.loads(line)
        for line in (FINAL_DIR / "experiment_ledger.jsonl").read_text().splitlines()
        if line
    ]
    corrupt = [record for record in ledger_records if record["expected_corrupt"]]
    clean = [record for record in ledger_records if not record["expected_corrupt"]]
    families = sorted({str(record["fault_family"]) for record in corrupt})

    def macro(detector: str) -> float:
        recalls = []
        for family in families:
            family_records = [record for record in corrupt if record["fault_family"] == family]
            recalls.append(
                sum(bool(record[detector]) for record in family_records) / len(family_records)
            )
        return sum(recalls) / len(recalls)

    recomputed = {
        "n_records": len(ledger_records),
        "n_corrupt": len(corrupt),
        "n_clean": len(clean),
        "n_fault_families": len(families),
        "semantic_detected": sum(bool(record["semantic_grader_detected"]) for record in corrupt),
        "manifest_detected": sum(bool(record["manifest_only_detected"]) for record in corrupt),
        "semantic_macro_recall": macro("semantic_grader_detected"),
        "manifest_macro_recall": macro("manifest_only_detected"),
        "semantic_clean_rejected": sum(
            not bool(record["semantic_grader_accepted"]) for record in clean
        ),
    }
    recomputed["absolute_macro_recall_gain"] = (
        recomputed["semantic_macro_recall"] - recomputed["manifest_macro_recall"]
    )
    checks["headline_recomputed"] = recomputed
    checks["headline_matches_claims"] = (
        recomputed["n_records"] == 52
        and recomputed["n_corrupt"] == 48
        and recomputed["n_clean"] == 4
        and recomputed["n_fault_families"] == 12
        and recomputed["semantic_detected"] == 48
        and recomputed["manifest_detected"] == 4
        and recomputed["semantic_clean_rejected"] == 0
        and abs(float(recomputed["absolute_macro_recall_gain"]) - 11 / 12) < 1e-12
    )

    public_text_suffixes = {".tex", ".md", ".json", ".jsonl", ".csv", ".bib", ".sty", ".bst"}
    public_files = sorted(
        path
        for path in FINAL_DIR.rglob("*")
        if path.is_file() and path.suffix in public_text_suffixes
    )
    public_text = "\n".join(path.read_text(errors="replace") for path in public_files)
    checks["public_text_anonymous"] = not any(
        token.lower() in public_text.lower() for token in BANNED_PUBLIC_TOKENS
    )
    checks["public_text_files_scanned"] = len(public_files)
    checks["claim_ceiling"] = claims["claim_ceiling"]
    checks["scientific_status"] = claims["scientific_status"]

    boolean_results = [
        checks["unique_claim_ids"],
        checks["unique_evidence_ids"],
        checks["claim_evidence_resolves"],
        checks["claim_tiers_within_ceiling"],
        all(local_hash_checks.values()),
        all(remote_hash_checks.values()),
        checks["remote_packet_hash"],
        checks["headline_matches_claims"],
        checks["public_text_anonymous"],
    ]
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "pass" if all(boolean_results) else "fail",
        "auditor": "scripts/research/audit_paper_bundle.py",
        "checks": checks,
        "claim_ceiling_decision": {
            "tier": "T2",
            "class": "infrastructure_only",
            "publication_scientific_claim_released": False,
        },
        "unresolved_blockers": [],
    }
    _write_json(output, result)
    return result


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def audit_format(output: Path, visual_inspection_passed: bool) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="anonymous-paper-build-") as temporary:
        build = Path(temporary) / "submission"
        shutil.copytree(FINAL_DIR, build)
        for suffix in ("aux", "bbl", "blg", "log", "out"):
            for path in build.glob(f"*.{suffix}"):
                path.unlink()
        _run(
            [
                "pdflatex",
                "-no-shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "paper.tex",
            ],
            build,
        )
        _run(["bibtex", "paper"], build)
        _run(
            [
                "pdflatex",
                "-no-shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "paper.tex",
            ],
            build,
        )
        _run(
            [
                "pdflatex",
                "-no-shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "paper.tex",
            ],
            build,
        )
        built_pdf = build / "paper.pdf"
        shutil.copy2(built_pdf, FINAL_DIR / "paper.pdf")
        log = (build / "paper.log").read_text(errors="replace")
        pdf_info = _run(["pdfinfo", "paper.pdf"], build).stdout
        font_info = _run(["pdffonts", "paper.pdf"], build).stdout
        strings = _run(["strings", "paper.pdf"], build).stdout
        render_dir = build / "render"
        render_dir.mkdir()
        _run(["pdftoppm", "-png", "-r", "120", "paper.pdf", str(render_dir / "page")], build)
        rendered_pages = sorted(render_dir.glob("page-*.png"))

        page_match = re.search(r"^Pages:\s+(\d+)$", pdf_info, re.MULTILINE)
        pages = int(page_match.group(1)) if page_match else -1
        page_size_match = re.search(r"^Page size:\s+(.+)$", pdf_info, re.MULTILINE)
        page_size = page_size_match.group(1) if page_size_match else "unknown"
        font_rows = [line.split() for line in font_info.splitlines()[2:] if line.strip()]
        fonts_embedded = bool(font_rows) and all(
            len(row) >= 7 and row[-5] == "yes" for row in font_rows
        )
        undefined = bool(
            re.search(
                r"undefined citations|undefined references|There were undefined|"
                r"Citation .* undefined|Reference .* undefined",
                log,
                re.IGNORECASE,
            )
        )
        overfull = "Overfull \\hbox" in log or "Overfull \\vbox" in log
        leaked_tokens = [
            token for token in BANNED_PUBLIC_TOKENS if token.lower() in strings.lower()
        ]
        style_hashes = {
            "icml2026.sty": _sha256(FINAL_DIR / "icml2026.sty"),
            "icml2026.bst": _sha256(FINAL_DIR / "icml2026.bst"),
        }
        checks = {
            "compiled_with_shell_escape_disabled": True,
            "page_count": pages,
            "page_count_in_required_range": 2 <= pages <= 4,
            "page_size": page_size,
            "us_letter": "612 x 792" in page_size,
            "undefined_citations_or_references": undefined,
            "overfull_boxes": overfull,
            "fonts_embedded": fonts_embedded,
            "font_count": len(font_rows),
            "rendered_page_count": len(rendered_pages),
            "manual_visual_inspection_passed": visual_inspection_passed,
            "pdf_path_or_identity_leaks": leaked_tokens,
            "anonymous_source_scan_passed": not leaked_tokens,
            "official_style_hashes": style_hashes,
            "official_style_hashes_match_frozen_sources": style_hashes
            == {
                "icml2026.sty": "7cdcf90f6a59c5219e7f15c88f7ed09fcaf598dad91e6cdddc4dc3cb0e397a95",
                "icml2026.bst": "0ec3d5eb9b02efb7e0b44a32f3775882f42a743d0bdc618f34e6936309b98764",
            },
            "figure_pdfs_present": all(
                path.is_file()
                for path in (
                    FINAL_DIR / "figures/evidence_workflow.pdf",
                    FINAL_DIR / "figures/fault_detection_recall.pdf",
                )
            ),
        }
        pass_values = [
            checks["compiled_with_shell_escape_disabled"],
            checks["page_count_in_required_range"],
            checks["us_letter"],
            not checks["undefined_citations_or_references"],
            not checks["overfull_boxes"],
            checks["fonts_embedded"],
            checks["rendered_page_count"] == pages,
            checks["manual_visual_inspection_passed"],
            not checks["pdf_path_or_identity_leaks"],
            checks["official_style_hashes_match_frozen_sources"],
            checks["figure_pdfs_present"],
        ]
        result: dict[str, Any] = {
            "schema_version": "1.0.0",
            "status": "pass" if all(pass_values) else "fail",
            "auditor": "scripts/research/audit_paper_bundle.py",
            "paper_pdf_sha256": _sha256(FINAL_DIR / "paper.pdf"),
            "checks": checks,
            "nonblocking_warnings": [
                "The ICML style emits a benign empty-anchor warning in anonymous mode.",
                "Underfull boxes and the expected disabled-shell-escape epstopdf "
                "warning may appear."
            ],
        }
        _write_json(output, result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-output", type=Path, default=FINAL_DIR / "claim_audit.json")
    parser.add_argument("--format-output", type=Path, default=FINAL_DIR / "format_audit.json")
    parser.add_argument("--visual-inspection-passed", action="store_true")
    args = parser.parse_args()
    claim_result = audit_claims(args.claim_output)
    format_result = audit_format(args.format_output, args.visual_inspection_passed)
    print(json.dumps({"claim": claim_result["status"], "format": format_result["status"]}))


if __name__ == "__main__":
    main()
