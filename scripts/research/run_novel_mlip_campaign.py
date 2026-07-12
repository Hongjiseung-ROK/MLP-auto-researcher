#!/usr/bin/env python3
"""CLI entrypoint for real-GPU baseline and sequential campaign rounds."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlip_research_agent.research.novel_mlip.runtime import run_baseline, run_round


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["baseline", "round"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--round-index", type=int)
    args = parser.parse_args()
    common = {
        "root": args.root,
        "dataset_path": args.dataset,
        "split_path": args.split,
        "checkpoint_path": args.checkpoint,
        "checkpoint_manifest_path": args.checkpoint_manifest,
        "spec_path": args.spec,
        "authorization_path": args.authorization,
        "provider": args.provider,
    }
    if args.phase == "baseline":
        if args.round_index is not None:
            parser.error("baseline does not accept --round-index")
        state = run_baseline(**common)
    else:
        if args.round_index is None:
            parser.error("round requires --round-index")
        state = run_round(**common, round_index=args.round_index)
    print(
        f"campaign phase complete: round={state.completed_round} "
        f"spec={state.research_spec_sha256}"
    )


if __name__ == "__main__":
    main()
