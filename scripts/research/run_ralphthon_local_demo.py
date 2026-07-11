#!/usr/bin/env python3
"""Run the Ralphthon local Auto Research demonstration and grade its trace.

Local CPU only: a deterministic synthetic fixture, two connected iterations,
no frozen test data, no hidden labels, no Cu partitions, no remote compute,
no claims. Usage:

    python scripts/research/run_ralphthon_local_demo.py [--run-id RUN_ID]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlip_research_agent.research.auto_research import (  # noqa: E402
    AutoResearchController,
    LocalDemoConfig,
    MutationPolicy,
    SyntheticAggregateEvaluator,
    SyntheticQuadraticAdapter,
)
from mlip_research_agent.verification.trace_grader import grade_trace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Override the run id")
    args = parser.parse_args()

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    demo = LocalDemoConfig.load(REPO_ROOT / "configs/research/ralphthon_local_demo.yaml")
    policy = MutationPolicy.load(REPO_ROOT / demo.mutation_policy_path)
    run_id = args.run_id or f"ralphthon-demo-{git_commit[:8]}"
    run_dir = REPO_ROOT / demo.output_root / run_id

    controller = AutoResearchController(
        run_id=run_id,
        run_dir=run_dir,
        objective=demo.objective.sealed(),
        mutation_policy=policy,
        adapter=SyntheticQuadraticAdapter(),
        evaluator=SyntheticAggregateEvaluator(),
        acceptance=demo.acceptance,
        base_config=demo.base_config,
        fixture_seed=demo.fixture_seed,
        git_commit=git_commit,
    )
    final_state = controller.run()
    print(f"run_id: {run_id}")
    print(f"run_dir: {run_dir}")
    print(f"terminal state: {final_state.value}")

    report = grade_trace(run_dir, demo.acceptance)
    print(f"trace grade: {'PASS' if report.passed else 'FAIL'} "
          f"({report.n_iterations_graded} iterations graded)")
    for violation in report.violations:
        print(f"  VIOLATION [{violation.check}] {violation.location}: {violation.detail}")
    (run_dir / "trace_grade.json").write_text(report.model_dump_json(indent=2) + "\n")
    return 0 if report.passed and final_state.value == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
