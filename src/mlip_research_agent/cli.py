"""Command-line interface: validate campaigns, run them, emit env reports."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from mlip_research_agent.provenance.bundle import build_provenance
from mlip_research_agent.provenance.environment_report import generate_environment_report
from mlip_research_agent.provenance.report import generate_run_report
from mlip_research_agent.runtime.executor import WorkflowExecutor
from mlip_research_agent.schemas.campaign import load_campaign
from mlip_research_agent.workflows.compiler import compile_campaign


def _cmd_validate(args: argparse.Namespace) -> int:
    campaign = load_campaign(Path(args.campaign))
    workflow = compile_campaign(campaign)
    print(f"campaign {campaign.name!r} valid (hash {campaign.content_hash()[:12]})")
    print(f"workflow {workflow.name!r} valid: {' -> '.join(workflow.topological_order())}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    campaign = load_campaign(Path(args.campaign))
    workflow = compile_campaign(campaign)

    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.is_dir():
            print(f"error: resume directory not found: {run_dir}", file=sys.stderr)
            return 2
        run_id = run_dir.name
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{campaign.name}-seed{campaign.seed}-{stamp}"
        run_dir = Path(args.output_root) / run_id

    executor = WorkflowExecutor(
        workflow,
        run_dir=run_dir,
        run_id=run_id,
        max_retries_per_step=campaign.budget.max_retries_per_step,
    )
    result = executor.execute(resume=bool(args.resume))
    build_provenance(run_dir, campaign, workflow, command=sys.argv)
    report_path = generate_run_report(run_dir)

    print(f"run {run_id}: {result.status}")
    if result.failures:
        for failure in result.failures:
            print(
                f"  failure {failure.failure_id} [{failure.failure_class.value}] "
                f"-> {failure.recommended_action.value}"
            )
    print(f"report: {report_path}")
    return 0 if result.status == "completed" else 1


def _cmd_env_report(args: argparse.Namespace) -> int:
    env_file = Path(args.environment_file) if args.environment_file else None
    path = generate_environment_report(
        Path(args.output),
        environment_file=env_file,
        commands_executed=args.record_command or [],
    )
    print(f"environment report: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlip-agent",
        description="Autonomous active-learning scientist for MLIPs (mock vertical slice)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="validate a campaign YAML and its workflow")
    p_validate.add_argument("--campaign", required=True)
    p_validate.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="execute a campaign")
    p_run.add_argument("--campaign", required=True)
    p_run.add_argument("--output-root", default="artifacts/runs")
    p_run.add_argument("--resume", default=None, help="existing run directory to resume")
    p_run.set_defaults(func=_cmd_run)

    p_env = sub.add_parser("env-report", help="write the bootstrap environment report")
    p_env.add_argument("--output", default="artifacts/bootstrap/environment-report.json")
    p_env.add_argument("--environment-file", default="environment.yml")
    p_env.add_argument("--record-command", action="append", default=None)
    p_env.set_defaults(func=_cmd_env_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
