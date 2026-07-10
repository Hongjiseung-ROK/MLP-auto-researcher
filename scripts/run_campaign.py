#!/usr/bin/env python
"""Thin wrapper around the mlip-agent CLI for direct script execution."""

import sys

from mlip_research_agent.cli import main

DEFAULT_ARGS = ["run", "--campaign", "configs/example_campaign.yaml"]

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_ARGS))
