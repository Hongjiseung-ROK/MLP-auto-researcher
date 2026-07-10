#!/usr/bin/env python
"""Generate artifacts/bootstrap/environment-report.json for the current env."""

import sys

from mlip_research_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["env-report", *sys.argv[1:]]))
