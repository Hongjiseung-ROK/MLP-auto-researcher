"""Benchmark-neutral Auto Research execution adapters."""

from mlip_research_agent.research.auto_research.adapters.base import (
    AdapterRunResult,
    BenchmarkAdapter,
)
from mlip_research_agent.research.auto_research.adapters.mace_phase2 import MACEPhase2Adapter
from mlip_research_agent.research.auto_research.adapters.synthetic import (
    SyntheticQuadraticAdapter,
)

__all__ = [
    "AdapterRunResult",
    "BenchmarkAdapter",
    "MACEPhase2Adapter",
    "SyntheticQuadraticAdapter",
]
