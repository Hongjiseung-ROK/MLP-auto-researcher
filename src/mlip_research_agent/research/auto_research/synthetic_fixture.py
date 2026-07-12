"""Compatibility imports for the moved synthetic benchmark adapter."""

from mlip_research_agent.research.auto_research.adapters.base import (
    AdapterRunResult,
    BenchmarkAdapter,
)
from mlip_research_agent.research.auto_research.adapters.synthetic import (
    DIVERGENCE_LEARNING_RATE,
    FIXTURE_NAME,
    FIXTURE_VERSION,
    N_SYNTHETIC_RECORDS,
    SyntheticQuadraticAdapter,
    synthetic_validation_residuals,
)

__all__ = [
    "DIVERGENCE_LEARNING_RATE",
    "FIXTURE_NAME",
    "FIXTURE_VERSION",
    "N_SYNTHETIC_RECORDS",
    "AdapterRunResult",
    "BenchmarkAdapter",
    "SyntheticQuadraticAdapter",
    "synthetic_validation_residuals",
]
