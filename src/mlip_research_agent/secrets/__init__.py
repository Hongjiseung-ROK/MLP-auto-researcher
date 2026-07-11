"""Secure secret loading for external scientific data providers.

Only status objects (names and booleans) cross this package boundary; key
values go straight into the process environment and nowhere else.
"""

from mlip_research_agent.secrets.api_env import (
    SUPPORTED_KEY_NAMES,
    ApiEnvStatus,
    load_mp_api_key,
    mp_api_key_env,
)

__all__ = [
    "SUPPORTED_KEY_NAMES",
    "ApiEnvStatus",
    "load_mp_api_key",
    "mp_api_key_env",
]
