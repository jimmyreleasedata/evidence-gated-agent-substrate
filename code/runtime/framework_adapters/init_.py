"""Framework adapter declarations for Phase 1/2 readiness and Phase 6 demos."""

from .base import REQUIRED_HOOKS, FrameworkAdapterDeclaration
from .trl_adapter import TRL_ADAPTER_DECLARATION
from .verl_adapter import VERL_ADAPTER_DECLARATION

__all__ = [
    "FrameworkAdapterDeclaration",
    "REQUIRED_HOOKS",
    "TRL_ADAPTER_DECLARATION",
    "VERL_ADAPTER_DECLARATION",
]
