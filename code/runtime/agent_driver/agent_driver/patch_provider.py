"""Patch-provider specialization for SWE reference traffic."""

from __future__ import annotations

from .llm_driver import LLMDriver, LLMDriverConfig


class PatchProviderDriver(LLMDriver):
    def __init__(self, config: LLMDriverConfig, *, generate_fn):
        super().__init__(config, generate_fn=generate_fn)
