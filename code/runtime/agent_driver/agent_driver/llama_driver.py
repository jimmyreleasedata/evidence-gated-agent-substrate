"""Llama-family driver specialization."""

from __future__ import annotations

from dataclasses import dataclass

from .llm_driver import LLMDriver, LLMDriverConfig


@dataclass(frozen=True, slots=True)
class LlamaDriverConfig(LLMDriverConfig):
    pass


class LlamaDriver(LLMDriver):
    def __init__(self, config: LlamaDriverConfig, *, generate_fn):
        super().__init__(config, generate_fn=generate_fn)
