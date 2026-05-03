"""Qwen reference LLM driver implementation."""

from __future__ import annotations

from dataclasses import dataclass

from .llm_driver import GenerateFn, LLMDriver, LLMDriverConfig


@dataclass(frozen=True, slots=True)
class QwenDriverConfig(LLMDriverConfig):
    pass


class QwenDriver(LLMDriver):
    def __init__(self, config: QwenDriverConfig, *, generate_fn: GenerateFn) -> None:
        super().__init__(config, generate_fn=generate_fn)
