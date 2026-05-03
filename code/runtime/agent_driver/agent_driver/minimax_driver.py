"""MiniMax-family driver specialization."""

from __future__ import annotations

from dataclasses import dataclass

from .llm_driver import LLMDriver, LLMDriverConfig


@dataclass(frozen=True, slots=True)
class MiniMaxDriverConfig(LLMDriverConfig):
    pass


class MiniMaxDriver(LLMDriver):
    def __init__(self, config: MiniMaxDriverConfig, *, generate_fn):
        super().__init__(config, generate_fn=generate_fn)
