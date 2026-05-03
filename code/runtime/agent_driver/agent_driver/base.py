"""Shared agent/driver contract primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from typing import Any


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DriverMetadata:
    driver_id: str
    driver_type: str
    driver_version: str
    model_backend: str
    model_id: str
    model_version: str
    model_path: str
    tokenizer_path: str
    policy_version: str
    prompt_template: str
    action_parser_version: str
    budget: int
    seed: int
    model_family: str = "unknown"
    backend_engine: str = ""
    backend_version: str = ""
    model_path_or_hf_id: str = ""
    model_revision: str = ""

    @property
    def prompt_template_hash(self) -> str:
        return _hash_text(self.prompt_template)

    @property
    def model_path_hash(self) -> str:
        return _hash_text(self.model_path)

    @property
    def tokenizer_hash(self) -> str:
        return _hash_text(self.tokenizer_path)

    def to_row(self) -> dict[str, Any]:
        return {
            "driver_id": self.driver_id,
            "driver_type": self.driver_type,
            "driver_version": self.driver_version,
            "model_family": self.model_family,
            "model_backend": self.model_backend,
            "backend_engine": self.backend_engine,
            "backend_version": self.backend_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_path_or_hf_id": self.model_path_or_hf_id,
            "model_revision": self.model_revision,
            "model_path_hash": self.model_path_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "policy_version": self.policy_version,
            "prompt_template_hash": self.prompt_template_hash,
            "action_parser_version": self.action_parser_version,
            "budget": self.budget,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ModelActionRecord:
    metadata: DriverMetadata
    obs_payload: str
    prompt_text: str
    action_text: str
    parsed_action: str
    parse_status: str
    invalid_action: bool
    model_latency_ms: float
    queue_wait_ms: float
    prompt_tokens: int
    completion_tokens: int

    def to_event_rows(self) -> list[dict[str, Any]]:
        base = self.metadata.to_row()
        prompt_hash = _hash_text(self.prompt_text)
        obs_hash = _hash_text(self.obs_payload)
        action_text_hash = _hash_text(self.action_text)
        raw_output_hash = action_text_hash
        parsed_action_hash = _hash_text(self.parsed_action)
        common = {
            **base,
            "prompt_hash": prompt_hash,
            "obs_hash": obs_hash,
            "action_text_hash": action_text_hash,
            "raw_output_hash": raw_output_hash,
            "parsed_action_hash": parsed_action_hash,
            "parse_status": self.parse_status,
            "invalid_action": self.invalid_action,
            "model_latency_ms": self.model_latency_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }
        return [
            {"event": "model_request_submit", **common},
            {"event": "model_request_start", **common},
            {"event": "model_request_end", **common},
        ]


class BaseAgentDriver(ABC):
    def __init__(self, metadata: DriverMetadata) -> None:
        self.metadata = metadata

    def metadata_row(self) -> dict[str, Any]:
        return self.metadata.to_row()

    @abstractmethod
    def observe(self, obs: Any, task_context: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def act(self, obs: Any, task_context: dict[str, Any], budget: int) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def on_feedback(self, step_result: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def finalize(self, run_result: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
