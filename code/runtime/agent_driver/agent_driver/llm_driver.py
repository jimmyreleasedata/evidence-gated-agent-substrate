"""Generic contract-conforming LLM driver."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable

from .base import BaseAgentDriver, DriverMetadata, ModelActionRecord


GenerateFn = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class LLMDriverConfig:
    driver_id: str
    driver_version: str
    model_backend: str
    backend_engine: str
    backend_version: str
    model_id: str
    model_family: str
    model_revision: str
    model_path_or_hf_id: str
    tokenizer_path: str
    policy_version: str
    prompt_template: str
    action_parser_version: str
    budget: int
    seed: int


class LLMDriver(BaseAgentDriver):
    def __init__(self, config: LLMDriverConfig, *, generate_fn: GenerateFn) -> None:
        self.config = config
        self._generate_fn = generate_fn
        super().__init__(
            DriverMetadata(
                driver_id=config.driver_id,
                driver_type="llm_agent",
                driver_version=config.driver_version,
                model_family=config.model_family,
                model_backend=config.model_backend,
                backend_engine=config.backend_engine,
                backend_version=config.backend_version,
                model_id=config.model_id,
                model_version=config.model_revision,
                model_path=config.model_path_or_hf_id,
                model_path_or_hf_id=config.model_path_or_hf_id,
                model_revision=config.model_revision,
                tokenizer_path=config.tokenizer_path,
                policy_version=config.policy_version,
                prompt_template=config.prompt_template,
                action_parser_version=config.action_parser_version,
                budget=config.budget,
                seed=config.seed,
            )
        )

    def observe(self, obs: Any, task_context: dict[str, Any]) -> dict[str, Any]:
        return {"obs": obs, "task_context": task_context}

    def _render_prompt(self, obs: Any, task_context: dict[str, Any]) -> str:
        return (
            self.config.prompt_template.replace("{obs}", str(obs)).replace(
                "{task_context}",
                str(task_context),
            )
        )

    def _parse_action(self, action_text: str) -> tuple[dict[str, Any], str, bool]:
        parsed_action = {"raw": action_text}
        parse_status = "ok" if action_text else "empty"
        invalid_action = not bool(action_text)
        return parsed_action, parse_status, invalid_action

    def act(self, obs: Any, task_context: dict[str, Any], budget: int) -> dict[str, Any]:
        prompt_text = self._render_prompt(obs, task_context)
        started = monotonic()
        action_text = self._generate_fn(prompt_text)
        latency_ms = round((monotonic() - started) * 1000.0, 3)
        parsed_action, parse_status, invalid_action = self._parse_action(action_text)
        record = ModelActionRecord(
            metadata=self.metadata,
            obs_payload=str(obs),
            prompt_text=prompt_text,
            action_text=action_text,
            parsed_action=str(parsed_action),
            parse_status=parse_status,
            invalid_action=invalid_action,
            model_latency_ms=latency_ms,
            queue_wait_ms=0.0,
            prompt_tokens=max(len(prompt_text.split()), 1),
            completion_tokens=max(len(action_text.split()), 1) if action_text else 0,
        )
        return {
            "action_text": action_text,
            "parsed_action": parsed_action,
            "parse_status": parse_status,
            "invalid_action": invalid_action,
            "events": record.to_event_rows(),
            "budget": budget,
        }

    def on_feedback(self, step_result: dict[str, Any]) -> dict[str, Any]:
        return {"feedback": step_result}

    def finalize(self, run_result: dict[str, Any]) -> dict[str, Any]:
        return {"metadata": self.metadata_row(), "run_result": run_result}
