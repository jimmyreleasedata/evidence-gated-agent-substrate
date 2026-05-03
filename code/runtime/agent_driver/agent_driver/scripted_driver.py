"""Scripted reference/controller driver wrapper."""

from __future__ import annotations

from typing import Any, Callable

from .base import BaseAgentDriver, DriverMetadata


class ScriptedDriver(BaseAgentDriver):
    def __init__(
        self,
        *,
        driver_id: str,
        driver_type: str,
        driver_version: str,
        policy_version: str,
        action_parser_version: str,
        budget: int,
        seed: int,
        act_fn: Callable[[Any, dict[str, Any], int], dict[str, Any]],
    ) -> None:
        self._act_fn = act_fn
        super().__init__(
            DriverMetadata(
                driver_id=driver_id,
                driver_type=driver_type,
                driver_version=driver_version,
                model_backend="scripted",
                model_id=driver_id,
                model_version=driver_version,
                model_path=driver_id,
                tokenizer_path=driver_id,
                policy_version=policy_version,
                prompt_template=driver_id,
                action_parser_version=action_parser_version,
                budget=budget,
                seed=seed,
            )
        )

    def observe(self, obs: Any, task_context: dict[str, Any]) -> dict[str, Any]:
        return {"obs": obs, "task_context": task_context}

    def act(self, obs: Any, task_context: dict[str, Any], budget: int) -> dict[str, Any]:
        return self._act_fn(obs, task_context, budget)

    def on_feedback(self, step_result: dict[str, Any]) -> dict[str, Any]:
        return {"feedback": step_result}

    def finalize(self, run_result: dict[str, Any]) -> dict[str, Any]:
        return {"metadata": self.metadata_row(), "run_result": run_result}
