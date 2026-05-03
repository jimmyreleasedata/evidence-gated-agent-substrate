"""Environment backends for the MiniWoB++ adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from adapters.miniwob.tasks import MiniWobTaskSpec


@dataclass(slots=True)
class MockMiniWobEnv:
    task: MiniWobTaskSpec
    telemetry_mode: str

    def reset(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "instruction": self.task.instruction,
            "telemetry_mode": self.telemetry_mode,
            "backend": "mock",
        }

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        t0 = time.perf_counter_ns()
        telemetry_penalty_ms = {"off": 0, "basic": 1, "full": 2}.get(self.telemetry_mode, 1)
        time.sleep((self.task.step_delay_ms + telemetry_penalty_ms) / 1000.0)
        success = action == self.task.default_action
        reward = 1.0 if success else 0.0
        t1 = time.perf_counter_ns()
        latency_ms = (t1 - t0) / 1_000_000.0
        observation = {
            "last_action": action,
            "goal_reached": success,
            "task_id": self.task.task_id,
        }
        info = {
            "backend": "mock",
            "step_latency_ms": latency_ms,
            "telemetry_penalty_ms": telemetry_penalty_ms,
        }
        return observation, reward, True, info

    def close(self) -> None:
        return None


def make_env(task: MiniWobTaskSpec, backend: str, telemetry_mode: str) -> MockMiniWobEnv:
    if backend != "mock":
        raise RuntimeError(
            "real MiniWoB backend is not implemented in this repository yet; use backend=mock"
        )
    return MockMiniWobEnv(task=task, telemetry_mode=telemetry_mode)
