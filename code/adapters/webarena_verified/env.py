"""Mock environment for WebArena Verified live sanity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from adapters.webarena_verified.tasks import WebArenaTaskSpec


@dataclass(slots=True)
class MockWebArenaEnv:
    task: WebArenaTaskSpec
    telemetry_mode: str
    regime: str = "baseline"
    offered_load: int = 1

    def reset(self) -> dict[str, Any]:
        return {
            "url": self.task.start_url,
            "site": self.task.site,
            "instruction": self.task.instruction,
        }

    def step(self, action: str) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        telemetry_penalty_ms = {"off": 0, "basic": 3, "full": 7}.get(self.telemetry_mode, 3)
        load = max(int(self.offered_load), 1)
        render_wait_ms = 18 + telemetry_penalty_ms
        network_trace_align_ms = 7 + telemetry_penalty_ms
        dom_get_ms = 5 + telemetry_penalty_ms
        screenshot_ms = 9 + telemetry_penalty_ms

        if self.regime == "webarena_verified_live_stressed":
            render_wait_ms += 11 * max(load - 1, 0)
            network_trace_align_ms += 8 * max(load - 1, 0)
            screenshot_ms += 5 * max(load - 1, 0)
            dom_get_ms += 2 * max(load - 1, 0)

        time.sleep((render_wait_ms + network_trace_align_ms + dom_get_ms + screenshot_ms) / 1000.0)

        success = self.task.expected_slug in action
        observation = {
            "current_url": f"https://{self.task.site}.example/{self.task.expected_slug}",
            "title": f"{self.task.expected_slug} - Mock Page",
            "site": self.task.site,
        }
        info = {
            "render_wait_ms": float(render_wait_ms),
            "network_trace_align_ms": float(network_trace_align_ms),
            "dom_get_ms": float(dom_get_ms),
            "screenshot_ms": float(screenshot_ms),
            "regime": self.regime,
            "offered_load": load,
            "session_crash": False,
            "timeout": False,
            "reset_needed": False,
        }
        return observation, success, info

    def close(self) -> None:
        return None


def make_env(
    task: WebArenaTaskSpec,
    backend: str,
    telemetry_mode: str,
    *,
    regime: str = "baseline",
    offered_load: int = 1,
) -> MockWebArenaEnv:
    if backend != "mock":
        raise RuntimeError(
            "real WebArena Verified backend is not wired yet; use backend=mock for offline-first bring-up"
        )
    return MockWebArenaEnv(task=task, telemetry_mode=telemetry_mode, regime=regime, offered_load=offered_load)
