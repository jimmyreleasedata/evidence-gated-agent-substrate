"""Mock evaluator and freeze metadata for WebArena Verified."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluatorFreezeInfo:
    evaluator_id: str = "webarena_verified_evaluator"
    evaluator_version: str = "0.1.0"
    trace_contract: str = "R1"
    expected_trace_schema_version: str = "1.0.0"
    notes: str = "Mock offline evaluator for fixture-first bring-up."


@dataclass(slots=True)
class MockWebArenaEvaluator:
    freeze_info: EvaluatorFreezeInfo = EvaluatorFreezeInfo()

    def evaluate_live(self, task_id: str, observation: dict[str, Any], success: bool) -> dict[str, Any]:
        score = 1.0 if success else 0.0
        return {
            "task_id": task_id,
            "score": score,
            "passed": bool(success),
            "current_url": observation.get("current_url"),
            "freeze_info": asdict(self.freeze_info),
        }

    def evaluate_replay(self, task_id: str, source_summary: dict[str, Any] | None, source_event_count: int) -> dict[str, Any]:
        source_passed = bool((source_summary or {}).get("passed", True))
        return {
            "task_id": task_id,
            "score": 1.0 if source_passed else 0.0,
            "passed": source_passed,
            "source_event_count": source_event_count,
            "freeze_info": asdict(self.freeze_info),
        }
