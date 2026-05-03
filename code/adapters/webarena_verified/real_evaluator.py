"""Official-compatible deterministic evaluator wrapper for real WebArena replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from adapters.webarena_verified.real_trace_loader import RealWebArenaTraceBundle


@dataclass(frozen=True, slots=True)
class RealWebArenaEvaluatorInfo:
    evaluator_id: str = "webarena_verified_real_evaluator"
    trace_contract: str = "R1"
    expected_trace_schema_version: str = "1.0.0"
    notes: str = "Deterministic offline evaluator consuming real WebArena Verified trace/evaluator artifacts."


class OfficialCompatibleWebArenaEvaluator:
    def __init__(self, required_version: str) -> None:
        if not required_version.strip():
            raise ValueError("evaluator version is required")
        self.required_version = required_version
        self.info = RealWebArenaEvaluatorInfo()

    def evaluate_replay(self, bundle: RealWebArenaTraceBundle) -> dict[str, Any]:
        payload = bundle.evaluator_result
        observed_version = str(payload.get("evaluator_version") or payload.get("version") or "").strip()
        if not observed_version:
            raise ValueError(f"evaluator version missing for {bundle.task_id}")
        if observed_version != self.required_version:
            raise ValueError(
                f"evaluator version mismatch for {bundle.task_id}: expected {self.required_version}, got {observed_version}"
            )

        passed = bool(payload.get("passed"))
        score = float(payload.get("score", 1.0 if passed else 0.0))
        evaluator_id = str(payload.get("evaluator_id") or self.info.evaluator_id)
        return {
            "task_id": bundle.task_id,
            "passed": passed,
            "score": score,
            "trace_hash": bundle.trace_hash,
            "task_manifest_hash": bundle.task_manifest_hash,
            "evaluator_id": evaluator_id,
            "evaluator_version": observed_version,
            "freeze_info": {
                **asdict(self.info),
                "evaluator_id": evaluator_id,
                "evaluator_version": observed_version,
            },
        }
