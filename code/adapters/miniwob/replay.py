"""R0 statistical replay helpers for MiniWoB++ traces."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import math


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, math.ceil((pct / 100.0) * len(xs)) - 1))
    return xs[idx]


def _duration_ms(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    start = datetime.fromisoformat(rows[0]["timestamp_wall"])
    end = datetime.fromisoformat(rows[-1]["timestamp_wall"])
    return (end - start).total_seconds() * 1000.0


def statistical_replay(trace_paths: list[Path]) -> dict[str, Any]:
    durations_ms: list[float] = []
    event_counts: list[int] = []

    for trace_path in trace_paths:
        rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        durations_ms.append(_duration_ms(rows))
        event_counts.append(len(rows))

    return {
        "replay_class": "R0",
        "trace_count": len(trace_paths),
        "event_count_total": sum(event_counts),
        "duration_ms_p50": _percentile(durations_ms, 50),
        "duration_ms_p95": _percentile(durations_ms, 95),
        "duration_ms_p99": _percentile(durations_ms, 99),
        "events_per_trace_p50": _percentile([float(v) for v in event_counts], 50),
    }
