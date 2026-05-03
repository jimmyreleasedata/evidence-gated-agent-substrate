#!/usr/bin/env python3
"""MiniWoB++ smoke and sweep runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import csv
import json
import sys
import time

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from adapters.miniwob.config import MiniWobConfig
from adapters.miniwob.env import make_env
from adapters.miniwob.replay import statistical_replay
from adapters.miniwob.tasks import V1_TASKS, default_task_ids, get_task
from runner.event_bus import EventBus
from runner.run_context import RunContext
from trace.schema.events import EventType
from trace.validators.schema_validator import ValidationResult, validate_jsonl


def _policy_action(task_id: str) -> str:
    return get_task(task_id).default_action


def _summary_path(run_root: Path) -> Path:
    return run_root / "summary.json"


def _write_summary(path: Path, summary: dict[str, Any]) -> Path:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run_single_task(task_id: str, config: MiniWobConfig) -> dict[str, Any]:
    task = get_task(task_id)
    context = RunContext.create(
        output_root=config.output_root,
        task_family="miniwob",
        telemetry_mode=config.telemetry_mode,
        environment_id="miniwob++-mock",
        model_id="mock-policy",
        model_version="0.1.0",
        policy_version="heuristic-v1",
        replay_class="R0",
    )
    bus = EventBus(context)
    env = make_env(task=task, backend=config.backend, telemetry_mode=config.telemetry_mode)
    episode_id = f"{task.task_id}-episode-0001"
    t0 = time.perf_counter_ns()

    bus.emit(
        EventType.RUN_START,
        episode_id=episode_id,
        step_id=0,
        task_id=task.task_id,
        payload={"backend": config.backend, "task": asdict(task)},
    )

    reset_obs = env.reset()
    bus.emit(
        EventType.ENV_RESET,
        episode_id=episode_id,
        step_id=1,
        task_id=task.task_id,
        payload={"observation": reset_obs},
    )

    action = _policy_action(task.task_id)
    bus.emit(
        EventType.POLICY_DECISION,
        episode_id=episode_id,
        step_id=2,
        task_id=task.task_id,
        payload={"action": action},
    )

    bus.emit(
        EventType.ENV_STEP_START,
        episode_id=episode_id,
        step_id=3,
        task_id=task.task_id,
        payload={"action": action},
    )
    observation, reward, done, info = env.step(action)
    bus.emit(
        EventType.ENV_STEP_END,
        episode_id=episode_id,
        step_id=4,
        task_id=task.task_id,
        cpu_pct=3.0 if config.telemetry_mode != "off" else None,
        rss_mb=96.0 if config.telemetry_mode == "full" else None,
        queue_wait_ms=0.0,
        payload={"observation": observation, "reward": reward, "done": done, "info": info},
    )

    outcome = "success" if done and reward > 0 else "failure"
    if outcome != "success":
        bus.emit(
            EventType.FAILURE,
            episode_id=episode_id,
            step_id=5,
            task_id=task.task_id,
            error_class="policy_failure",
            payload={"action": action},
        )

    t1 = time.perf_counter_ns()
    bus.emit(
        EventType.RUN_END,
        episode_id=episode_id,
        step_id=6,
        task_id=task.task_id,
        payload={
            "outcome": outcome,
            "duration_ms": (t1 - t0) / 1_000_000.0,
            "telemetry_mode": config.telemetry_mode,
        },
    )
    env.close()

    telemetry_enabled = context.telemetry_enabled()
    validation = validate_jsonl(context.paths.events_jsonl) if telemetry_enabled else ValidationResult(valid=True, event_count=0)
    trace_path = str(context.paths.events_jsonl) if telemetry_enabled else None
    summary = {
        "run_id": context.run_id,
        "task_id": task.task_id,
        "category": task.category,
        "backend": config.backend,
        "telemetry_mode": config.telemetry_mode,
        "outcome": outcome,
        "duration_ms": (t1 - t0) / 1_000_000.0,
        "event_count": len(bus.events),
        "trace_path": trace_path,
        "manifest_path": str(
            context.write_manifest(
                {
                    "event_count": len(bus.events),
                    "task_id": task.task_id,
                    "trace_enabled": telemetry_enabled,
                }
            )
        ),
        "validation_pass": validation.valid,
    }
    summary_path = _write_summary(_summary_path(context.paths.root), summary)
    summary["summary_path"] = str(summary_path)
    return summary


def run_live(task_ids: list[str], config: MiniWobConfig) -> list[dict[str, Any]]:
    expanded_task_ids: list[str] = []
    for _ in range(config.repeat):
        expanded_task_ids.extend(task_ids)

    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
        summaries = list(pool.map(lambda task_id: _run_single_task(task_id, config), expanded_task_ids))
    return summaries


def write_aggregate_csv(path: Path, summaries: list[dict[str, Any]]) -> Path:
    fieldnames = [
        "run_id",
        "task_id",
        "category",
        "backend",
        "telemetry_mode",
        "outcome",
        "duration_ms",
        "event_count",
        "validation_pass",
        "trace_path",
        "manifest_path",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "replay"), default="live")
    parser.add_argument("--task-id", help="Run a single MiniWoB task.")
    parser.add_argument("--all-tasks", action="store_true", help="Run the full MiniWoB v1 slice.")
    parser.add_argument("--telemetry-mode", choices=("off", "basic", "full"), default="basic")
    parser.add_argument("--backend", choices=("mock",), default="mock")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/miniwob"))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--aggregate-path", type=Path)
    parser.add_argument("--trace-glob", help="Glob for replay mode.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = MiniWobConfig(
        output_root=args.output_root,
        telemetry_mode=args.telemetry_mode,
        backend=args.backend,
        concurrency=args.concurrency,
        repeat=args.repeat,
    )

    if args.mode == "replay":
        if not args.trace_glob:
            raise SystemExit("--trace-glob is required for replay mode")
        trace_paths = sorted(Path().glob(args.trace_glob))
        report = statistical_replay(trace_paths)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.all_tasks:
        task_ids = default_task_ids()
    elif args.task_id:
        task_ids = [args.task_id]
    else:
        task_ids = [V1_TASKS[0].task_id]

    summaries = run_live(task_ids, config)
    aggregate_path = args.aggregate_path
    if aggregate_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        aggregate_path = config.output_root / f"aggregate_{timestamp}.csv"
    write_aggregate_csv(aggregate_path, summaries)
    print(
        json.dumps(
            {
                "task_count": len(summaries),
                "aggregate_path": str(aggregate_path),
                "telemetry_mode": config.telemetry_mode,
                "concurrency": config.concurrency,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
