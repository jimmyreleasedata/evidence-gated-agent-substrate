#!/usr/bin/env python3
"""Offline-first runner for WebArena Verified."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, ""):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from adapters.webarena_verified.config import WebArenaVerifiedConfig
from adapters.webarena_verified.env import make_env
from adapters.webarena_verified.evaluator import MockWebArenaEvaluator
from adapters.webarena_verified.replay import load_json, replay_trace, variance_report
from adapters.webarena_verified.tasks import default_task_ids, get_task
from runner.event_bus import EventBus
from runner.run_context import RunContext
from trace.schema.events import EventType
from trace.validators.schema_validator import ValidationResult, validate_jsonl


def resolve_replay_source_trace(source_trace: Path | None, repo_root: Path = REPO_ROOT) -> Path:
    if source_trace is not None:
        return source_trace

    candidate_roots = (
        repo_root / "artifacts" / "webarena_verified" / "live",
        repo_root / "artifacts" / "debug_webarena_smoke" / "live",
    )
    for root in candidate_roots:
        candidates = sorted(
            root.glob("*/events.jsonl"),
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
            reverse=True,
        )
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        "No default WebArena replay source found; pass --source-trace or place a trace bundle under artifacts/webarena_verified/live"
    )


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_live_sanity(config: WebArenaVerifiedConfig) -> dict:
    task = get_task(config.task_id)
    evaluator = MockWebArenaEvaluator()
    context = RunContext.create(
        output_root=config.output_root,
        task_family="webarena_verified",
        telemetry_mode=config.telemetry_mode,
        environment_id="webarena-verified-mock",
        model_id="mock-web-policy",
        model_version="0.1.0",
        policy_version="heuristic-v1",
        verifier_id=evaluator.freeze_info.evaluator_id,
        verifier_version=evaluator.freeze_info.evaluator_version,
        replay_class="R1",
    )
    bus = EventBus(context)
    env = make_env(
        task,
        config.backend,
        config.telemetry_mode,
        regime=config.regime,
        offered_load=config.offered_load,
    )
    episode_id = f"{task.task_id}-episode-0001"
    t0 = time.perf_counter_ns()

    bus.emit(
        EventType.RUN_START,
        episode_id=episode_id,
        step_id=0,
        task_id=task.task_id,
        payload={
            "mode": "live_sanity",
            "instruction": task.instruction,
            "start_url": task.start_url,
            "regime": config.regime,
            "offered_load": int(config.offered_load),
        },
    )

    reset_obs = env.reset()
    bus.emit(
        EventType.ENV_RESET,
        episode_id=episode_id,
        step_id=1,
        task_id=task.task_id,
        payload={"observation": reset_obs},
    )

    action = f"open:{task.expected_slug}"
    bus.emit(
        EventType.POLICY_DECISION,
        episode_id=episode_id,
        step_id=2,
        task_id=task.task_id,
        payload={"action": action},
    )

    bus.emit(
        EventType.TOOL_CALL_SUBMIT,
        episode_id=episode_id,
        step_id=3,
        task_id=task.task_id,
        tool_id="browser.dom_snapshot",
        payload={"tool": "dom_snapshot"},
    )
    bus.emit(
        EventType.TOOL_CALL_START,
        episode_id=episode_id,
        step_id=4,
        task_id=task.task_id,
        tool_id="browser.screenshot",
        payload={"tool": "screenshot"},
    )
    bus.emit(
        EventType.ENV_STEP_START,
        episode_id=episode_id,
        step_id=5,
        task_id=task.task_id,
        payload={"action": action},
    )
    observation, success, info = env.step(action)
    bus.emit(
        EventType.TOOL_CALL_END,
        episode_id=episode_id,
        step_id=6,
        task_id=task.task_id,
        tool_id="browser.capture_done",
        payload=info,
    )
    bus.emit(
        EventType.ENV_STEP_END,
        episode_id=episode_id,
        step_id=7,
        task_id=task.task_id,
        queue_wait_ms=0.0,
        cpu_pct=5.0 if config.telemetry_mode != "off" else None,
        rss_mb=128.0 if config.telemetry_mode == "full" else None,
        payload={"observation": observation, "info": info, "success": success},
    )

    bus.emit(
        EventType.VERIFIER_START,
        episode_id=episode_id,
        step_id=8,
        task_id=task.task_id,
        payload={"mode": "live_sanity"},
    )
    evaluation = evaluator.evaluate_live(task.task_id, observation, success)
    bus.emit(
        EventType.VERIFIER_END,
        episode_id=episode_id,
        step_id=9,
        task_id=task.task_id,
        payload=evaluation,
    )

    t1 = time.perf_counter_ns()
    bus.emit(
        EventType.RUN_END,
        episode_id=episode_id,
        step_id=10,
        task_id=task.task_id,
        payload={
            "mode": "live_sanity",
            "passed": evaluation["passed"],
            "duration_ms": (t1 - t0) / 1_000_000.0,
        },
    )
    env.close()

    telemetry_enabled = context.telemetry_enabled()
    validation = validate_jsonl(context.paths.events_jsonl) if telemetry_enabled else ValidationResult(valid=True, event_count=0)
    trace_path = str(context.paths.events_jsonl) if telemetry_enabled else None
    trace_bytes = context.paths.events_jsonl.stat().st_size if telemetry_enabled and context.paths.events_jsonl.exists() else 0
    summary = {
        "mode": "live_sanity",
        "run_id": context.run_id,
        "task_id": task.task_id,
        "system_version": config.system_version,
        "regime": config.regime,
        "offered_load": int(config.offered_load),
        "backend": config.backend,
        "telemetry_mode": config.telemetry_mode,
        "passed": evaluation["passed"],
        "score": evaluation["score"],
        "duration_ms": (t1 - t0) / 1_000_000.0,
        "event_count": len(bus.events),
        "trace_bytes": trace_bytes,
        "render_wait_ms": info["render_wait_ms"],
        "network_trace_align_ms": info["network_trace_align_ms"],
        "dom_get_ms": info["dom_get_ms"],
        "screenshot_ms": info["screenshot_ms"],
        "evaluator_latency_ms": 1.0,
        "trace_path": trace_path,
        "validation_pass": validation.valid,
        "evaluator_freeze": evaluation["freeze_info"],
        "manifest_path": str(
            context.write_manifest(
                {
                    "task_id": task.task_id,
                    "task_family": "webarena_verified",
                    "backend": config.backend,
                    "telemetry_mode": config.telemetry_mode,
                    "event_count": len(bus.events),
                    "trace_enabled": telemetry_enabled,
                    "evaluator_freeze": evaluation["freeze_info"],
                    "system_version": config.system_version,
                    "regime": config.regime,
                    "offered_load": int(config.offered_load),
                }
            )
        ),
    }
    summary_path = _write_json(context.paths.root / "summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "replay"), default="live")
    parser.add_argument("--task-id", default="search_product_specs")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--telemetry-mode", choices=("off", "basic", "full"), default="basic")
    parser.add_argument("--backend", choices=("mock",), default="mock")
    parser.add_argument(
        "--system-version",
        choices=("sync", "naive_async", "scheduler_only", "replay_only", "control_plane_only", "full_system"),
        default="full_system",
    )
    parser.add_argument(
        "--regime",
        choices=("baseline", "webarena_verified_live_stressed"),
        default="baseline",
    )
    parser.add_argument("--offered-load", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/webarena_verified"))
    parser.add_argument("--source-trace", type=Path)
    parser.add_argument("--source-summary", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = WebArenaVerifiedConfig(
        output_root=args.output_root,
        telemetry_mode=args.telemetry_mode,
        backend=args.backend,
        system_version=args.system_version,
        regime=args.regime,
        offered_load=args.offered_load,
        task_id=args.task_id,
        source_trace=args.source_trace,
        source_summary=args.source_summary,
    )

    if args.mode == "live":
        task_ids = default_task_ids() if args.all_tasks else [config.task_id]
        summaries = []
        for task_id in task_ids:
            task_root = config.output_root / task_id if args.all_tasks else config.output_root
            task_config = WebArenaVerifiedConfig(
                output_root=task_root,
                telemetry_mode=config.telemetry_mode,
                backend=config.backend,
                system_version=config.system_version,
                regime=config.regime,
                offered_load=config.offered_load,
                task_id=task_id,
                source_trace=config.source_trace,
                source_summary=config.source_summary,
            )
            summaries.append(run_live_sanity(task_config))
        aggregate_path = config.output_root / f"aggregate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        write_csv(
            aggregate_path,
            summaries,
            [
                "mode",
                "run_id",
                "task_id",
                "backend",
                "system_version",
                "regime",
                "offered_load",
                "telemetry_mode",
                "passed",
                "score",
                "duration_ms",
                "event_count",
                "render_wait_ms",
                "network_trace_align_ms",
                "dom_get_ms",
                "screenshot_ms",
                "evaluator_latency_ms",
                "validation_pass",
                "trace_path",
                "manifest_path",
                "summary_path",
            ],
        )
        print(
            json.dumps(
                {
                    "aggregate_path": str(aggregate_path),
                    "task_count": len(summaries),
                    "trace_roots": [summary["trace_path"] for summary in summaries],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    try:
        source_trace = resolve_replay_source_trace(config.source_trace, repo_root=REPO_ROOT)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    summary = replay_trace(
        source_trace,
        config.output_root,
        config.telemetry_mode,
        system_version=config.system_version,
    )
    if config.source_summary and config.source_summary.exists():
        live_summary = load_json(config.source_summary)
        if live_summary is not None:
            variance = variance_report(live_summary, summary)
            variance_path = config.output_root / f"variance_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            _write_json(variance_path, variance)
            summary["variance_path"] = str(variance_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
