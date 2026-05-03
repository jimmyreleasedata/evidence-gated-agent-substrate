"""Real-upstream WebArena Verified replay runner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from adapters.webarena_verified.real_config import WebArenaVerifiedRealConfig
from adapters.webarena_verified.real_evaluator import OfficialCompatibleWebArenaEvaluator
from adapters.webarena_verified.real_trace_loader import RealWebArenaTraceBundle, load_real_trace_bundles
from runner.event_bus import EventBus
from runner.run_context import RunContext
from trace.schema.events import EventType
from trace.validators.schema_validator import validate_jsonl


def _read_trace_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_step_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in rows:
        event_type = row.get("event_type")
        if event_type == "env_step_start":
            current = {"action": row.get("payload", {}).get("action"), "step_id": row.get("step_id")}
        elif event_type == "env_step_end" and current is not None:
            current["observation"] = row.get("payload", {}).get("observation")
            current["info"] = row.get("payload", {}).get("info", {})
            pairs.append(current)
            current = None
    return pairs


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_aggregate(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames = [
        "task_id",
        "passed",
        "score",
        "trace_hash",
        "task_manifest_hash",
        "evaluator_version",
        "implementation_source",
        "backend",
        "summary_path",
        "manifest_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def run_real_replay_task(
    bundle: RealWebArenaTraceBundle,
    config: WebArenaVerifiedRealConfig,
    evaluator: OfficialCompatibleWebArenaEvaluator,
) -> dict[str, Any]:
    source_rows = _read_trace_rows(bundle.trace_path)
    context = RunContext.create(
        output_root=config.output_root,
        task_family="webarena_verified",
        telemetry_mode=config.telemetry_mode,
        environment_id="webarena_verified_official_r1",
        model_id=str(bundle.task_manifest.get("model_id") or "captured-trace"),
        model_version=str(bundle.task_manifest.get("model_version") or "captured"),
        policy_version=str(bundle.task_manifest.get("policy_version") or "captured"),
        verifier_id=evaluator.info.evaluator_id,
        verifier_version=config.evaluator_version,
        replay_class="R1",
    )
    bus = EventBus(context)
    episode_id = f"{bundle.task_id}-episode-0001"

    bus.emit(
        EventType.RUN_START,
        episode_id=episode_id,
        step_id=0,
        task_id=bundle.task_id,
        payload={
            "mode": "official_replay",
            "implementation_source": "real_upstream_replay",
            "trace_bundle_ref": bundle.trace_bundle_ref,
            "trace_hash": bundle.trace_hash,
            "task_manifest_hash": bundle.task_manifest_hash,
            "upstream_commit": config.upstream_commit,
        },
    )
    bus.emit(
        EventType.REPLAY_CONSUME,
        episode_id=episode_id,
        step_id=1,
        task_id=bundle.task_id,
        replay_source=str(bundle.trace_path),
        payload={"source_event_count": len(source_rows), "trace_hash": bundle.trace_hash},
    )

    step_id = 2
    for pair in _extract_step_pairs(source_rows):
        bus.emit(
            EventType.ENV_STEP_START,
            episode_id=episode_id,
            step_id=step_id,
            task_id=bundle.task_id,
            payload={"action": pair.get("action")},
        )
        step_id += 1
        bus.emit(
            EventType.ENV_STEP_END,
            episode_id=episode_id,
            step_id=step_id,
            task_id=bundle.task_id,
            payload={"observation": pair.get("observation"), "info": pair.get("info", {})},
        )
        step_id += 1

    bus.emit(
        EventType.VERIFIER_START,
        episode_id=episode_id,
        step_id=step_id,
        task_id=bundle.task_id,
        payload={"evaluator_version": config.evaluator_version},
    )
    step_id += 1
    evaluation = evaluator.evaluate_replay(bundle)
    bus.emit(
        EventType.VERIFIER_END,
        episode_id=episode_id,
        step_id=step_id,
        task_id=bundle.task_id,
        payload=evaluation,
    )
    step_id += 1
    bus.emit(
        EventType.RUN_END,
        episode_id=episode_id,
        step_id=step_id,
        task_id=bundle.task_id,
        payload={"passed": evaluation["passed"], "score": evaluation["score"]},
    )

    manifest_path = context.write_manifest(
        {
            "backend": "webarena_verified_official_r1",
            "implementation_source": "real_upstream_replay",
            "upstream_root": str(config.upstream_root),
            "upstream_commit": config.upstream_commit,
            "task_id": bundle.task_id,
            "upstream_task_id": bundle.upstream_task_id,
            "trace_path": str(bundle.trace_path),
            "trace_hash": bundle.trace_hash,
            "task_manifest_path": str(bundle.task_manifest_path),
            "task_manifest_hash": bundle.task_manifest_hash,
            "evaluator_result_path": str(bundle.evaluator_result_path),
            "evaluator_version": config.evaluator_version,
            "trace_bundle_ref": bundle.trace_bundle_ref,
            "freeze_info": evaluation["freeze_info"],
            "selected_task_ids": [bundle.task_id],
        }
    )
    validation = validate_jsonl(context.paths.events_jsonl) if context.telemetry_enabled() else None
    summary = {
        "mode": "official_replay",
        "task_id": bundle.task_id,
        "upstream_task_id": bundle.upstream_task_id,
        "run_id": context.run_id,
        "implementation_source": "real_upstream_replay",
        "backend": "webarena_verified_official_r1",
        "replay_class": "R1",
        "evaluator_version": config.evaluator_version,
        "upstream_commit": config.upstream_commit,
        "trace_hash": bundle.trace_hash,
        "task_manifest_hash": bundle.task_manifest_hash,
        "validation_pass": True if validation is None else validation.valid,
        "passed": evaluation["passed"],
        "score": evaluation["score"],
        "trace_path": str(context.paths.events_jsonl) if context.telemetry_enabled() else None,
        "summary_path": str(context.paths.root / "summary.json"),
        "manifest_path": str(manifest_path),
        "environment_id": "webarena_verified_official_r1",
        "freeze_info": evaluation["freeze_info"],
    }
    summary_path = _write_json(context.paths.root / "summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary


def run_real_replay(config: WebArenaVerifiedRealConfig) -> list[dict[str, Any]]:
    evaluator = OfficialCompatibleWebArenaEvaluator(config.evaluator_version)
    return [run_real_replay_task(bundle, config, evaluator) for bundle in load_real_trace_bundles(config)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/webarena_verified_real"))
    parser.add_argument("--telemetry-mode", choices=("off", "basic", "full"), default="basic")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = WebArenaVerifiedRealConfig.from_env(output_root=args.output_root, telemetry_mode=args.telemetry_mode)
    rows = run_real_replay(config)
    aggregate_path = _write_aggregate(config.output_root / "aggregate_real_replay.csv", rows)
    print(json.dumps({"task_count": len(rows), "aggregate_path": str(aggregate_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
