"""R2 replay checks for the SWE-Gym mock harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from runner.event_bus import EventBus
from runner.run_context import RunContext
from trace.schema.events import EventType
from trace.validators.schema_validator import validate_jsonl


REQUIRED_FREEZE_FIELDS = [
    "task_instance_id",
    "repo_commit",
    "image",
    "dependency_lock",
    "build_command",
    "test_command",
    "seed_policy",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def replay_from_manifest(manifest_path: Path, output_root: Path, telemetry_mode: str = "basic") -> dict[str, Any]:
    t0 = time.perf_counter_ns()
    manifest = load_json(manifest_path)
    missing = [field for field in REQUIRED_FREEZE_FIELDS if field not in manifest or manifest[field] in (None, "", {})]
    if missing:
        raise ValueError(f"missing required R2 freeze fields: {missing}")

    context = RunContext.create(
        output_root=output_root,
        task_family="swe_gym",
        telemetry_mode=telemetry_mode,
        environment_id="swebench-harness-mock",
        snapshot_id=manifest["image"].get("sif_hash"),
        verifier_id="swebench_harness",
        verifier_version="0.1.0",
        replay_class="R2",
    )
    bus = EventBus(context)

    task_id = manifest["task_id"]
    episode_id = f"{manifest['task_instance_id']}-replay"
    bus.emit(
        EventType.RUN_START,
        episode_id=episode_id,
        step_id=0,
        task_id=task_id,
        payload={"mode": "snapshot_replay", "source_manifest": str(manifest_path)},
    )
    for idx, field_name in enumerate(REQUIRED_FREEZE_FIELDS, start=1):
        bus.emit(
            EventType.REPLAY_CONSUME,
            episode_id=episode_id,
            step_id=idx,
            task_id=task_id,
            payload={"freeze_field": field_name, "value": manifest[field_name]},
        )
    bus.emit(
        EventType.VERIFIER_START,
        episode_id=episode_id,
        step_id=len(REQUIRED_FREEZE_FIELDS) + 1,
        task_id=task_id,
        payload={"mode": "snapshot_replay"},
    )
    bus.emit(
        EventType.VERIFIER_END,
        episode_id=episode_id,
        step_id=len(REQUIRED_FREEZE_FIELDS) + 2,
        task_id=task_id,
        payload={"passed": True, "validated_freeze_fields": REQUIRED_FREEZE_FIELDS},
    )
    t1 = time.perf_counter_ns()
    bus.emit(
        EventType.RUN_END,
        episode_id=episode_id,
        step_id=len(REQUIRED_FREEZE_FIELDS) + 3,
        task_id=task_id,
        payload={"mode": "snapshot_replay", "passed": True, "duration_ms": (t1 - t0) / 1_000_000.0},
    )

    validation = validate_jsonl(context.paths.events_jsonl)
    summary = {
        "mode": "snapshot_replay",
        "run_id": context.run_id,
        "task_id": task_id,
        "validation_pass": validation.valid,
        "trace_path": str(context.paths.events_jsonl),
        "manifest_path": str(context.write_manifest({"source_manifest": str(manifest_path)})),
        "validated_freeze_fields": REQUIRED_FREEZE_FIELDS,
        "passed": True,
        "duration_ms": (t1 - t0) / 1_000_000.0,
    }
    summary_path = context.paths.root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
