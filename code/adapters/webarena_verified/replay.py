"""Offline replay helpers for WebArena Verified traces."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import time
import uuid

from pydantic import ValidationError

from adapters.webarena_verified.evaluator import MockWebArenaEvaluator
from runner.event_bus import EventBus
from runner.run_context import RunContext
from trace.schema.events import EventType, TraceEvent
from trace.validators.schema_validator import validate_jsonl


SUPPORTED_TRACE_SCHEMA_VERSIONS = {"1.0.0"}


class ReplayError(RuntimeError):
    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason)


class ReplayContractError(ReplayError):
    pass


class ReplayIntegrityError(ReplayError):
    pass


class ReplayUnsupportedVersionError(ReplayError):
    pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_signature(payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _deterministic_step_sequence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step_id": row["step_id"],
            "event_type": row["event_type"],
            "payload_signature": _payload_signature(row.get("payload", {})),
        }
        for row in rows
    ]


def _classify_validation_error(exc: Exception) -> str:
    text = str(exc)
    if "Field required" in text:
        return "missing_required_field"
    return "schema_validation_error"


def _validate_trace_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ReplayIntegrityError("partial_trace", "partial_trace: empty_trace")

    events: list[TraceEvent] = []
    for row in rows:
        try:
            event = TraceEvent.model_validate(row)
        except ValidationError as exc:
            reason = _classify_validation_error(exc)
            raise ReplayIntegrityError(reason, f"{reason}: {exc}") from exc
        try:
            datetime.fromisoformat(event.timestamp_wall)
        except ValueError as exc:
            raise ReplayIntegrityError("malformed_timestamp", f"malformed_timestamp: {event.timestamp_wall}") from exc
        events.append(event)

    if events[0].task_family != "webarena_verified":
        raise ReplayIntegrityError(
            "unexpected_task_family",
            f"unexpected_task_family: {events[0].task_family}",
        )

    event_types = [event.event_type.value for event in events]
    if event_types[0] != "run_start":
        raise ReplayIntegrityError("missing_run_start", f"missing_run_start: got {event_types[0]}")
    if event_types[-1] != "run_end":
        raise ReplayIntegrityError("missing_run_end", f"missing_run_end: got {event_types[-1]}")

    seen_step_ids: set[int] = set()
    seen_span_ids: set[str] = set()
    last_step_id: int | None = None
    env_step_start_count = 0
    env_step_end_count = 0
    for event in events:
        if event.step_id in seen_step_ids:
            raise ReplayIntegrityError("duplicate_step_id", f"duplicate_step_id: {event.step_id}")
        if event.span_id in seen_span_ids:
            raise ReplayIntegrityError("duplicate_span_id", f"duplicate_span_id: {event.span_id}")
        if last_step_id is not None and event.step_id < last_step_id:
            raise ReplayIntegrityError("out_of_order_step_id", f"out_of_order_step_id: {event.step_id} < {last_step_id}")
        if event.event_type.value == "env_step_start":
            env_step_start_count += 1
        if event.event_type.value == "env_step_end":
            env_step_end_count += 1
        seen_step_ids.add(event.step_id)
        seen_span_ids.add(event.span_id)
        last_step_id = event.step_id

    if env_step_end_count > env_step_start_count:
        raise ReplayIntegrityError("missing_env_step_start", "missing_env_step_start: env_step_end without start")
    if env_step_start_count > env_step_end_count:
        raise ReplayIntegrityError("missing_env_step_end", "missing_env_step_end: env_step_start without end")


def _extract_contract(
    rows: list[dict[str, Any]],
    source_summary: dict[str, Any] | None,
    source_manifest: dict[str, Any] | None,
    evaluator: MockWebArenaEvaluator,
) -> dict[str, Any]:
    first = rows[0]
    manifest = source_manifest or {}
    summary = source_summary or {}
    freeze = asdict(evaluator.freeze_info)

    trace_schema_version = str(first["trace_schema_version"])
    if trace_schema_version not in SUPPORTED_TRACE_SCHEMA_VERSIONS:
        raise ReplayUnsupportedVersionError(
            "unsupported_trace_schema_version",
            f"unsupported_trace_schema_version: {trace_schema_version}",
        )
    if freeze["expected_trace_schema_version"] != trace_schema_version:
        raise ReplayUnsupportedVersionError(
            "unsupported_trace_schema_version",
            f"unsupported_trace_schema_version: expected {freeze['expected_trace_schema_version']} got {trace_schema_version}",
        )

    replay_class = str(manifest.get("replay_class") or freeze["trace_contract"])
    if replay_class != "R1":
        raise ReplayContractError("replay_class_not_r1", f"replay_class_not_r1: {replay_class}")

    source_hash = summary.get("task_manifest_hash")
    manifest_hash = manifest.get("task_manifest_hash")
    if source_hash and manifest_hash and source_hash != manifest_hash:
        raise ReplayContractError(
            "task_manifest_hash_mismatch",
            f"task_manifest_hash_mismatch: {source_hash} != {manifest_hash}",
        )

    summary_verifier_version = summary.get("verifier_version")
    manifest_verifier_version = manifest.get("verifier_version")
    expected_verifier_version = freeze["evaluator_version"]
    observed_versions = {
        value
        for value in (summary_verifier_version, manifest_verifier_version, expected_verifier_version)
        if value is not None
    }
    if len(observed_versions) > 1:
        raise ReplayContractError(
            "verifier_version_mismatch",
            f"verifier_version_mismatch: {sorted(observed_versions)}",
        )

    summary_environment_id = summary.get("environment_id")
    manifest_environment_id = manifest.get("environment_id")
    if manifest_environment_id and first.get("environment_id") != manifest_environment_id:
        raise ReplayContractError(
            "environment_id_mismatch",
            f"environment_id_mismatch: {first.get('environment_id')} != {manifest_environment_id}",
        )
    if summary_environment_id and manifest_environment_id and summary_environment_id != manifest_environment_id:
        raise ReplayContractError(
            "environment_id_mismatch",
            f"environment_id_mismatch: {summary_environment_id} != {manifest_environment_id}",
        )

    summary_snapshot_id = summary.get("snapshot_id")
    manifest_snapshot_id = manifest.get("snapshot_id")
    if manifest_snapshot_id and first.get("snapshot_id") != manifest_snapshot_id:
        raise ReplayContractError(
            "snapshot_id_mismatch",
            f"snapshot_id_mismatch: {first.get('snapshot_id')} != {manifest_snapshot_id}",
        )
    if summary_snapshot_id and manifest_snapshot_id and summary_snapshot_id != manifest_snapshot_id:
        raise ReplayContractError(
            "snapshot_id_mismatch",
            f"snapshot_id_mismatch: {summary_snapshot_id} != {manifest_snapshot_id}",
        )

    verifier_id = manifest.get("verifier_id") or summary.get("verifier_id") or freeze["evaluator_id"]
    verifier_version = manifest.get("verifier_version") or summary.get("verifier_version") or freeze["evaluator_version"]
    task_manifest_hash = manifest_hash or source_hash

    return {
        "task_id": first["task_id"],
        "trace_id": first["trace_id"],
        "source_run_id": first["run_id"],
        "policy_version": first.get("policy_version") or summary.get("policy_version") or manifest.get("policy_version"),
        "trace_schema_version": trace_schema_version,
        "replay_class": replay_class,
        "environment_id": first.get("environment_id") or manifest_environment_id or summary_environment_id,
        "snapshot_id": first.get("snapshot_id") or manifest_snapshot_id or summary_snapshot_id,
        "verifier_id": verifier_id,
        "verifier_version": verifier_version,
        "evaluator_id": freeze["evaluator_id"],
        "evaluator_version": freeze["evaluator_version"],
        "task_manifest_hash": task_manifest_hash,
        "freeze_info": freeze,
    }


def _build_failure_summary(
    *,
    trace_path: Path,
    output_root: Path,
    telemetry_mode: str,
    system_version: str,
    source_summary: dict[str, Any] | None,
    source_manifest: dict[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    failure_root = output_root / f"webarena_verified_replay_failure_{uuid.uuid4().hex[:8]}"
    failure_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "offline_replay",
        "run_id": failure_root.name,
        "task_id": (source_summary or {}).get("task_id") or (source_manifest or {}).get("task_id"),
        "system_version": system_version,
        "trace_id": (source_summary or {}).get("trace_id") or (source_manifest or {}).get("trace_id"),
        "source_trace": str(trace_path),
        "source_run_id": (source_summary or {}).get("run_id") or (source_manifest or {}).get("run_id"),
        "source_event_count": 0,
        "replay_event_count": 0,
        "passed": False,
        "score": 0.0,
        "duration_ms": 0.0,
        "validation_status": "fail",
        "validation_pass": False,
        "failure_reasons": [reason],
        "replay_class": (source_manifest or {}).get("replay_class"),
        "trace_schema_version": (source_manifest or {}).get("trace_schema_version"),
        "policy_version": (source_summary or {}).get("policy_version") or (source_manifest or {}).get("policy_version"),
        "verifier_id": (source_summary or {}).get("verifier_id") or (source_manifest or {}).get("verifier_id"),
        "verifier_version": (source_summary or {}).get("verifier_version") or (source_manifest or {}).get("verifier_version"),
        "evaluator_id": ((source_summary or {}).get("evaluator_freeze") or {}).get("evaluator_id"),
        "evaluator_version": ((source_summary or {}).get("evaluator_freeze") or {}).get("evaluator_version"),
        "environment_id": (source_summary or {}).get("environment_id") or (source_manifest or {}).get("environment_id"),
        "snapshot_id": (source_summary or {}).get("snapshot_id") or (source_manifest or {}).get("snapshot_id"),
        "task_manifest_hash": (source_summary or {}).get("task_manifest_hash") or (source_manifest or {}).get("task_manifest_hash"),
        "deterministic_step_sequence": [],
        "source_trace_sha256": _trace_sha256(trace_path) if trace_path.exists() else None,
        "evaluator_output": None,
        "trace_path": None,
        "manifest_path": None,
    }
    summary_path = failure_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def replay_trace(
    trace_path: Path,
    output_root: Path,
    telemetry_mode: str = "basic",
    strict: bool = False,
    system_version: str = "full_system",
) -> dict[str, Any]:
    t0 = time.perf_counter_ns()
    source_summary = load_json(trace_path.parent / "summary.json")
    source_manifest = load_json(trace_path.parent / "run_manifest.json")
    evaluator = MockWebArenaEvaluator()

    try:
        try:
            rows = load_jsonl(trace_path)
        except json.JSONDecodeError as exc:
            raise ReplayIntegrityError("corrupted_trace", f"corrupted_trace: {exc}") from exc

        validation = validate_jsonl(trace_path)
        if not validation.valid:
            first_issue = validation.issues[0].message if validation.issues else "validation_failed"
            if "trace should start with run_start" in first_issue:
                raise ReplayIntegrityError("missing_run_start", f"missing_run_start: {first_issue}")
            if "trace should end with run_end" in first_issue:
                raise ReplayIntegrityError("missing_run_end", f"missing_run_end: {first_issue}")
            if "step_id regression" in first_issue:
                raise ReplayIntegrityError("out_of_order_step_id", f"out_of_order_step_id: {first_issue}")
            if "Field required" in first_issue:
                raise ReplayIntegrityError("missing_required_field", f"missing_required_field: {first_issue}")
            raise ReplayIntegrityError("schema_validation_error", f"schema_validation_error: {first_issue}")

        _validate_trace_rows(rows)
        contract = _extract_contract(rows, source_summary, source_manifest, evaluator)
    except ReplayError as exc:
        if strict:
            raise
        return _build_failure_summary(
            trace_path=trace_path,
            output_root=output_root,
            telemetry_mode=telemetry_mode,
            system_version=system_version,
            source_summary=source_summary,
            source_manifest=source_manifest,
            reason=exc.reason,
        )

    first = rows[0]
    context = RunContext.create(
        output_root=output_root,
        task_family="webarena_verified",
        telemetry_mode=telemetry_mode,
        environment_id=contract["environment_id"],
        snapshot_id=contract["snapshot_id"],
        model_id=first.get("model_id"),
        model_version=first.get("model_version"),
        policy_version=contract["policy_version"],
        verifier_id=contract["verifier_id"],
        verifier_version=contract["verifier_version"],
        replay_class=contract["replay_class"],
    )
    bus = EventBus(context)
    episode_id = f"{contract['task_id']}-replay-0001"

    bus.emit(
        EventType.RUN_START,
        episode_id=episode_id,
        step_id=0,
        task_id=contract["task_id"],
        payload={
            "mode": "offline_replay",
            "source_trace": str(trace_path),
            "source_run_id": contract["source_run_id"],
        },
    )

    for idx, row in enumerate(rows, start=1):
        bus.emit(
            EventType.REPLAY_CONSUME,
            episode_id=episode_id,
            step_id=idx,
            task_id=contract["task_id"],
            parent_step_id=row.get("step_id"),
            payload={
                "source_event_type": row.get("event_type"),
                "source_span_id": row.get("span_id"),
                "source_timestamp_wall": row.get("timestamp_wall"),
            },
        )

    bus.emit(
        EventType.VERIFIER_START,
        episode_id=episode_id,
        step_id=len(rows) + 1,
        task_id=contract["task_id"],
        payload={"mode": "offline_replay"},
    )
    evaluation = evaluator.evaluate_replay(contract["task_id"], source_summary, len(rows))
    bus.emit(
        EventType.VERIFIER_END,
        episode_id=episode_id,
        step_id=len(rows) + 2,
        task_id=contract["task_id"],
        payload=evaluation,
    )
    t1 = time.perf_counter_ns()
    bus.emit(
        EventType.RUN_END,
        episode_id=episode_id,
        step_id=len(rows) + 3,
        task_id=contract["task_id"],
        payload={
            "mode": "offline_replay",
            "passed": evaluation["passed"],
            "duration_ms": (t1 - t0) / 1_000_000.0,
        },
    )

    replay_validation = validate_jsonl(context.paths.events_jsonl)
    validation_pass = replay_validation.valid
    summary = {
        "mode": "offline_replay",
        "run_id": context.run_id,
        "task_id": contract["task_id"],
        "system_version": system_version,
        "trace_id": contract["trace_id"],
        "source_trace": str(trace_path),
        "source_run_id": contract["source_run_id"],
        "source_event_count": len(rows),
        "replay_event_count": len(bus.events),
        "passed": bool(evaluation["passed"] and validation_pass),
        "score": float(evaluation["score"]) if validation_pass else 0.0,
        "duration_ms": (t1 - t0) / 1_000_000.0,
        "validation_status": "pass" if validation_pass else "fail",
        "validation_pass": validation_pass,
        "failure_reasons": [] if validation_pass else ["replay_trace_validation_failed"],
        "replay_class": contract["replay_class"],
        "trace_schema_version": contract["trace_schema_version"],
        "policy_version": contract["policy_version"],
        "verifier_id": contract["verifier_id"],
        "verifier_version": contract["verifier_version"],
        "evaluator_id": contract["evaluator_id"],
        "evaluator_version": contract["evaluator_version"],
        "environment_id": contract["environment_id"],
        "snapshot_id": contract["snapshot_id"],
        "task_manifest_hash": contract["task_manifest_hash"],
        "deterministic_step_sequence": _deterministic_step_sequence(rows),
        "source_trace_sha256": _trace_sha256(trace_path),
        "evaluator_output": {
            "task_id": evaluation["task_id"],
            "score": evaluation["score"],
            "passed": evaluation["passed"],
            "source_event_count": evaluation["source_event_count"],
            "freeze_info": evaluation["freeze_info"],
        },
        "trace_path": str(context.paths.events_jsonl),
        "manifest_path": str(
            context.write_manifest(
                {
                    "source_trace": str(trace_path),
                    "source_run_id": contract["source_run_id"],
                    "evaluator_freeze": contract["freeze_info"],
                    "task_manifest_hash": contract["task_manifest_hash"],
                    "system_version": system_version,
                }
            )
        ),
    }
    summary_path = context.paths.root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def variance_report(live_summary: dict[str, Any], replay_summary: dict[str, Any]) -> dict[str, Any]:
    live_duration = float(live_summary.get("duration_ms", math.nan))
    replay_duration = float(replay_summary.get("duration_ms", math.nan))
    return {
        "task_id": live_summary.get("task_id"),
        "live_passed": bool(live_summary.get("passed", False)),
        "replay_passed": bool(replay_summary.get("passed", False)),
        "pass_delta": int(bool(replay_summary.get("passed", False))) - int(bool(live_summary.get("passed", False))),
        "live_duration_ms": live_duration,
        "replay_duration_ms": replay_duration,
        "duration_delta_ms": replay_duration - live_duration if not math.isnan(live_duration) and not math.isnan(replay_duration) else math.nan,
        "live_event_count": int(live_summary.get("event_count", 0)),
        "replay_event_count": int(replay_summary.get("replay_event_count", 0)),
        "event_count_delta": int(replay_summary.get("replay_event_count", 0)) - int(live_summary.get("event_count", 0)),
    }
