"""Run-context helpers for standardized benchmark-suite outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import uuid

from trace.schema.events import (
    ReplayClass,
    TelemetryMode,
    TraceEvent,
    EventType,
    monotonic_now_ns,
    new_span_id,
    new_trace_id,
    utc_now_iso,
)
from trace.schema.version import SUITE_VERSION, TRACE_SCHEMA_VERSION


@dataclass(slots=True)
class RunPaths:
    root: Path
    events_jsonl: Path
    events_parquet: Path
    trace_duckdb: Path
    run_manifest: Path


@dataclass(slots=True)
class RunContext:
    run_id: str
    task_family: str
    telemetry_mode: TelemetryMode
    paths: RunPaths
    environment_id: str | None = None
    snapshot_id: str | None = None
    model_id: str | None = None
    model_version: str | None = None
    policy_version: str | None = None
    verifier_id: str | None = None
    verifier_version: str | None = None
    replay_class: ReplayClass | None = None
    trace_id: str = field(default_factory=new_trace_id)

    @classmethod
    def create(
        cls,
        output_root: Path,
        task_family: str,
        telemetry_mode: str = "basic",
        run_id: str | None = None,
        environment_id: str | None = None,
        snapshot_id: str | None = None,
        model_id: str | None = None,
        model_version: str | None = None,
        policy_version: str | None = None,
        verifier_id: str | None = None,
        verifier_version: str | None = None,
        replay_class: str | None = None,
    ) -> "RunContext":
        actual_run_id = run_id or (
            f"{task_family}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        root = output_root / actual_run_id
        root.mkdir(parents=True, exist_ok=True)
        paths = RunPaths(
            root=root,
            events_jsonl=root / "events.jsonl",
            events_parquet=root / "events.parquet",
            trace_duckdb=root / "trace.duckdb",
            run_manifest=root / "run_manifest.json",
        )
        return cls(
            run_id=actual_run_id,
            task_family=task_family,
            telemetry_mode=TelemetryMode(telemetry_mode),
            paths=paths,
            environment_id=environment_id,
            snapshot_id=snapshot_id,
            model_id=model_id,
            model_version=model_version,
            policy_version=policy_version,
            verifier_id=verifier_id,
            verifier_version=verifier_version,
            replay_class=ReplayClass(replay_class) if replay_class else None,
        )

    def make_event(
        self,
        event_type: EventType | str,
        episode_id: str,
        step_id: int,
        task_id: str,
        parent_step_id: int | None = None,
        parent_span_id: str | None = None,
        tool_id: str | None = None,
        return_code: int | None = None,
        error_class: str | None = None,
        retry_count: int = 0,
        payload: dict[str, Any] | None = None,
        **metrics: Any,
    ) -> TraceEvent:
        return TraceEvent(
            suite_version=SUITE_VERSION,
            trace_schema_version=TRACE_SCHEMA_VERSION,
            run_id=self.run_id,
            episode_id=episode_id,
            step_id=step_id,
            trace_id=self.trace_id,
            span_id=new_span_id(),
            timestamp_wall=utc_now_iso(),
            timestamp_mono=monotonic_now_ns(),
            event_type=EventType(event_type),
            task_family=self.task_family,
            task_id=task_id,
            parent_step_id=parent_step_id,
            parent_span_id=parent_span_id,
            environment_id=self.environment_id,
            snapshot_id=self.snapshot_id,
            model_id=self.model_id,
            model_version=self.model_version,
            policy_version=self.policy_version,
            verifier_id=self.verifier_id,
            verifier_version=self.verifier_version,
            tool_id=tool_id,
            return_code=return_code,
            error_class=error_class,
            retry_count=retry_count,
            telemetry_mode=self.telemetry_mode,
            replay_class=self.replay_class,
            payload=payload or {},
            **metrics,
        )

    def telemetry_enabled(self) -> bool:
        return self.telemetry_mode is not TelemetryMode.OFF

    def manifest_dict(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        manifest = {
            "suite_version": SUITE_VERSION,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "task_family": self.task_family,
            "environment_id": self.environment_id,
            "snapshot_id": self.snapshot_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "policy_version": self.policy_version,
            "verifier_id": self.verifier_id,
            "verifier_version": self.verifier_version,
            "telemetry_mode": self.telemetry_mode.value,
            "replay_class": self.replay_class.value if self.replay_class else None,
            "paths": {
                "root": str(self.paths.root),
                "events_jsonl": str(self.paths.events_jsonl),
                "events_parquet": str(self.paths.events_parquet),
                "trace_duckdb": str(self.paths.trace_duckdb),
            },
        }
        if extra:
            manifest.update(extra)
        return manifest

    def write_manifest(self, extra: dict[str, Any] | None = None) -> Path:
        self.paths.run_manifest.write_text(
            json.dumps(self.manifest_dict(extra), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self.paths.run_manifest
