"""Event-bus helpers for benchmark-suite trace emission."""

from __future__ import annotations

from typing import Any

from runner.run_context import RunContext
from trace.exporters.jsonl_writer import JsonlWriter
from trace.schema.events import EventType, TraceEvent


class _NoOpWriter:
    def write_event(self, event: TraceEvent | dict[str, Any]) -> None:
        return None


class EventBus:
    """Collects events in memory and mirrors them to JSONL."""

    def __init__(self, run_context: RunContext) -> None:
        self.run_context = run_context
        self.events: list[TraceEvent] = []
        self.writer = JsonlWriter(run_context.paths.events_jsonl) if run_context.telemetry_enabled() else _NoOpWriter()

    def emit(
        self,
        event_type: EventType | str,
        episode_id: str,
        step_id: int,
        task_id: str,
        **kwargs: Any,
    ) -> TraceEvent:
        event = self.run_context.make_event(
            event_type=event_type,
            episode_id=episode_id,
            step_id=step_id,
            task_id=task_id,
            **kwargs,
        )
        return self.emit_event(event)

    def emit_event(self, event: TraceEvent) -> TraceEvent:
        if self.run_context.telemetry_enabled():
            self.events.append(event)
            self.writer.write_event(event)
        return event

    def to_rows(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]
