"""JSONL export helpers for benchmark-suite traces."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any
import json

from trace.schema.events import TraceEvent


def _event_to_row(event: TraceEvent | dict[str, Any]) -> dict[str, Any]:
    if isinstance(event, TraceEvent):
        return event.to_dict()
    return dict(event)


def write_jsonl(path: Path, events: Iterable[TraceEvent | dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(_event_to_row(event), ensure_ascii=False) + "\n")
    return path


class JsonlWriter:
    """Simple append-only JSONL writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_event(self, event: TraceEvent | dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_event_to_row(event), ensure_ascii=False) + "\n")
