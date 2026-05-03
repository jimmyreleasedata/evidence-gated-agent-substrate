"""Configuration helpers for the MiniWoB++ adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class MiniWobConfig:
    output_root: Path = Path("artifacts/miniwob")
    telemetry_mode: str = "basic"
    backend: str = "mock"
    concurrency: int = 1
    repeat: int = 1
    task_ids: list[str] = field(default_factory=list)
