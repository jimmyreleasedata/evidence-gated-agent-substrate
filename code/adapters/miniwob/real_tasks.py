"""Load real MiniWoB task selections from a manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import yaml


@dataclass(frozen=True, slots=True)
class RealMiniWobTaskSpec:
    task_id: str
    env_id: str
    instruction: str
    action_sequence: list[str]
    task_manifest_hash: str


def _hash_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_real_task_specs(path: Path) -> list[RealMiniWobTaskSpec]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path} does not define any MiniWoB real tasks")
    manifest_hash = _hash_path(path)
    rows: list[RealMiniWobTaskSpec] = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        env_id = str(task.get("env_id") or task_id)
        instruction = str(task.get("instruction") or "")
        action_sequence = [str(item) for item in task.get("action_sequence", [])]
        if not task_id:
            raise ValueError(f"{path} contains a MiniWoB task without task_id")
        rows.append(
            RealMiniWobTaskSpec(
                task_id=task_id,
                env_id=env_id,
                instruction=instruction,
                action_sequence=action_sequence,
                task_manifest_hash=manifest_hash,
            )
        )
    return rows
