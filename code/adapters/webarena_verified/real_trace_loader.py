"""Load real WebArena Verified trace/evaluator bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from adapters.webarena_verified.real_config import WebArenaVerifiedRealConfig


@dataclass(slots=True)
class RealWebArenaTraceBundle:
    task_id: str
    upstream_task_id: str
    trace_path: Path
    evaluator_result_path: Path
    task_manifest_path: Path
    step_reconstruction_path: Path | None
    trace_hash: str
    task_manifest_hash: str
    trace_bundle_ref: str
    task_manifest: dict[str, Any]
    evaluator_result: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_structured(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _read_selection_spec(path: Path) -> list[dict[str, Any]]:
    if path.suffix in {".yaml", ".yml", ".json"}:
        payload = _read_structured(path)
        if isinstance(payload, dict) and "tasks" in payload:
            tasks = payload["tasks"] or []
            if not isinstance(tasks, list):
                raise ValueError(f"tasks must be a list in {path}")
            return [item if isinstance(item, dict) else {"task_id": str(item)} for item in tasks]
        if isinstance(payload, list):
            return [item if isinstance(item, dict) else {"task_id": str(item)} for item in payload]
        raise ValueError(f"unsupported task selection payload in {path}")
    return [
        {"task_id": line.strip()}
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _resolve_bundle_paths(
    config: WebArenaVerifiedRealConfig,
    task_entry: dict[str, Any],
) -> tuple[str, Path, Path, Path, Path | None, str]:
    task_id = str(task_entry.get("task_id") or task_entry.get("upstream_task_id") or "").strip()
    if not task_id:
        raise ValueError(f"task entry missing task_id: {task_entry}")

    task_root = config.trace_root / task_id
    trace_path = Path(task_entry.get("trace_path") or task_root / "events.jsonl").expanduser().resolve(strict=False)
    evaluator_result_path = Path(
        task_entry.get("evaluator_result_path") or task_root / "evaluation.json"
    ).expanduser().resolve(strict=False)
    task_manifest_path = Path(
        task_entry.get("task_manifest_path") or config.upstream_root / "tasks" / f"{task_id}.json"
    ).expanduser().resolve(strict=False)
    step_reconstruction = task_entry.get("step_reconstruction_path")
    step_reconstruction_path = Path(step_reconstruction).expanduser().resolve(strict=False) if step_reconstruction else None
    trace_bundle_ref = str(task_entry.get("trace_bundle_ref") or trace_path)
    return task_id, trace_path, evaluator_result_path, task_manifest_path, step_reconstruction_path, trace_bundle_ref


def load_real_trace_bundles(config: WebArenaVerifiedRealConfig) -> list[RealWebArenaTraceBundle]:
    bundles: list[RealWebArenaTraceBundle] = []
    for task_entry in _read_selection_spec(config.task_selection_path):
        task_id, trace_path, evaluator_result_path, task_manifest_path, step_reconstruction_path, trace_bundle_ref = _resolve_bundle_paths(
            config,
            task_entry,
        )
        missing_paths = [
            str(path)
            for path in (trace_path, evaluator_result_path, task_manifest_path)
            if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(f"missing WebArena real replay assets for {task_id}: {', '.join(missing_paths)}")

        task_manifest = _read_structured(task_manifest_path)
        if not isinstance(task_manifest, dict):
            raise ValueError(f"invalid task manifest for {task_id}: {task_manifest_path}")
        upstream_task_id = str(task_manifest.get("upstream_task_id") or task_manifest.get("task_id") or task_id)
        if upstream_task_id != task_id:
            raise ValueError(f"task manifest mismatch for {task_id}: got {upstream_task_id}")

        evaluator_result = _read_structured(evaluator_result_path)
        if not isinstance(evaluator_result, dict):
            raise ValueError(f"invalid evaluator result for {task_id}: {evaluator_result_path}")
        bundles.append(
            RealWebArenaTraceBundle(
                task_id=task_id,
                upstream_task_id=upstream_task_id,
                trace_path=trace_path,
                evaluator_result_path=evaluator_result_path,
                task_manifest_path=task_manifest_path,
                step_reconstruction_path=step_reconstruction_path,
                trace_hash=_sha256(trace_path),
                task_manifest_hash=_sha256(task_manifest_path),
                trace_bundle_ref=trace_bundle_ref,
                task_manifest=task_manifest,
                evaluator_result=evaluator_result,
            )
        )
    return bundles
