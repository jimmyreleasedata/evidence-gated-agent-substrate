"""Runtime configuration for real WebArena Verified replay."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import json


DEFAULT_TASK_MANIFEST = Path("manifests/tasks/webarena_verified_real_v1.yaml")


def _env_any(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value
    return None


def _bootstrap_metadata(upstream_root: Path) -> dict[str, str]:
    path = upstream_root.parent / "bootstrap_metadata.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items() if value is not None}


@dataclass(slots=True)
class WebArenaVerifiedRealConfig:
    output_root: Path
    telemetry_mode: str
    upstream_root: Path
    task_selection_path: Path
    trace_root: Path
    evaluator_version: str
    upstream_commit: str
    task_manifest_path: Path

    @classmethod
    def from_env(
        cls,
        *,
        output_root: Path = Path("artifacts/webarena_verified_real"),
        telemetry_mode: str = "basic",
        task_manifest_path: Path = DEFAULT_TASK_MANIFEST,
    ) -> "WebArenaVerifiedRealConfig":
        upstream_root = _env_any("WEBAV_VERIFIED_ROOT", "NIPS_WEBARENA_VERIFIED_ROOT")
        task_selection = _env_any("NIPS_WEBARENA_VERIFIED_TASKS")
        trace_root = _env_any("NIPS_WEBARENA_VERIFIED_TRACE_ROOT")
        upstream_root_path = Path(upstream_root).expanduser().resolve(strict=False) if upstream_root else None
        bootstrap = _bootstrap_metadata(upstream_root_path) if upstream_root_path is not None else {}
        evaluator_version = _env_any("NIPS_WEBARENA_VERIFIED_EVALUATOR_VERSION") or bootstrap.get("evaluator_version")
        upstream_commit = _env_any("NIPS_WEBARENA_VERIFIED_COMMIT") or bootstrap.get("repo_commit")

        missing = [
            name
            for name, value in [
                ("WEBAV_VERIFIED_ROOT|NIPS_WEBARENA_VERIFIED_ROOT", upstream_root),
                ("NIPS_WEBARENA_VERIFIED_TASKS", task_selection),
                ("NIPS_WEBARENA_VERIFIED_TRACE_ROOT", trace_root),
                ("NIPS_WEBARENA_VERIFIED_EVALUATOR_VERSION", evaluator_version),
                ("NIPS_WEBARENA_VERIFIED_COMMIT", upstream_commit),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"missing required WebArena real replay env vars: {', '.join(missing)}")

        return cls(
            output_root=output_root,
            telemetry_mode=telemetry_mode,
            upstream_root=upstream_root_path,
            task_selection_path=Path(task_selection).expanduser().resolve(strict=False),
            trace_root=Path(trace_root).expanduser().resolve(strict=False),
            evaluator_version=str(evaluator_version),
            upstream_commit=str(upstream_commit),
            task_manifest_path=task_manifest_path,
        )


# Backward-compatible alias used by older helpers.
RealWebArenaVerifiedConfig = WebArenaVerifiedRealConfig
