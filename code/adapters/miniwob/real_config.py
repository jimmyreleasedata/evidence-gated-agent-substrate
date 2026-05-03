"""Runtime configuration for a real MiniWoB backend."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_TASK_MANIFEST = Path("manifests/tasks/miniwob_real_v1.yaml")


def _env_any(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value
    return None


@dataclass(slots=True)
class MiniWobRealConfig:
    output_root: Path
    telemetry_mode: str
    backend: str
    headless: bool
    upstream_root: Path
    task_selection_path: Path
    task_manifest_path: Path

    @classmethod
    def from_env(
        cls,
        *,
        output_root: Path = Path("artifacts/miniwob_real"),
        telemetry_mode: str = "basic",
        task_manifest_path: Path = DEFAULT_TASK_MANIFEST,
    ) -> "MiniWobRealConfig":
        upstream_root = _env_any("MINIWOB_ROOT", "NIPS_MINIWOB_ROOT")
        backend = _env_any("NIPS_MINIWOB_BACKEND") or "browsergym_miniwob"
        headless_value = (_env_any("NIPS_MINIWOB_HEADLESS") or "true").lower()
        task_selection = _env_any("NIPS_MINIWOB_TASKS") or str(task_manifest_path)

        missing = [
            name
            for name, value in [
                ("MINIWOB_ROOT|NIPS_MINIWOB_ROOT", upstream_root),
                ("NIPS_MINIWOB_TASKS", task_selection),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"missing required MiniWoB real env vars: {', '.join(missing)}")

        return cls(
            output_root=output_root,
            telemetry_mode=telemetry_mode,
            backend=backend,
            headless=headless_value not in {"0", "false", "no"},
            upstream_root=Path(upstream_root).expanduser().resolve(strict=False),
            task_selection_path=Path(task_selection).expanduser().resolve(strict=False),
            task_manifest_path=task_manifest_path,
        )
