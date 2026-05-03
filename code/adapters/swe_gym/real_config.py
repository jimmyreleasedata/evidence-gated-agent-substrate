"""Configuration for the real SWE slice path."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_TASK_MANIFEST = Path("manifests/tasks/swe_real_v1.yaml")
DEFAULT_IMAGE_MANIFEST = Path("manifests/images/swe_real_images.yaml")


def _env_any(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value
    return None


@dataclass(slots=True)
class RealSweConfig:
    output_root: Path
    telemetry_mode: str
    backend: str
    task_selection_path: Path
    image_root: Path
    upstream_root: Path | None
    upstream_gym_root: Path | None
    image_manifest_path: Path
    task_manifest_path: Path
    harness_version: str
    dataset_version: str
    system_version: str = "full_system"
    regime: str = "baseline"
    offered_load: int = 1
    verifier_workers: int = 1
    queue_cap: str = "normal"
    limit: int | None = None
    patch_conditions: tuple[str, ...] = ("oracle", "noop", "known_bad")
    allow_host_fallback: bool = True

    @classmethod
    def from_env(
        cls,
        *,
        output_root: Path = Path("artifacts/swe_real"),
        telemetry_mode: str = "basic",
        task_manifest_path: Path = DEFAULT_TASK_MANIFEST,
        image_manifest_path: Path = DEFAULT_IMAGE_MANIFEST,
    ) -> "RealSweConfig":
        task_selection = _env_any("NIPS_SWE_REAL_TASKS")
        image_root = _env_any("NIPS_SWE_IMAGE_ROOT")
        backend = _env_any("NIPS_SWE_CONTAINER_BACKEND") or "apptainer"
        upstream_root = _env_any("SWE_BENCH_ROOT", "NIPS_SWEBENCH_ROOT")
        upstream_gym_root = _env_any("SWE_GYM_ROOT", "NIPS_SWEGYM_ROOT")
        harness_version = _env_any("NIPS_SWE_HARNESS_VERSION") or "official_or_compatible"
        dataset_version = _env_any("NIPS_SWE_DATASET_VERSION") or "official_selected_slice"
        system_version = _env_any("NIPS_SWE_SYSTEM_VERSION") or "full_system"
        regime = _env_any("NIPS_SWE_REGIME") or "baseline"
        offered_load = int(_env_any("NIPS_SWE_OFFERED_LOAD") or "1")
        verifier_workers = int(_env_any("NIPS_SWE_VERIFIER_WORKERS") or "1")
        queue_cap = _env_any("NIPS_SWE_QUEUE_CAP") or "normal"
        image_manifest_override = _env_any("NIPS_SWE_IMAGES_MANIFEST")

        missing = [
            name
            for name, value in [
                ("NIPS_SWE_REAL_TASKS", task_selection),
                ("NIPS_SWE_IMAGE_ROOT", image_root),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"missing required SWE real env vars: {', '.join(missing)}")

        return cls(
            output_root=output_root,
            telemetry_mode=telemetry_mode,
            backend=backend,
            task_selection_path=Path(task_selection).expanduser().resolve(strict=False),
            image_root=Path(image_root).expanduser().resolve(strict=False),
            upstream_root=Path(upstream_root).expanduser().resolve(strict=False) if upstream_root else None,
            upstream_gym_root=Path(upstream_gym_root).expanduser().resolve(strict=False) if upstream_gym_root else None,
            image_manifest_path=Path(image_manifest_override).expanduser().resolve(strict=False)
            if image_manifest_override
            else image_manifest_path,
            task_manifest_path=task_manifest_path,
            harness_version=harness_version,
            dataset_version=dataset_version,
            system_version=system_version,
            regime=regime,
            offered_load=offered_load,
            verifier_workers=verifier_workers,
            queue_cap=queue_cap,
        )


# Backward-compatible alias used by existing tests and older helpers.
SweRealConfig = RealSweConfig
