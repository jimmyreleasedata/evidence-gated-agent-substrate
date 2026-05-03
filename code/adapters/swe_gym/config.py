"""Configuration for the SWE-Gym mock harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from runtime.external_paths import DEFAULT_LOCAL_PATHS_MANIFEST, load_local_paths


@dataclass(slots=True)
class SweGymConfig:
    output_root: Path = Path("artifacts/swe_gym")
    telemetry_mode: str = "basic"
    backend: str = "apptainer"
    system_version: str = "full_system"
    regime: str = "baseline"
    offered_load: int = 1
    task_id: str = "fix_answer"
    dry_run: bool = False
    allow_host_fallback: bool = True
    source_manifest: Path | None = None
    repo_root: Path | None = None

    @classmethod
    def from_runtime_defaults(
        cls,
        *,
        output_root: Path = Path("artifacts/swe_gym"),
        telemetry_mode: str = "basic",
        system_version: str = "full_system",
        regime: str = "baseline",
        offered_load: int = 1,
        task_id: str = "fix_answer",
        backend: str = "apptainer",
        dry_run: bool = False,
        allow_host_fallback: bool = True,
        source_manifest: Path | None = None,
        manifest_path: Path = DEFAULT_LOCAL_PATHS_MANIFEST,
    ) -> "SweGymConfig":
        local_paths = load_local_paths(manifest_path)
        return cls(
            output_root=output_root,
            telemetry_mode=telemetry_mode,
            backend=backend,
            system_version=system_version,
            regime=regime,
            offered_load=offered_load,
            task_id=task_id,
            dry_run=dry_run,
            allow_host_fallback=allow_host_fallback,
            source_manifest=source_manifest,
            repo_root=local_paths.swebench_repo,
        )
