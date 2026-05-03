"""Configuration for the WebArena Verified adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from runtime.external_paths import DEFAULT_LOCAL_PATHS_MANIFEST, load_local_paths


@dataclass(slots=True)
class WebArenaVerifiedConfig:
    output_root: Path = Path("artifacts/webarena_verified")
    telemetry_mode: str = "basic"
    backend: str = "mock"
    system_version: str = "full_system"
    regime: str = "baseline"
    offered_load: int = 1
    task_id: str = "search_product_specs"
    source_trace: Path | None = None
    source_summary: Path | None = None
    repo_root: Path | None = None

    @classmethod
    def from_runtime_defaults(
        cls,
        *,
        output_root: Path = Path("artifacts/webarena_verified"),
        telemetry_mode: str = "basic",
        system_version: str = "full_system",
        regime: str = "baseline",
        offered_load: int = 1,
        task_id: str = "search_product_specs",
        source_trace: Path | None = None,
        source_summary: Path | None = None,
        manifest_path: Path = DEFAULT_LOCAL_PATHS_MANIFEST,
    ) -> "WebArenaVerifiedConfig":
        local_paths = load_local_paths(manifest_path)
        repo_root = local_paths.browsergym_repo
        return cls(
            output_root=output_root,
            telemetry_mode=telemetry_mode,
            backend="browsergym" if repo_root is not None else "mock",
            system_version=system_version,
            regime=regime,
            offered_load=offered_load,
            task_id=task_id,
            source_trace=source_trace,
            source_summary=source_summary,
            repo_root=repo_root,
        )
