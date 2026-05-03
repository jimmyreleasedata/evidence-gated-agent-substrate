"""Docker backend abstraction with dry-run compatibility."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import time

from runtime.container_backend.base import (
    ContainerBackend,
    ExecResult,
    ImageRef,
    MaterializedRuntime,
    stable_sha256,
)


class DockerBackend(ContainerBackend):
    name = "docker"

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run

    def build_image(self, image_source: str, output_path: Path | None = None) -> ImageRef:
        return ImageRef(
            image_name=Path(image_source.replace("docker://", "").replace("/", "_")).name,
            image_source=image_source,
            image_digest=f"sha256:{stable_sha256(image_source)[:32]}",
            sif_path=str(output_path) if output_path else None,
            sif_hash=None,
        )

    def materialize_runtime(
        self,
        image: ImageRef,
        runtime_root: Path,
        workdir_name: str,
        bind_mounts: dict[str, str] | None = None,
    ) -> MaterializedRuntime:
        runtime_root.mkdir(parents=True, exist_ok=True)
        workdir = runtime_root / workdir_name
        workdir.mkdir(parents=True, exist_ok=True)
        return MaterializedRuntime(
            backend_name=self.name,
            runtime_root=runtime_root,
            workdir=workdir,
            image=image,
            bind_mounts=bind_mounts or {},
            metadata={"dry_run": self.dry_run},
        )

    def exec(
        self,
        runtime: MaterializedRuntime,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
    ) -> ExecResult:
        t0 = time.perf_counter_ns()
        if self.dry_run:
            t1 = time.perf_counter_ns()
            return ExecResult(
                backend_name=f"{self.name}-dry-run",
                command=list(command),
                return_code=0,
                stdout="",
                stderr="",
                duration_ms=(t1 - t0) / 1_000_000.0,
                metadata={"runtime": asdict(runtime)},
            )

        docker_cmd = ["docker", "run", "--rm", runtime.image.image_source]
        docker_cmd.extend(command)
        completed = subprocess.run(
            docker_cmd,
            cwd=cwd or runtime.workdir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        t1 = time.perf_counter_ns()
        return ExecResult(
            backend_name=self.name,
            command=docker_cmd,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=(t1 - t0) / 1_000_000.0,
            metadata={"runtime": asdict(runtime)},
        )

    def cleanup(self, runtime: MaterializedRuntime) -> None:
        if runtime.workdir.exists():
            shutil.rmtree(runtime.workdir, ignore_errors=True)
