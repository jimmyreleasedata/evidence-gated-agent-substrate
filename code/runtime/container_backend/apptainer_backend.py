"""Apptainer-first container backend with optional host fallback."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ApptainerBackend(ContainerBackend):
    name = "apptainer"

    def __init__(self, dry_run: bool = False, allow_host_fallback: bool = True) -> None:
        self.dry_run = dry_run
        self.allow_host_fallback = allow_host_fallback

    def build_image(self, image_source: str, output_path: Path | None = None) -> ImageRef:
        sif_path = str(output_path) if output_path is not None else None
        digest = f"sha256:{stable_sha256(image_source)[:32]}"
        sif_hash = f"sha256:{stable_sha256(sif_path or image_source)[-32:]}"
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not output_path.exists() and not self.dry_run:
                subprocess.run(
                    ["apptainer", "pull", str(output_path), image_source],
                    check=True,
                    text=True,
                    capture_output=True,
                )
            if output_path.exists():
                sif_hash = f"sha256:{_sha256_file(output_path)}"
        return ImageRef(
            image_name=Path(image_source.replace("docker://", "").replace("/", "_")).name,
            image_source=image_source,
            image_digest=digest,
            sif_path=sif_path,
            sif_hash=sif_hash,
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
            metadata={
                "dry_run": self.dry_run,
                "allow_host_fallback": self.allow_host_fallback,
            },
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
        target_cwd = cwd or runtime.workdir

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

        if runtime.image.sif_path and Path(runtime.image.sif_path).exists():
            apptainer_cmd = ["apptainer", "exec"]
            for host_path, container_path in runtime.bind_mounts.items():
                apptainer_cmd.extend(["--bind", f"{host_path}:{container_path}"])
            apptainer_cmd.append(runtime.image.sif_path)
            apptainer_cmd.extend(command)
            completed = subprocess.run(
                apptainer_cmd,
                cwd=target_cwd,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
            t1 = time.perf_counter_ns()
            return ExecResult(
                backend_name=self.name,
                command=apptainer_cmd,
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_ms=(t1 - t0) / 1_000_000.0,
                metadata={"runtime": asdict(runtime)},
            )

        if not self.allow_host_fallback:
            t1 = time.perf_counter_ns()
            return ExecResult(
                backend_name=self.name,
                command=list(command),
                return_code=1,
                stdout="",
                stderr="Apptainer image not available and host fallback disabled",
                duration_ms=(t1 - t0) / 1_000_000.0,
                metadata={"runtime": asdict(runtime)},
            )

        completed = subprocess.run(
            command,
            cwd=target_cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        t1 = time.perf_counter_ns()
        return ExecResult(
            backend_name=f"{self.name}-host-fallback",
            command=list(command),
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=(t1 - t0) / 1_000_000.0,
            metadata={"runtime": asdict(runtime)},
        )

    def cleanup(self, runtime: MaterializedRuntime) -> None:
        if runtime.workdir.exists():
            shutil.rmtree(runtime.workdir, ignore_errors=True)
