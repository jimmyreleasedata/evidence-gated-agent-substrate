"""Base interfaces for container backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib


def stable_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ImageRef:
    image_name: str
    image_source: str
    image_digest: str
    sif_path: str | None = None
    sif_hash: str | None = None


@dataclass(slots=True)
class MaterializedRuntime:
    backend_name: str
    runtime_root: Path
    workdir: Path
    image: ImageRef
    bind_mounts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecResult:
    backend_name: str
    command: list[str]
    return_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ContainerBackend(ABC):
    name: str

    @abstractmethod
    def build_image(self, image_source: str, output_path: Path | None = None) -> ImageRef:
        raise NotImplementedError

    @abstractmethod
    def materialize_runtime(
        self,
        image: ImageRef,
        runtime_root: Path,
        workdir_name: str,
        bind_mounts: dict[str, str] | None = None,
    ) -> MaterializedRuntime:
        raise NotImplementedError

    @abstractmethod
    def exec(
        self,
        runtime: MaterializedRuntime,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
    ) -> ExecResult:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self, runtime: MaterializedRuntime) -> None:
        raise NotImplementedError
