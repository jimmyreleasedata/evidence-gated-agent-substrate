"""Local benchmark repository path resolution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_LOCAL_PATHS_MANIFEST = Path("manifests/local_paths.yaml")
DEFAULT_VENDOR_ROOT = Path("vendor")
DEFAULT_ASPLOS_ROOT = Path("artifact_release_root/asplos")
DEFAULT_SOSP26_ROOT = Path("artifact_release_root/sosp26")


@dataclass(slots=True)
class LocalRepoPaths:
    browsergym_repo: Path | None
    swebench_repo: Path | None

    def as_manifest_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        if self.browsergym_repo is not None:
            payload["browsergym_repo"] = str(self.browsergym_repo)
        if self.swebench_repo is not None:
            payload["swebench_repo"] = str(self.swebench_repo)
        return payload


@dataclass(slots=True)
class RuntimeStackPaths:
    asplos_root: Path | None
    sosp26_root: Path | None
    sglang_python: Path | None
    verl_bridge_path: Path | None


def _parse_manifest_map(text: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip().strip("'").strip('"')
    return payload


def _coerce_path(value: str | None, *, base_dir: Path | None = None) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve(strict=False)


def _default_vendor_path(vendor_root: Path, repo_dirname: str) -> Path | None:
    candidate = vendor_root / repo_dirname
    if candidate.exists():
        return candidate.resolve(strict=False)
    return None


def _existing_path(candidate: Path | None) -> Path | None:
    if candidate is None:
        return None
    if candidate.exists():
        return candidate.resolve(strict=False)
    return None


def discover_runtime_stack_paths() -> RuntimeStackPaths:
    asplos_root = _existing_path(
        _coerce_path(os.environ.get("NIPS_ASPLOS_ROOT")) or DEFAULT_ASPLOS_ROOT
    )
    sosp26_root = _existing_path(
        _coerce_path(os.environ.get("NIPS_SOSP26_ROOT")) or DEFAULT_SOSP26_ROOT
    )
    sglang_python = _existing_path(_coerce_path(os.environ.get("NIPS_SGLANG_PYTHON")))
    if sglang_python is None and sosp26_root is not None:
        sglang_python = _existing_path(sosp26_root / "env" / "conda-ray" / "bin" / "python")
    if sglang_python is None and asplos_root is not None:
        sglang_python = _existing_path(asplos_root / "env" / "conda-ray" / "bin" / "python")
    verl_bridge_path = None
    if asplos_root is not None:
        verl_bridge_path = _existing_path(asplos_root / "02_runtime" / "integration" / "verl_bridge.py")
    return RuntimeStackPaths(
        asplos_root=asplos_root,
        sosp26_root=sosp26_root,
        sglang_python=sglang_python,
        verl_bridge_path=verl_bridge_path,
    )


def load_local_paths(
    manifest_path: Path = DEFAULT_LOCAL_PATHS_MANIFEST,
    *,
    vendor_root: Path = DEFAULT_VENDOR_ROOT,
    required_fields: tuple[str, ...] = (),
) -> LocalRepoPaths:
    manifest_payload: dict[str, str] = {}
    if manifest_path.exists():
        manifest_payload = _parse_manifest_map(manifest_path.read_text(encoding="utf-8"))
    manifest_root = manifest_path.parent.resolve(strict=False)

    browsergym_repo = _coerce_path(
        os.environ.get("NIPS_BENCH_BROWSERGYM_REPO"),
    )
    swebench_repo = _coerce_path(
        os.environ.get("NIPS_BENCH_SWEBENCH_REPO"),
    )
    if browsergym_repo is None:
        browsergym_repo = _coerce_path(manifest_payload.get("browsergym_repo"), base_dir=manifest_root)
    if swebench_repo is None:
        swebench_repo = _coerce_path(manifest_payload.get("swebench_repo"), base_dir=manifest_root)

    paths = LocalRepoPaths(
        browsergym_repo=browsergym_repo or _default_vendor_path(vendor_root, "browsergym"),
        swebench_repo=swebench_repo or _default_vendor_path(vendor_root, "SWE-bench"),
    )
    missing_fields = [field for field in required_fields if getattr(paths, field) is None]
    if missing_fields:
        field_list = ", ".join(missing_fields)
        raise ValueError(f"missing required local path fields: {field_list}")
    return paths


def write_local_paths_manifest(
    paths: LocalRepoPaths,
    manifest_path: Path = DEFAULT_LOCAL_PATHS_MANIFEST,
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = paths.as_manifest_dict()
    body = "".join(f"{key}: {value}\n" for key, value in payload.items())
    manifest_path.write_text(body, encoding="utf-8")
    return manifest_path
