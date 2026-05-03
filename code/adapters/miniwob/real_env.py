"""Real MiniWoB backend resolution and wrappers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from adapters.miniwob.real_tasks import RealMiniWobTaskSpec


REAL_BACKENDS = {"browsergym_miniwob", "farama_selenium"}


@dataclass(frozen=True, slots=True)
class RealMiniWobBackendInfo:
    backend: str
    upstream_package_version: str
    browser_version: str
    driver_version: str


def _prepend_upstream_root(upstream_root: Path | None) -> None:
    if upstream_root is None:
        return
    root_str = str(upstream_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _package_version(package_name: str, default: str = "unknown") -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return default


def _resolve_playwright_chrome() -> Path | None:
    browsers_root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if not browsers_root:
        return None
    root = Path(browsers_root)
    candidates = sorted(root.glob("chromium-*/chrome-linux/chrome"), reverse=True)
    return candidates[0] if candidates else None


def _command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"


def normalize_env_id_for_backend(env_id: str, backend: str) -> str:
    if backend != "browsergym_miniwob":
        return env_id
    if env_id.startswith("browsergym/miniwob."):
        return env_id
    match = re.fullmatch(r"miniwob/([a-z0-9-]+)-v\d+", env_id)
    if not match:
        raise RuntimeError(
            f"MiniWoB BrowserGym backend requires BrowserGym-compatible env ids; got `{env_id}`"
        )
    return f"browsergym/miniwob.{match.group(1)}"


def resolve_browsergym_base_url(upstream_root: Path) -> str:
    value = os.environ.get("MINIWOB_URL", "").strip()
    if value:
        return value
    raise RuntimeError(
        "MiniWoB BrowserGym backend requires MINIWOB_URL to point at a real MiniWoB HTML base URL"
    )


def ensure_backend_available(backend: str, upstream_root: Path | None = None, *, force_missing: bool = False) -> None:
    if force_missing:
        raise RuntimeError(
            f"MiniWoB real backend `{backend}` is unavailable: missing upstream package/runtime prerequisites"
        )
    if backend not in REAL_BACKENDS:
        raise RuntimeError(f"unsupported MiniWoB real backend: {backend}")
    if upstream_root is None or not upstream_root.exists():
        raise RuntimeError(
            f"MiniWoB real backend `{backend}` requires MINIWOB_ROOT/NIPS_MINIWOB_ROOT to point to an upstream installation"
        )

    _prepend_upstream_root(upstream_root)
    if backend == "browsergym_miniwob":
        resolve_browsergym_base_url(upstream_root)
        module_name = "browsergym.miniwob"
    else:
        module_name = "miniwob"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"MiniWoB real backend `{backend}` requires importable module `{module_name}` under {upstream_root}"
        ) from exc


def backend_info(backend: str, upstream_root: Path) -> RealMiniWobBackendInfo:
    ensure_backend_available(backend, upstream_root)
    package_name = "browsergym-miniwob" if backend == "browsergym_miniwob" else "miniwob"
    browser_version = os.environ.get("NIPS_MINIWOB_BROWSER_VERSION", "unknown")
    driver_version = os.environ.get("NIPS_MINIWOB_DRIVER_VERSION", "unknown")
    if backend == "browsergym_miniwob":
        if browser_version == "unknown":
            chrome_path = _resolve_playwright_chrome()
            if chrome_path is not None:
                browser_version = _command_version([str(chrome_path), "--version"])
        if driver_version == "unknown":
            playwright_version = _package_version("playwright")
            if playwright_version != "unknown":
                driver_version = f"playwright-{playwright_version}"
    else:
        if browser_version == "unknown" and os.environ.get("CHROME_BIN"):
            browser_version = _command_version([os.environ["CHROME_BIN"], "--version"])
        if driver_version == "unknown" and os.environ.get("CHROMEDRIVER"):
            driver_version = _command_version([os.environ["CHROMEDRIVER"], "--version"])
    return RealMiniWobBackendInfo(
        backend=backend,
        upstream_package_version=_package_version(package_name),
        browser_version=browser_version,
        driver_version=driver_version,
    )


def make_real_env(task: RealMiniWobTaskSpec, *, backend: str, upstream_root: Path, headless: bool, seed: int) -> Any:
    ensure_backend_available(backend, upstream_root)
    _prepend_upstream_root(upstream_root)

    if backend == "browsergym_miniwob":
        resolve_browsergym_base_url(upstream_root)
        gym = importlib.import_module("gymnasium")
        importlib.import_module("browsergym")
        kwargs = {
            "disable_env_checker": True,
            "action_mapping": None,
        }
        if headless:
            kwargs["headless"] = True
        env_id = normalize_env_id_for_backend(task.env_id, backend)
        try:
            env = gym.make(env_id, **kwargs)
        except TypeError as exc:
            if "headless" not in kwargs or "headless" not in str(exc):
                raise
            kwargs.pop("headless", None)
            env = gym.make(env_id, **kwargs)
    else:
        gym = importlib.import_module("gymnasium")
        importlib.import_module("miniwob")
        env = gym.make(task.env_id)

    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        observation, info = reset_result
    else:
        observation, info = reset_result, {}
    return env, observation, info
