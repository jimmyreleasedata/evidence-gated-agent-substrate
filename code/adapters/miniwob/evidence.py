"""Evidence classification helpers for MiniWoB paper-facing rows."""

from __future__ import annotations

from typing import Any


MOCK_BACKEND = "mock"
MOCK_ENVIRONMENT_ID = "miniwob++-mock"
MOCK_MODEL_ID = "mock-policy"
REAL_BACKENDS = {"browsergym_miniwob", "farama_selenium"}


def _value(summary: dict[str, Any] | None, manifest: dict[str, Any] | None, key: str) -> Any:
    if summary and key in summary:
        return summary[key]
    if manifest and key in manifest:
        return manifest[key]
    return None


def infer_implementation_source(summary: dict[str, Any] | None, manifest: dict[str, Any] | None) -> str:
    explicit = _value(summary, manifest, "implementation_source")
    if explicit:
        return str(explicit)

    backend = _value(summary, manifest, "backend")
    environment_id = _value(summary, manifest, "environment_id")
    model_id = _value(summary, manifest, "model_id")
    replay_class = _value(summary, manifest, "replay_class")
    task_id = _value(summary, manifest, "task_id")
    upstream_package_version = _value(summary, manifest, "upstream_package_version")
    task_manifest_hash = _value(summary, manifest, "task_manifest_hash")

    if (
        backend in REAL_BACKENDS
        and replay_class == "R0"
        and task_id
        and upstream_package_version
        and task_manifest_hash
        and environment_id != MOCK_ENVIRONMENT_ID
        and model_id != MOCK_MODEL_ID
    ):
        return "real_upstream"

    if backend == MOCK_BACKEND or environment_id == MOCK_ENVIRONMENT_ID or model_id == MOCK_MODEL_ID:
        return "mock_fixture"

    return "unsupported"


def infer_realism_level(summary: dict[str, Any] | None, manifest: dict[str, Any] | None) -> str:
    implementation_source = infer_implementation_source(summary, manifest)
    if implementation_source == "real_upstream":
        return "real_upstream"
    if implementation_source == "mock_fixture":
        return "mock_fixture"
    return "unsupported"


def paper_facing_allowed(summary: dict[str, Any] | None, manifest: dict[str, Any] | None) -> bool:
    if infer_implementation_source(summary, manifest) != "real_upstream":
        return False
    browser_version = str(_value(summary, manifest, "browser_version") or "").strip().lower()
    driver_version = str(_value(summary, manifest, "driver_version") or "").strip().lower()
    if not browser_version or browser_version == "unknown":
        return False
    if not driver_version or driver_version == "unknown":
        return False
    return True
