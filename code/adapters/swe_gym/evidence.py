"""Evidence classification helpers for SWE paper-facing rows."""

from __future__ import annotations

from typing import Any


SYNTHETIC_PREFIX = "mock__"
SYNTHETIC_COMMIT_PREFIX = "deadbeef"


def _value(summary: dict[str, Any] | None, manifest: dict[str, Any] | None, key: str) -> Any:
    if summary and key in summary:
        return summary[key]
    if manifest and key in manifest:
        return manifest[key]
    return None


def infer_implementation_source(
    summary: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> str:
    explicit = _value(summary, manifest, "implementation_source")
    if explicit:
        return str(explicit)

    backend = _value(summary, manifest, "backend")
    instance_id = str(_value(summary, manifest, "instance_id") or _value(summary, manifest, "task_instance_id") or "")
    repo_commit = str(_value(summary, manifest, "base_commit") or _value(summary, manifest, "repo_commit") or "")
    repo = _value(summary, manifest, "repo")

    if (
        backend == "swebench_official_or_compatible"
        and repo
        and repo_commit
        and not repo_commit.startswith(SYNTHETIC_COMMIT_PREFIX)
        and not instance_id.startswith(SYNTHETIC_PREFIX)
    ):
        return "real_upstream"

    if instance_id.startswith(SYNTHETIC_PREFIX) or repo_commit.startswith(SYNTHETIC_COMMIT_PREFIX):
        return "synthetic_executable"

    return "unsupported"


def infer_realism_level(
    summary: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> str:
    implementation_source = infer_implementation_source(summary, manifest)
    if implementation_source == "real_upstream":
        return "real_upstream"
    if implementation_source == "synthetic_executable":
        return "synthetic_executable"
    return "unsupported"


def paper_facing_allowed(
    summary: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> bool:
    return infer_implementation_source(summary, manifest) == "real_upstream"

