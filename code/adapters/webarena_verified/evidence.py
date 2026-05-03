"""Evidence classification helpers for WebArena Verified."""

from __future__ import annotations

from typing import Any


REAL_BACKEND = "webarena_verified_official_r1"
LIVE_BACKEND = "webarena_verified_live_capture"
MOCK_BACKEND = "mock"
MOCK_ENVIRONMENT_ID = "webarena-verified-mock"
MOCK_MODEL_ID = "mock-web-policy"


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
    trace_hash = _value(summary, manifest, "trace_hash")
    evaluator_version = _value(summary, manifest, "evaluator_version") or _value(summary, manifest, "verifier_version")
    upstream_commit = _value(summary, manifest, "upstream_commit")

    if (
        backend == REAL_BACKEND
        or (
            replay_class == "R1"
            and trace_hash
            and evaluator_version
            and upstream_commit
            and environment_id != MOCK_ENVIRONMENT_ID
            and model_id != MOCK_MODEL_ID
        )
    ):
        return "real_upstream_replay"

    if (
        explicit == "real_upstream_live"
        or backend == LIVE_BACKEND
        or (
            evaluator_version
            and upstream_commit
            and _value(summary, manifest, "task_id")
            and _value(summary, manifest, "REDACTED_BROWSER_STATE_LABEL_hash")
            and _value(summary, manifest, "extra_headers_hash")
            and environment_id != MOCK_ENVIRONMENT_ID
            and model_id != MOCK_MODEL_ID
        )
    ):
        return "real_upstream_live"

    if backend == MOCK_BACKEND or environment_id == MOCK_ENVIRONMENT_ID or model_id == MOCK_MODEL_ID:
        return "mock_fixture"

    return "unsupported"


def infer_realism_level(summary: dict[str, Any] | None, manifest: dict[str, Any] | None) -> str:
    implementation_source = infer_implementation_source(summary, manifest)
    if implementation_source == "real_upstream_replay":
        return "real_upstream_replay"
    if implementation_source == "real_upstream_live":
        return "real_upstream_live"
    if implementation_source == "mock_fixture":
        return "mock_fixture"
    return "unsupported"


def paper_facing_allowed(summary: dict[str, Any] | None, manifest: dict[str, Any] | None) -> bool:
    return infer_implementation_source(summary, manifest) in {"real_upstream_replay", "real_upstream_live"}
