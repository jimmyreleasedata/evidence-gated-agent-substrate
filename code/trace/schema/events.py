"""Canonical event models for benchmark-suite traces."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


def new_trace_id() -> str:
    return uuid.uuid4().hex


def new_span_id() -> str:
    return uuid.uuid4().hex


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_now_ns() -> int:
    return time.monotonic_ns()


class EventType(str, Enum):
    RUN_START = "run_start"
    ENV_RESET = "env_reset"
    POLICY_DECISION = "policy_decision"
    LLM_REQUEST_SUBMIT = "llm_request_submit"
    LLM_REQUEST_START = "llm_request_start"
    LLM_REQUEST_END = "llm_request_end"
    TOOL_CALL_SUBMIT = "tool_call_submit"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    ENV_STEP_START = "env_step_start"
    ENV_STEP_END = "env_step_end"
    VERIFIER_START = "verifier_start"
    VERIFIER_END = "verifier_end"
    RETRY = "retry"
    FAILURE = "failure"
    REPLAY_CONSUME = "replay_consume"
    CHECKPOINT_SWITCH = "checkpoint_switch"
    RUN_END = "run_end"


class TelemetryMode(str, Enum):
    OFF = "off"
    BASIC = "basic"
    FULL = "full"


class ReplayClass(str, Enum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"


class TraceEvent(BaseModel):
    """Canonical event envelope used across all benchmark families."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    suite_version: str
    trace_schema_version: str
    run_id: str
    episode_id: str
    step_id: int
    trace_id: str
    span_id: str
    timestamp_wall: str
    timestamp_mono: int
    event_type: EventType
    task_family: str
    task_id: str

    parent_step_id: int | None = None
    parent_span_id: str | None = None
    environment_id: str | None = None
    snapshot_id: str | None = None
    model_id: str | None = None
    model_version: str | None = None
    policy_version: str | None = None
    verifier_id: str | None = None
    verifier_version: str | None = None
    tool_id: str | None = None
    return_code: int | None = None
    error_class: str | None = None
    retry_count: int = 0
    cpu_pct: float | None = None
    rss_mb: float | None = None
    gpu_mem_mb: float | None = None
    disk_read_mb: float | None = None
    disk_write_mb: float | None = None
    net_tx_mb: float | None = None
    net_rx_mb: float | None = None
    queue_wait_ms: float | None = None
    telemetry_mode: TelemetryMode = TelemetryMode.BASIC
    replay_class: ReplayClass | None = None
    replay_source: str | None = None
    replay_run_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id", "retry_count")
    @classmethod
    def _validate_non_negative_ints(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative")
        return value

    @field_validator("task_family", "task_id", "run_id", "episode_id", "trace_id", "span_id")
    @classmethod
    def _validate_non_empty_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be non-empty")
        return value

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
