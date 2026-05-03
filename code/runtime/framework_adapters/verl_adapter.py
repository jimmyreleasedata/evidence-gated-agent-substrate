from __future__ import annotations

from .base import FrameworkAdapterDeclaration


VERL_ADAPTER_DECLARATION = FrameworkAdapterDeclaration(
    framework="verl",
    evidence_level="L1",
    native_rollout_object="HybridFlow rollout batch / actor output",
    native_sample_store="framework replay or sample buffer",
    native_version_tags="policy version and checkpoint publish tags",
    env_or_tool_interface="Benchmark Suite adapter boundary",
    verifier_or_reward_interface="reward or verifier callback with result metadata",
    trace_or_observability_path="host logger plus runtime trace hook",
    rollout_backend_options=("vllm", "sglang"),
    missing_for_versioned_trajectory=[
        "trace_id",
        "request_id",
        "provenance_refs",
        "consume_ts",
        "replay_class",
    ],
    required_patch_surface="hook-level thin bridge",
    current_status="preferred primary target",
    planned_workload_for_proof="webarena_verified",
)

VERL_ADAPTER_DECLARATION.validate()
