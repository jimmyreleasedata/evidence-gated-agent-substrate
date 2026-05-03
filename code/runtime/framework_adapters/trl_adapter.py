from __future__ import annotations

from .base import FrameworkAdapterDeclaration


TRL_ADAPTER_DECLARATION = FrameworkAdapterDeclaration(
    framework="trl",
    evidence_level="L1",
    native_rollout_object="trainer rollout batch / sampled generations",
    native_sample_store="Hugging Face Dataset or trainer-side replay buffer",
    native_version_tags="checkpoint tag and trainer state metadata",
    env_or_tool_interface="Benchmark Suite adapter boundary",
    verifier_or_reward_interface="reward callback backed by evaluator or verifier outputs",
    trace_or_observability_path="host logger plus dataset export trace hook",
    rollout_backend_options=("vllm", "sglang"),
    missing_for_versioned_trajectory=[
        "trace_id",
        "request_id",
        "provenance_refs",
        "consume_ts",
        "replay_class",
    ],
    required_patch_surface="hook-level thin bridge",
    current_status="lightweight secondary target",
    planned_workload_for_proof="miniwob",
)

TRL_ADAPTER_DECLARATION.validate()
