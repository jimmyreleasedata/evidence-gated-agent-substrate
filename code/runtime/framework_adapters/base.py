"""Inert framework-adapter declarations copied from the ASPLOS bridge pattern."""

from __future__ import annotations

from dataclasses import dataclass, field


REQUIRED_HOOKS: tuple[str, ...] = (
    "rollout_emit_hook",
    "verifier_or_reward_hook",
    "replay_consume_hook",
    "learner_update_hook",
    "trace_provenance_hook",
)


@dataclass(slots=True)
class FrameworkAdapterDeclaration:
    framework: str
    evidence_level: str
    native_rollout_object: str
    native_sample_store: str
    native_version_tags: str
    env_or_tool_interface: str
    verifier_or_reward_interface: str
    trace_or_observability_path: str
    rollout_backend_options: tuple[str, ...] = ("vllm",)
    missing_for_versioned_trajectory: list[str] = field(default_factory=list)
    required_patch_surface: str = "hook-level thin bridge"
    current_status: str = ""
    planned_workload_for_proof: str = ""
    required_hooks: tuple[str, ...] = REQUIRED_HOOKS

    def validate(self) -> None:
        if self.evidence_level not in {"L0", "L1", "L2"}:
            raise ValueError(f"unsupported evidence level: {self.evidence_level}")
        if set(self.required_hooks) != set(REQUIRED_HOOKS):
            raise ValueError("required hook surface is incomplete")
        if not self.rollout_backend_options:
            raise ValueError("at least one rollout backend must be declared")
