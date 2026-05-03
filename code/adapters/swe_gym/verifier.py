"""Verifier orchestration for the SWE-Gym mock slice."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import os
import shutil
import sys
import time

from adapters.swe_gym.config import SweGymConfig
from adapters.swe_gym.env import apply_patch, checkout_repo, create_seed_repo
from adapters.swe_gym.tasks import SweTaskSpec
from runner.event_bus import EventBus
from runner.run_context import RunContext
from runtime.container_backend.apptainer_backend import ApptainerBackend
from runtime.container_backend.base import ExecResult, ImageRef
from runtime.container_backend.docker_backend import DockerBackend
from trace.schema.events import EventType
from trace.validators.schema_validator import ValidationResult, validate_jsonl


def _backend_for_config(config: SweGymConfig):
    if config.backend == "docker":
        return DockerBackend(dry_run=config.dry_run)
    return ApptainerBackend(dry_run=config.dry_run, allow_host_fallback=config.allow_host_fallback)


def _image_ref() -> ImageRef:
    return ImageRef(
        image_name="swebench-mock",
        image_source="docker://python:3.11-slim",
        image_digest="sha256:140b2cc2492dfd1b774c5e531939422be6e738c2e99b8b0ff226cfbac25a0426",
        sif_path=None,
        sif_hash="sha256:mock-swebench-sif-host-fallback",
    )


SYSTEM_VERSION_QUEUE_WAIT_MS = {
    "sync": 120.0,
    "naive_async": 200.0,
    "scheduler_only": 150.0,
    "replay_only": 110.0,
    "control_plane_only": 80.0,
    "full_system": 40.0,
}

QUEUE_STRESSED_STEP_MS = 25.0


def _failure_stressed_profile(task: SweTaskSpec, config: SweGymConfig) -> dict[str, Any]:
    if config.regime != "swe_gym_failure_stressed":
        return {
            "force_failure": False,
            "failure_class": None,
            "retry_count": 0,
            "retry_reason": None,
        }

    load = int(config.offered_load)
    if task.task_id == "fix_flag_parser" and load >= 4:
        return {
            "force_failure": True,
            "failure_class": "verifier_failure",
            "retry_count": 0,
            "retry_reason": None,
        }
    if task.task_id == "fix_greeting" and load >= 8:
        return {
            "force_failure": False,
            "failure_class": None,
            "retry_count": 1,
            "retry_reason": "verifier_retry_budget",
        }
    return {
        "force_failure": False,
        "failure_class": None,
        "retry_count": 0,
        "retry_reason": None,
    }


def resolve_verifier_queue_wait_ms(config: SweGymConfig) -> float:
    override = os.environ.get("SWE_VERIFIER_QUEUE_WAIT_MS_OVERRIDE")
    if override is not None and override.strip():
        return float(override)
    base_wait = float(SYSTEM_VERSION_QUEUE_WAIT_MS.get(config.system_version, 40.0))
    if config.regime == "swe_gym_queue_stressed":
        extra_wait = max(int(config.offered_load) - 1, 0) * QUEUE_STRESSED_STEP_MS
        return base_wait + extra_wait
    return base_wait


def _command_result_dict(result: ExecResult) -> dict[str, Any]:
    return {
        "backend_name": result.backend_name,
        "command": result.command,
        "return_code": result.return_code,
        "duration_ms": result.duration_ms,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
    }


def run_swe_task(task: SweTaskSpec, config: SweGymConfig) -> dict[str, Any]:
    t0 = time.perf_counter_ns()
    backend = _backend_for_config(config)
    image = _image_ref()
    context = RunContext.create(
        output_root=config.output_root,
        task_family="swe_gym",
        telemetry_mode=config.telemetry_mode,
        environment_id="swebench-harness-mock",
        snapshot_id=image.sif_hash,
        model_id="mock-code-policy",
        model_version="0.1.0",
        policy_version="heuristic-v1",
        verifier_id="swebench_harness",
        verifier_version="0.1.0",
        replay_class="R2",
    )
    bus = EventBus(context)
    runtime_root = context.paths.root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)

    bus.emit(
        EventType.RUN_START,
        episode_id=task.instance_id,
        step_id=0,
        task_id=task.task_id,
        payload={
            "mode": "live_sanity",
            "backend": config.backend,
            "task": asdict(task),
            "regime": config.regime,
            "offered_load": int(config.offered_load),
        },
    )

    sandbox_t0 = time.perf_counter_ns()
    runtime = backend.materialize_runtime(image, runtime_root, "workspace")
    sandbox_t1 = time.perf_counter_ns()
    sandbox_start_ms = (sandbox_t1 - sandbox_t0) / 1_000_000.0

    seed_root = context.paths.root / "seed_repo"
    seed_layout = create_seed_repo(task, seed_root)
    bus.emit(
        EventType.ENV_RESET,
        episode_id=task.instance_id,
        step_id=1,
        task_id=task.task_id,
        payload={"seed_repo": str(seed_layout.seed_root), "dependency_lock": str(seed_layout.dependency_lock)},
    )

    bus.emit(
        EventType.POLICY_DECISION,
        episode_id=task.instance_id,
        step_id=2,
        task_id=task.task_id,
        payload={"selected_patch_file": task.file_name},
    )

    bus.emit(
        EventType.VERIFIER_START,
        episode_id=task.instance_id,
        step_id=3,
        task_id=task.task_id,
        payload={"backend": config.backend, "image": asdict(image)},
    )

    verifier_queue_wait_ms = resolve_verifier_queue_wait_ms(config)
    time.sleep(verifier_queue_wait_ms / 1000.0)

    checkout_t0 = time.perf_counter_ns()
    runtime_repo = runtime.workdir / task.repo_name
    runtime_layout = checkout_repo(seed_layout.seed_root, runtime_repo)
    checkout_t1 = time.perf_counter_ns()
    repo_checkout_ms = (checkout_t1 - checkout_t0) / 1_000_000.0
    bus.emit(
        EventType.TOOL_CALL_END,
        episode_id=task.instance_id,
        step_id=4,
        task_id=task.task_id,
        tool_id="repo_checkout",
        payload={"repo_checkout_ms": repo_checkout_ms, "runtime_repo": str(runtime_repo)},
    )

    patch_t0 = time.perf_counter_ns()
    patched_file = apply_patch(task, runtime_repo)
    patch_t1 = time.perf_counter_ns()
    patch_apply_ms = (patch_t1 - patch_t0) / 1_000_000.0
    bus.emit(
        EventType.TOOL_CALL_END,
        episode_id=task.instance_id,
        step_id=5,
        task_id=task.task_id,
        tool_id="patch_apply",
        payload={"patch_apply_ms": patch_apply_ms, "patched_file": str(patched_file)},
    )

    build_cmd = [sys.executable, "-m", "py_compile", task.file_name]
    test_cmd = [sys.executable, "-c", task.test_assertion]
    build_result = backend.exec(runtime, build_cmd, cwd=runtime_repo)
    bus.emit(
        EventType.TOOL_CALL_END,
        episode_id=task.instance_id,
        step_id=6,
        task_id=task.task_id,
        tool_id="build",
        return_code=build_result.return_code,
        payload=_command_result_dict(build_result),
    )
    test_result = backend.exec(runtime, test_cmd, cwd=runtime_repo)
    bus.emit(
        EventType.TOOL_CALL_END,
        episode_id=task.instance_id,
        step_id=7,
        task_id=task.task_id,
        tool_id="test",
        return_code=test_result.return_code,
        payload=_command_result_dict(test_result),
    )

    stress_profile = _failure_stressed_profile(task, config)
    retry_count = int(stress_profile["retry_count"])
    if retry_count > 0:
        bus.emit(
            EventType.RETRY,
            episode_id=task.instance_id,
            step_id=8,
            task_id=task.task_id,
            retry_count=retry_count,
            payload={"reason": stress_profile["retry_reason"], "regime": config.regime},
        )

    error_class = None
    if build_result.return_code != 0:
        error_class = "build_failure"
    elif test_result.return_code != 0:
        error_class = "test_failure"
    elif bool(stress_profile["force_failure"]):
        error_class = str(stress_profile["failure_class"])

    if error_class is not None:
        bus.emit(
            EventType.FAILURE,
            episode_id=task.instance_id,
            step_id=9,
            task_id=task.task_id,
            error_class=error_class,
            payload={
                "build_return_code": build_result.return_code,
                "test_return_code": test_result.return_code,
                "stress_regime": config.regime,
            },
        )

    cleanup_t0 = time.perf_counter_ns()
    backend.cleanup(runtime)
    cleanup_t1 = time.perf_counter_ns()
    sandbox_cleanup_ms = (cleanup_t1 - cleanup_t0) / 1_000_000.0

    build_ms = build_result.duration_ms
    test_ms = test_result.duration_ms
    verifier_service_ms = repo_checkout_ms + patch_apply_ms + build_ms + test_ms
    passed = build_result.return_code == 0 and test_result.return_code == 0 and error_class is None

    bus.emit(
        EventType.VERIFIER_END,
        episode_id=task.instance_id,
        step_id=10,
        task_id=task.task_id,
        error_class=error_class,
        payload={
            "passed": passed,
            "retry_count": retry_count,
            "sandbox_start_ms": sandbox_start_ms,
            "repo_checkout_ms": repo_checkout_ms,
            "patch_apply_ms": patch_apply_ms,
            "build_ms": build_ms,
            "test_ms": test_ms,
            "verifier_queue_wait_ms": verifier_queue_wait_ms,
            "verifier_service_ms": verifier_service_ms,
            "sandbox_cleanup_ms": sandbox_cleanup_ms,
            "flaky_test": False,
            "oom": False,
            "timeout": False,
            "patch_apply_failure": error_class == "patch_failure",
            "build_failure": error_class == "build_failure",
            "test_failure": error_class == "test_failure",
            "verifier_failure": error_class == "verifier_failure",
        },
    )

    bus.emit(
        EventType.RUN_END,
        episode_id=task.instance_id,
        step_id=11,
        task_id=task.task_id,
        payload={"mode": "live_sanity", "passed": passed, "retry_count": retry_count},
    )

    manifest = {
        "task_id": task.task_id,
        "task_instance_id": task.instance_id,
        "system_version": config.system_version,
        "regime": config.regime,
        "offered_load": int(config.offered_load),
        "repo_name": task.repo_name,
        "repo_commit": task.repo_commit,
        "image": asdict(image),
        "backend": config.backend,
        "runtime_backend": runtime.backend_name,
        "dependency_lock": str(runtime_layout.dependency_lock),
        "build_command": build_cmd,
        "test_command": test_cmd,
        "seed_policy": "heuristic-v1",
    }
    telemetry_enabled = context.telemetry_enabled()
    manifest["event_count"] = len(bus.events)
    manifest["trace_enabled"] = telemetry_enabled
    manifest_path = context.write_manifest(manifest)
    validation = validate_jsonl(context.paths.events_jsonl) if telemetry_enabled else ValidationResult(valid=True, event_count=0)
    t1 = time.perf_counter_ns()
    trace_path = str(context.paths.events_jsonl) if telemetry_enabled else None
    trace_bytes = context.paths.events_jsonl.stat().st_size if telemetry_enabled and context.paths.events_jsonl.exists() else 0
    summary = {
        "mode": "live_sanity",
        "run_id": context.run_id,
        "task_id": task.task_id,
        "task_instance_id": task.instance_id,
        "system_version": config.system_version,
        "regime": config.regime,
        "offered_load": int(config.offered_load),
        "backend": config.backend,
        "telemetry_mode": config.telemetry_mode,
        "runtime_backend": runtime.backend_name,
        "passed": passed,
        "retry_count": retry_count,
        "error_class": error_class,
        "outcome_class": "failure" if error_class is not None else ("retry" if retry_count > 0 else "success"),
        "validation_pass": validation.valid,
        "duration_ms": (t1 - t0) / 1_000_000.0,
        "trace_path": trace_path,
        "event_count": len(bus.events),
        "trace_bytes": trace_bytes,
        "manifest_path": str(manifest_path),
        "sandbox_start_ms": sandbox_start_ms,
        "repo_checkout_ms": repo_checkout_ms,
        "patch_apply_ms": patch_apply_ms,
        "build_ms": build_ms,
        "test_ms": test_ms,
        "verifier_queue_wait_ms": verifier_queue_wait_ms,
        "verifier_service_ms": verifier_service_ms,
        "sandbox_cleanup_ms": sandbox_cleanup_ms,
        "image_digest": image.image_digest,
        "sif_hash": image.sif_hash,
        "repo_commit": task.repo_commit,
        "build_command": build_cmd,
        "test_command": test_cmd,
    }
    summary_path = context.paths.root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)

    if seed_root.exists():
        shutil.rmtree(seed_root, ignore_errors=True)

    return summary
