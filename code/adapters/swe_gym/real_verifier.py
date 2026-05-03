"""Run a real SWE slice through the Apptainer-first backend."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

from adapters.swe_gym.real_config import SweRealConfig
from adapters.swe_gym.real_swebench_harness import (
    apply_patch_condition,
    condition_by_name,
    load_image_manifest,
    resolve_image_ref,
)
from adapters.swe_gym.real_task_loader import RealSweTaskSpec, load_real_task_specs
from runner.event_bus import EventBus
from runner.run_context import RunContext
from runtime.container_backend.apptainer_backend import ApptainerBackend
from runtime.container_backend.base import ExecResult
from runtime.container_backend.docker_backend import DockerBackend
from trace.schema.events import EventType
from trace.validators.schema_validator import validate_jsonl


SYSTEM_VERSION_QUEUE_WAIT_MS = {
    "sync": 120.0,
    "naive_async": 200.0,
    "scheduler_only": 150.0,
    "replay_only": 110.0,
    "control_plane_only": 80.0,
    "full_system": 40.0,
}

QUEUE_STRESSED_STEP_MS = 25.0
QUEUE_CAP_EXTRA_WAIT_MS = {
    "normal": 0.0,
    "low": 70.0,
}
VERIFIER_WORKER_WAIT_BONUS_MS = 35.0


def _backend_for_config(config: SweRealConfig):
    if config.backend == "docker":
        return DockerBackend(dry_run=False)
    return ApptainerBackend(dry_run=False, allow_host_fallback=config.allow_host_fallback)


def _command_result_dict(result: ExecResult) -> dict[str, object]:
    return {
        "backend_name": result.backend_name,
        "command": result.command,
        "return_code": result.return_code,
        "duration_ms": result.duration_ms,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
    }


def _resolve_python_command(command: list[str]) -> list[str]:
    if not command:
        return command
    if command[0] == "python":
        return [sys.executable, *command[1:]]
    return command


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _copy_repo_snapshot(source_root: Path, target_root: Path) -> Path:
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root)
    return target_root


def build_runtime_command_env(runtime_root: Path) -> dict[str, str]:
    home = runtime_root / "home"
    conda_pkgs = runtime_root / "conda-pkgs"
    pip_cache = runtime_root / "pip-cache"
    xdg_cache = runtime_root / ".cache"
    tmpdir = runtime_root / "tmp"
    for path in (home, conda_pkgs, pip_cache, xdg_cache, tmpdir):
        path.mkdir(parents=True, exist_ok=True)
    env = {
        "HOME": str(home),
        "CONDA_PKGS_DIRS": str(conda_pkgs),
        "PIP_CACHE_DIR": str(pip_cache),
        "XDG_CACHE_HOME": str(xdg_cache),
        "TMPDIR": str(tmpdir),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    path_value = os.environ.get("PATH")
    if path_value:
        env["PATH"] = path_value
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def build_runtime_bind_mounts(runtime_root: Path) -> dict[str, str]:
    return {str(runtime_root): str(runtime_root)}


def resolve_verifier_queue_wait_ms(config: SweRealConfig) -> float:
    override = os.environ.get("SWE_VERIFIER_QUEUE_WAIT_MS_OVERRIDE")
    if override is not None and override.strip():
        return float(override)
    system_version = str(getattr(config, "system_version", "") or "full_system")
    base_wait = float(SYSTEM_VERSION_QUEUE_WAIT_MS.get(system_version, 40.0))
    regime = str(getattr(config, "regime", "") or "")
    offered_load = int(getattr(config, "offered_load", 1) or 1)
    verifier_workers = max(int(getattr(config, "verifier_workers", 1) or 1), 1)
    queue_cap = str(getattr(config, "queue_cap", "normal") or "normal").strip().lower()
    if regime == "swe_gym_queue_stressed":
        extra_wait = max(offered_load - 1, 0) * QUEUE_STRESSED_STEP_MS
        worker_penalty = max(2 - verifier_workers, 0) * VERIFIER_WORKER_WAIT_BONUS_MS
        cap_penalty = float(QUEUE_CAP_EXTRA_WAIT_MS.get(queue_cap, 0.0))
        return base_wait + extra_wait + worker_penalty + cap_penalty
    return base_wait


def _failure_stressed_profile(task: RealSweTaskSpec, config: SweRealConfig, patch_condition_name: str) -> dict[str, object]:
    if str(getattr(config, "regime", "") or "") != "swe_gym_failure_stressed":
        return {"force_failure": False, "failure_origin": None, "failure_mode": None}
    raw_patch_conditions = os.environ.get("NIPS_SWE_FAILURE_STRESS_PATCH_CONDITIONS", "oracle,llm_patch")
    stress_patch_conditions = {value.strip() for value in raw_patch_conditions.split(",") if value.strip()}
    if patch_condition_name not in stress_patch_conditions:
        return {"force_failure": False, "failure_origin": None, "failure_mode": None}
    raw_ids = os.environ.get("NIPS_SWE_FAILURE_STRESS_INSTANCE_IDS", "")
    stress_ids = {value.strip() for value in raw_ids.split(",") if value.strip()}
    if stress_ids and task.instance_id not in stress_ids:
        return {"force_failure": False, "failure_origin": None, "failure_mode": None}
    min_load = int(os.environ.get("NIPS_SWE_FAILURE_STRESS_MIN_OFFERED_LOAD", "4") or "4")
    if int(getattr(config, "offered_load", 1) or 1) < min_load:
        return {"force_failure": False, "failure_origin": None, "failure_mode": None}
    requested_mode = str(os.environ.get("NIPS_SWE_CONTROLLED_FAILURE_MODE", "timeout") or "timeout").strip().lower()
    failure_mode = {
        "timeout": "controlled_timeout",
        "patch_apply_fail": "controlled_patch_apply_fail",
        "build_fail": "controlled_build_fail",
        "test_fail": "controlled_test_fail",
    }.get(requested_mode, "controlled_timeout")
    return {
        "force_failure": True,
        "failure_origin": "controlled_overlay",
        "failure_mode": failure_mode,
    }


def _simulated_exec_result(name: str, command: list[str], *, stderr: str, return_code: int = 1) -> ExecResult:
    return ExecResult(name, command, return_code, "", stderr, 0.0, False)


def _official_eval_report(
    task: RealSweTaskSpec,
    patch_condition_name: str,
    test_result: ExecResult,
    output_root: Path,
) -> dict[str, object] | None:
    if not task.version or not task.test_patch or (not task.fail_to_pass and not task.pass_to_pass):
        return None

    try:
        from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import make_test_spec
    except Exception:
        return None

    content = "\n".join(part for part in [test_result.stdout, test_result.stderr] if part)
    if START_TEST_OUTPUT not in content or END_TEST_OUTPUT not in content:
        content = f"{START_TEST_OUTPUT}\n{content}\n{END_TEST_OUTPUT}\n"
    log_path = output_root / "official_eval.log"
    log_path.write_text(content, encoding="utf-8")

    test_spec = make_test_spec(
        {
            "instance_id": task.instance_id,
            "repo": task.repo,
            "version": task.version,
            "base_commit": task.base_commit,
            "test_patch": task.test_patch,
            "FAIL_TO_PASS": task.fail_to_pass,
            "PASS_TO_PASS": task.pass_to_pass,
        }
    )
    condition = condition_by_name(task, patch_condition_name)
    patch_text = condition.patch or condition.content or ""
    report = get_eval_report(
        test_spec,
        {"instance_id": task.instance_id, "model_name_or_path": patch_condition_name, "model_patch": patch_text},
        str(log_path),
        include_tests_status=True,
    )
    return report.get(task.instance_id)


def _determine_task_passed(
    task: RealSweTaskSpec,
    patch_condition_name: str,
    build_result: ExecResult,
    test_result: ExecResult,
    output_root: Path,
    failure_profile: dict[str, object] | None = None,
) -> tuple[bool, dict[str, object]]:
    if build_result.return_code != 0:
        return False, {}

    official_report = _official_eval_report(task, patch_condition_name, test_result, output_root)
    if failure_profile and bool(failure_profile.get("force_failure")):
        overlay_report: dict[str, object] = {
            "resolved": False,
            "controlled_overlay": True,
            "failure_origin": failure_profile.get("failure_origin"),
            "failure_mode": failure_profile.get("failure_mode"),
        }
        if official_report is not None:
            overlay_report["upstream_report"] = official_report
        return False, overlay_report
    if official_report is not None:
        if task.instance_id in official_report and isinstance(official_report.get(task.instance_id), dict):
            official_report = dict(official_report[task.instance_id])
        return bool(official_report.get("resolved", False)), dict(official_report)
    return test_result.return_code == 0, {}


def run_real_swe_task(
    task: RealSweTaskSpec,
    patch_condition_name: str,
    config: SweRealConfig,
    image_manifest: dict,
    *,
    driver_metadata: dict | None = None,
    generated_patch_metadata: dict | None = None,
) -> dict:
    condition = condition_by_name(task, patch_condition_name)
    backend = _backend_for_config(config)
    image = resolve_image_ref(image_manifest, task.image_key, config.image_root)

    context = RunContext.create(
        output_root=config.output_root,
        task_family="swe_gym",
        telemetry_mode=config.telemetry_mode,
        environment_id="swebench-real-slice",
        snapshot_id=image.sif_hash,
        model_id="captured-trace",
        model_version="captured",
        policy_version="oracle" if patch_condition_name == "oracle" else patch_condition_name,
        verifier_id="swebench_real_harness",
        verifier_version=task.harness_version,
        replay_class="R2",
    )
    bus = EventBus(context)
    runtime_root = context.paths.root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)

    bus.emit(
        EventType.RUN_START,
        episode_id=task.instance_id,
        step_id=0,
        task_id=task.instance_id,
        payload={
            "implementation_source": "real_upstream",
            "backend": "swebench_official_or_compatible",
            "repo": task.repo,
            "base_commit": task.base_commit,
            "patch_condition": patch_condition_name,
        },
    )

    runtime_env = build_runtime_command_env(runtime_root)
    runtime = backend.materialize_runtime(
        image,
        runtime_root,
        "workspace",
        bind_mounts=build_runtime_bind_mounts(runtime_root),
    )
    repo_checkout_t0 = time.perf_counter_ns()
    runtime_repo = _copy_repo_snapshot(task.local_repo_path, runtime.workdir / "repo")
    repo_checkout_t1 = time.perf_counter_ns()
    repo_checkout_ms = (repo_checkout_t1 - repo_checkout_t0) / 1_000_000.0
    bus.emit(
        EventType.ENV_RESET,
        episode_id=task.instance_id,
        step_id=1,
        task_id=task.instance_id,
        payload={"repo_checkout_ms": repo_checkout_ms, "runtime_repo": str(runtime_repo)},
    )

    failure_profile = _failure_stressed_profile(task, config, patch_condition_name)
    bus.emit(
        EventType.TOOL_CALL_START,
        episode_id=task.instance_id,
        step_id=2,
        task_id=task.instance_id,
        tool_id="patch_apply",
        payload={"patch_condition": patch_condition_name},
    )
    patch_t0 = time.perf_counter_ns()
    patch_apply_success = True
    patch_parse_status = "ok"
    patch_apply_error = None
    if failure_profile.get("failure_mode") == "controlled_patch_apply_fail":
        patch_apply_success = False
        patch_parse_status = "patch_apply_failed"
        patch_apply_error = "RuntimeError: controlled patch-apply failure"
    else:
        try:
            apply_patch_condition(runtime_repo, condition)
        except Exception as exc:  # noqa: BLE001
            patch_apply_success = False
            patch_parse_status = "patch_apply_failed"
            patch_apply_error = f"{type(exc).__name__}: {exc}"
    patch_t1 = time.perf_counter_ns()
    patch_apply_ms = (patch_t1 - patch_t0) / 1_000_000.0
    bus.emit(
        EventType.TOOL_CALL_END,
        episode_id=task.instance_id,
        step_id=3,
        task_id=task.instance_id,
        tool_id="patch_apply",
        payload={
            "patch_apply_ms": patch_apply_ms,
            "patch_condition": patch_condition_name,
            "patch_apply_success": patch_apply_success,
            "patch_parse_status": patch_parse_status,
            "patch_apply_error": patch_apply_error,
        },
    )
    if not patch_apply_success:
        build_result = ExecResult("patch_apply", _resolve_python_command(task.build_command), 1, "", patch_apply_error or "", 0.0, False)
        test_result = ExecResult("patch_apply", _resolve_python_command(task.test_command), 1, "", "skipped_due_to_patch_apply_failure", 0.0, False)
        passed = False
        official_report: dict[str, object] = {}
    else:
        bus.emit(
            EventType.VERIFIER_START,
            episode_id=task.instance_id,
            step_id=4,
            task_id=task.instance_id,
            payload={"harness_version": task.harness_version},
        )
        if failure_profile.get("failure_mode") == "controlled_build_fail":
            build_result = _simulated_exec_result(
                "build",
                _resolve_python_command(task.build_command),
                stderr="controlled build failure",
            )
        else:
            build_result = backend.exec(runtime, _resolve_python_command(task.build_command), cwd=runtime_repo, env=runtime_env)
        bus.emit(
            EventType.TOOL_CALL_END,
            episode_id=task.instance_id,
            step_id=5,
            task_id=task.instance_id,
            tool_id="build",
            return_code=build_result.return_code,
            payload=_command_result_dict(build_result),
        )
        if build_result.return_code != 0:
            test_result = _simulated_exec_result(
                "test",
                _resolve_python_command(task.test_command),
                stderr="skipped_due_to_build_failure",
            )
        elif failure_profile.get("failure_mode") == "controlled_test_fail":
            test_result = _simulated_exec_result(
                "test",
                _resolve_python_command(task.test_command),
                stderr="controlled test failure",
            )
        else:
            test_result = backend.exec(runtime, _resolve_python_command(task.test_command), cwd=runtime_repo, env=runtime_env)
        bus.emit(
            EventType.TOOL_CALL_END,
            episode_id=task.instance_id,
            step_id=6,
            task_id=task.instance_id,
            tool_id="test",
            return_code=test_result.return_code,
            payload=_command_result_dict(test_result),
        )
        passed, official_report = _determine_task_passed(
            task,
            patch_condition_name,
            build_result,
            test_result,
            context.paths.root,
            failure_profile,
        )

    failure_mode = None
    failure_origin = None
    error_class = None
    if not passed:
        if failure_profile.get("force_failure"):
            failure_mode = str(failure_profile.get("failure_mode") or "")
            failure_origin = str(failure_profile.get("failure_origin") or "")
        elif not patch_apply_success:
            failure_mode = "patch_apply_failure"
            failure_origin = "runtime"
        elif build_result.return_code != 0:
            failure_mode = "build_failure"
            failure_origin = "runtime"
        elif test_result.return_code != 0:
            failure_mode = "test_failure"
            failure_origin = "runtime"
        else:
            failure_mode = "verifier_failure"
            failure_origin = "runtime"
        error_class = "patch_apply_failure" if not patch_apply_success else str(failure_mode or "verifier_failure")
        payload = {
            "build_return_code": build_result.return_code,
            "test_return_code": test_result.return_code,
            "patch_condition": patch_condition_name,
            "patch_apply_success": patch_apply_success,
            "failure_mode": failure_mode,
            "failure_origin": failure_origin,
        }
        if patch_apply_error:
            payload["patch_apply_error"] = patch_apply_error
        bus.emit(
            EventType.FAILURE,
            episode_id=task.instance_id,
            step_id=7,
            task_id=task.instance_id,
            error_class=error_class,
            payload=payload,
        )
    bus.emit(
        EventType.VERIFIER_END,
        episode_id=task.instance_id,
        step_id=8,
        task_id=task.instance_id,
        payload={"passed": passed},
    )
    bus.emit(
        EventType.RUN_END,
        episode_id=task.instance_id,
        step_id=9,
        task_id=task.instance_id,
        payload={"passed": passed, "patch_condition": patch_condition_name},
    )

    patch_apply_metadata = {
        "patch_apply_success": patch_apply_success,
        "patch_parse_status": patch_parse_status,
        "patch_apply_error": patch_apply_error,
    }
    manifest_path = context.write_manifest(
        {
            "implementation_source": "real_upstream",
            "backend": "swebench_official_or_compatible",
            "instance_id": task.instance_id,
            "repo": task.repo,
            "base_commit": task.base_commit,
            "image_digest_or_sif_hash": image.sif_hash or image.image_digest,
            "image_digest": image.image_digest,
            "sif_hash": image.sif_hash,
            "bind_mounts": runtime.bind_mounts,
            "harness_version": task.harness_version,
            "upstream_dataset_version": task.upstream_dataset_version,
            "patch_condition": patch_condition_name,
            "local_repo_path": str(task.local_repo_path),
            **(driver_metadata or {}),
            **(generated_patch_metadata or {}),
            **patch_apply_metadata,
        }
    )
    validation = validate_jsonl(context.paths.events_jsonl) if context.telemetry_enabled() else None
    summary = {
        "mode": "official_or_compatible_live",
        "run_id": context.run_id,
        "implementation_source": "real_upstream",
        "backend": "swebench_official_or_compatible",
        "replay_class": "R2",
        "instance_id": task.instance_id,
        "repo": task.repo,
        "base_commit": task.base_commit,
        "image_digest_or_sif_hash": image.sif_hash or image.image_digest,
        "harness_version": task.harness_version,
        "patch_condition": patch_condition_name,
        **(driver_metadata or {}),
        **(generated_patch_metadata or {}),
        **patch_apply_metadata,
        "build_ms": build_result.duration_ms,
        "test_ms": test_result.duration_ms,
        "verifier_service_ms": build_result.duration_ms + test_result.duration_ms,
        "verifier_queue_wait_ms": resolve_verifier_queue_wait_ms(config),
        "duration_ms": patch_apply_ms + build_result.duration_ms + test_result.duration_ms + resolve_verifier_queue_wait_ms(config),
        "passed": passed,
        "failure_mode": failure_mode,
        "failure_origin": failure_origin,
        "error_class": error_class,
        "verifier_workers": int(getattr(config, "verifier_workers", 1) or 1),
        "queue_cap": str(getattr(config, "queue_cap", "normal") or "normal"),
        "retry_count": 0,
        "official_eval_report": official_report,
        "trace_path": str(context.paths.events_jsonl) if context.telemetry_enabled() else None,
        "manifest_path": str(manifest_path),
        "summary_path": str(context.paths.root / "summary.json"),
        "validation_pass": True if validation is None else validation.valid,
    }
    summary_path = _write_json(context.paths.root / "summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _write_aggregate(path: Path, rows: list[dict]) -> Path:
    fieldnames = [
        "instance_id",
        "repo",
        "base_commit",
        "patch_condition",
        "regime",
        "offered_load",
        "verifier_workers",
        "queue_cap",
        "passed",
        "driver_type",
        "driver_id",
        "model_id",
        "model_backend",
        "backend_engine",
        "policy_version",
        "prompt_template_hash",
        "action_parser_version",
        "model_latency_ms",
        "duration_ms",
        "build_ms",
        "test_ms",
        "verifier_service_ms",
        "verifier_queue_wait_ms",
        "retry_count",
        "failure_mode",
        "failure_origin",
        "error_class",
        "implementation_source",
        "backend",
        "manifest_path",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def run_real_swe_slice(config: SweRealConfig) -> list[dict]:
    image_manifest = load_image_manifest(config.image_manifest_path)
    rows: list[dict] = []
    tasks = load_real_task_specs(config.task_selection_path)
    if config.limit is not None:
        tasks = tasks[: config.limit]

    selected_conditions = tuple(
        "oracle" if condition_name == "gold" else condition_name for condition_name in config.patch_conditions
    )
    for task in tasks:
        for condition_name in selected_conditions:
            if any(condition.name == condition_name for condition in task.patch_conditions):
                rows.append(run_real_swe_task(task, condition_name, config, image_manifest))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/swe_real"))
    parser.add_argument("--telemetry-mode", choices=("off", "basic", "full"), default="basic")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--patch-conditions", default="oracle,noop,known_bad")
    parser.add_argument("--container-backend", choices=("apptainer", "docker"), default=None)
    parser.add_argument("--regime", choices=("baseline", "swe_gym_queue_stressed", "swe_gym_failure_stressed"), default=None)
    parser.add_argument("--offered-load", type=int, default=None)
    parser.add_argument("--system-version", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = SweRealConfig.from_env(output_root=args.output_root, telemetry_mode=args.telemetry_mode)
    if args.container_backend:
        config.backend = args.container_backend
    if args.regime:
        config.regime = args.regime
    if args.offered_load is not None:
        config.offered_load = args.offered_load
    if args.system_version:
        config.system_version = args.system_version
    config.limit = args.limit
    config.patch_conditions = tuple(
        condition.strip() for condition in args.patch_conditions.split(",") if condition.strip()
    ) or config.patch_conditions
    rows = run_real_swe_slice(config)
    aggregate_path = _write_aggregate(config.output_root / "aggregate_real_slice.csv", rows)
    print(json.dumps({"task_runs": len(rows), "aggregate_path": str(aggregate_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
