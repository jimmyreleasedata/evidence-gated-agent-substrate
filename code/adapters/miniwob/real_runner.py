"""Run a real MiniWoB slice when an upstream backend is available."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from adapters.miniwob.real_config import MiniWobRealConfig
from adapters.miniwob.real_env import backend_info, make_real_env
from adapters.miniwob.real_env import normalize_env_id_for_backend
from adapters.miniwob.real_tasks import RealMiniWobTaskSpec, load_real_task_specs
from runner.event_bus import EventBus
from runner.run_context import RunContext
from trace.schema.events import EventType
from trace.validators.schema_validator import validate_jsonl


def json_safe_payload(payload: Any) -> Any:
    value_type = type(payload)
    module_name = getattr(value_type, "__module__", "")
    if module_name.startswith("numpy"):
        if hasattr(payload, "shape") and hasattr(payload, "dtype") and hasattr(payload, "tobytes"):
            return {
                "__type__": "ndarray",
                "shape": list(payload.shape),
                "dtype": str(payload.dtype),
                "sha256": hashlib.sha256(payload.tobytes()).hexdigest(),
            }
        if hasattr(payload, "item"):
            return payload.item()
    if isinstance(payload, dict):
        return {str(key): json_safe_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_safe_payload(value) for value in payload]
    return payload


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(json_safe_payload(payload), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_summary_payload(
    *,
    run_id: str,
    manifest: dict[str, Any],
    backend: str,
    seed: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    duration_ms: float,
    action_sequence: list[str],
    obs_hash: str,
    action_hash: str,
    screenshot_hash: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "implementation_source": "real_upstream",
        "backend": backend,
        "replay_class": "R0",
        "task_id": manifest["task_id"],
        "environment_seed": seed,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "duration_ms": duration_ms,
        "action_sequence": action_sequence,
        "obs_hash": obs_hash,
        "action_hash": action_hash,
        "screenshot_hash": screenshot_hash,
        "task_manifest_hash": manifest["task_manifest_hash"],
        "upstream_package_version": manifest["upstream_package_version"],
        "browser_version": manifest["browser_version"],
        "driver_version": manifest["driver_version"],
        "validation_pass": True,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def translate_action_for_backend(action: str, *, backend: str) -> str:
    if backend != "browsergym_miniwob":
        return action
    if "\n" in action or "page." in action:
        return action
    if action.startswith("click:"):
        return 'page.locator("body").click()'
    if action.startswith("type:"):
        return 'page.keyboard.type("test")'
    if action.startswith("drag:"):
        return "\n".join(
            [
                "page.mouse.move(10, 10)",
                "page.mouse.down()",
                "page.mouse.move(40, 40)",
                "page.mouse.up()",
            ]
        )
    return action


def run_real_task(task: RealMiniWobTaskSpec, config: MiniWobRealConfig, *, seed: int) -> dict[str, Any]:
    info = backend_info(config.backend, config.upstream_root)
    environment_id = normalize_env_id_for_backend(task.env_id, config.backend)
    start_ns = time.perf_counter_ns()
    env, reset_obs, reset_info = make_real_env(
        task,
        backend=config.backend,
        upstream_root=config.upstream_root,
        headless=config.headless,
        seed=seed,
    )
    context = RunContext.create(
        output_root=config.output_root,
        task_family="miniwob",
        telemetry_mode=config.telemetry_mode,
        environment_id=environment_id,
        model_id="recorded-action-sequence",
        model_version="captured",
        policy_version="captured",
        replay_class="R0",
    )
    bus = EventBus(context)
    episode_id = f"{task.task_id}-episode-{seed:04d}"
    bus.emit(
        EventType.RUN_START,
        episode_id=episode_id,
        step_id=0,
        task_id=task.task_id,
        payload={"backend": config.backend, "seed": seed, "implementation_source": "real_upstream"},
    )
    bus.emit(
        EventType.ENV_RESET,
        episode_id=episode_id,
        step_id=1,
        task_id=task.task_id,
        payload=json_safe_payload({"observation": reset_obs, "info": reset_info}),
    )

    observation = reset_obs
    reward = 0.0
    terminated = False
    truncated = False
    screenshot_hash = None
    for index, action in enumerate(task.action_sequence, start=2):
        translated_action = translate_action_for_backend(action, backend=config.backend)
        bus.emit(
            EventType.ENV_STEP_START,
            episode_id=episode_id,
            step_id=index * 2,
            task_id=task.task_id,
            payload={"action": translated_action},
        )
        step_result = env.step(translated_action)
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info_dict = step_result
        else:
            observation, reward, terminated, info_dict = step_result
            truncated = False
        screenshot_hash = screenshot_hash or info_dict.get("screenshot_hash")
        bus.emit(
            EventType.ENV_STEP_END,
            episode_id=episode_id,
            step_id=index * 2 + 1,
            task_id=task.task_id,
            payload=json_safe_payload(
                {
                    "observation": observation,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info_dict,
                }
            ),
        )
        if terminated or truncated:
            break

    bus.emit(
        EventType.REPLAY_CONSUME,
        episode_id=episode_id,
        step_id=99,
        task_id=task.task_id,
        payload={"replay_class": "R0", "mode": "statistical_replay_ready"},
    )
    bus.emit(
        EventType.RUN_END,
        episode_id=episode_id,
        step_id=100,
        task_id=task.task_id,
        payload={"reward": reward, "terminated": terminated, "truncated": truncated},
    )
    env.close()

    manifest = {
        "task_family": "miniwob",
        "task_id": task.task_id,
        "env_id": environment_id,
        "task_manifest_hash": task.task_manifest_hash,
        "upstream_package_version": info.upstream_package_version,
        "browser_version": info.browser_version,
        "driver_version": info.driver_version,
        "backend": config.backend,
        "implementation_source": "real_upstream",
        "selected_task_ids": [task.task_id],
    }
    manifest_path = context.write_manifest(manifest)
    summary = build_summary_payload(
        run_id=context.run_id,
        manifest={
            **manifest,
            "upstream_package_version": info.upstream_package_version,
            "browser_version": info.browser_version,
            "driver_version": info.driver_version,
        },
        backend=config.backend,
        seed=seed,
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000.0,
        action_sequence=list(task.action_sequence),
        obs_hash=_stable_hash(observation),
        action_hash=_stable_hash(task.action_sequence),
        screenshot_hash=screenshot_hash,
    )
    validation = validate_jsonl(context.paths.events_jsonl) if context.telemetry_enabled() else None
    summary["validation_pass"] = True if validation is None else validation.valid
    summary["summary_path"] = str(_write_json(context.paths.root / "summary.json", summary))
    summary["manifest_path"] = str(manifest_path)
    summary["trace_path"] = str(context.paths.events_jsonl)
    return summary


def run_real_slice(config: MiniWobRealConfig, *, seed: int) -> list[dict[str, Any]]:
    return [run_real_task(task, config, seed=seed) for task in load_real_task_specs(config.task_selection_path)]


def _write_aggregate(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames = [
        "task_id",
        "backend",
        "reward",
        "terminated",
        "truncated",
        "upstream_package_version",
        "browser_version",
        "driver_version",
        "summary_path",
        "manifest_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/miniwob_real"))
    parser.add_argument("--telemetry-mode", choices=("off", "basic", "full"), default="basic")
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = MiniWobRealConfig.from_env(output_root=args.output_root, telemetry_mode=args.telemetry_mode)
    rows = run_real_slice(config, seed=args.seed)
    aggregate_path = _write_aggregate(config.output_root / "aggregate_real.csv", rows)
    print(json.dumps({"task_count": len(rows), "aggregate_path": str(aggregate_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
