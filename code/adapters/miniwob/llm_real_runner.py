"""Generic OpenAI-compatible MiniWoB runner on top of the real browser-backed backend."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

from adapters.miniwob.real_config import MiniWobRealConfig
from adapters.miniwob.real_env import backend_info, make_real_env, normalize_env_id_for_backend
from adapters.miniwob.real_runner import _stable_hash, json_safe_payload, translate_action_for_backend
from adapters.miniwob.real_tasks import RealMiniWobTaskSpec, load_real_task_specs
from adapters.webarena_verified.llm_live_runner import _load_registry_model
from runner.event_bus import EventBus
from runner.run_context import RunContext
from runtime.agent_driver.agent_driver.llm_driver import LLMDriver
from runtime.agent_driver.agent_driver.llm_driver import LLMDriverConfig
from runtime.model_backend_client import OpenAICompatibleClient
from runtime.rl_closure.backends import served_model_alias
from trace.schema.events import EventType
from trace.validators.schema_validator import validate_jsonl


@dataclass(frozen=True, slots=True)
class ParsedMiniWobAction:
    canonical_action: str
    parse_status: str
    invalid_action: bool


class OpenAIMiniWobDriver(LLMDriver):
    def __init__(self, config: LLMDriverConfig, *, generate_fn, base_url: str = "", served_model_name: str = "") -> None:
        super().__init__(config, generate_fn=generate_fn)
        self.base_url = base_url
        self.served_model_name = served_model_name

    def _parse_action(self, action_text: str) -> tuple[dict[str, Any], str, bool]:
        parsed = parse_llm_miniwob_action(action_text)
        return (
            {"canonical_action": parsed.canonical_action},
            parsed.parse_status,
            parsed.invalid_action,
        )


@dataclass(frozen=True, slots=True)
class MiniWobWorkerSpec:
    task: RealMiniWobTaskSpec
    output_root: Path
    backend: str
    upstream_root: Path
    headless: bool
    telemetry_mode: str
    registry_path: Path
    model_id: str
    host: str
    port: int
    budget: int
    seed: int


def parse_llm_miniwob_action(action_text: str) -> ParsedMiniWobAction:
    try:
        payload = json.loads(action_text)
    except json.JSONDecodeError:
        return ParsedMiniWobAction(canonical_action="noop", parse_status="invalid_json", invalid_action=True)
    if not isinstance(payload, dict):
        return ParsedMiniWobAction(canonical_action="noop", parse_status="invalid_shape", invalid_action=True)
    action = str(payload.get("action") or "").strip().lower()
    if action == "click":
        return ParsedMiniWobAction(canonical_action="click:body", parse_status="ok", invalid_action=False)
    if action == "type":
        text = str(payload.get("text") or "").strip()
        if not text:
            return ParsedMiniWobAction(canonical_action="noop", parse_status="missing_text", invalid_action=True)
        return ParsedMiniWobAction(canonical_action=f"type:{text}", parse_status="ok", invalid_action=False)
    if action == "drag":
        return ParsedMiniWobAction(canonical_action="drag:body", parse_status="ok", invalid_action=False)
    return ParsedMiniWobAction(canonical_action="noop", parse_status="unsupported_action", invalid_action=True)


def _build_observation(
    task: RealMiniWobTaskSpec,
    *,
    observation: Any,
    info: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    obs_payload = json_safe_payload(observation)
    if isinstance(obs_payload, dict):
        compact_observation = {
            "goal": obs_payload.get("goal"),
            "url": obs_payload.get("url"),
            "open_pages_titles": obs_payload.get("open_pages_titles"),
            "open_pages_urls": obs_payload.get("open_pages_urls"),
            "active_page_index": obs_payload.get("active_page_index"),
            "last_action": obs_payload.get("last_action"),
            "last_action_error": obs_payload.get("last_action_error"),
            "elapsed_time": obs_payload.get("elapsed_time"),
            "chat_messages": obs_payload.get("chat_messages"),
            "focused_element_bid": obs_payload.get("focused_element_bid"),
            "has_visual_state": "screenshot" in obs_payload,
            "has_dom_state": "dom_object" in obs_payload,
            "has_accessibility_tree": "axtree_object" in obs_payload,
            "extra_element_property_count": len(obs_payload.get("extra_element_properties") or {}),
        }
    else:
        compact_observation = obs_payload
    return {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "env_id": task.env_id,
        "backend": backend,
        "observation": compact_observation,
        "info": json_safe_payload(info),
    }


def _task_aliases(task: RealMiniWobTaskSpec) -> set[str]:
    aliases = {task.task_id, task.env_id}
    env_id = task.env_id
    if env_id.startswith("browsergym/"):
        aliases.add(env_id.removeprefix("browsergym/"))
    if "." in env_id:
        aliases.add(env_id.replace(".", "/"))
    if env_id.endswith("-v1"):
        aliases.add(env_id[:-3])
    task_id = task.task_id
    if "/" in task_id:
        aliases.add(task_id.split("/", 1)[1])
    return {alias.strip() for alias in aliases if alias and alias.strip()}


def _driver_action(
    driver: LLMDriver,
    task: RealMiniWobTaskSpec,
    *,
    observation: Any,
    info: dict[str, Any],
    backend: str,
) -> tuple[dict[str, Any], ParsedMiniWobAction]:
    obs_payload = _build_observation(task, observation=observation, info=info, backend=backend)
    task_context = {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "env_id": task.env_id,
        "backend": backend,
    }
    action = driver.act(obs_payload, task_context, budget=driver.metadata.budget)
    parsed = parse_llm_miniwob_action(str(action["action_text"]))
    return action, parsed


def make_openai_miniwob_driver(
    model: dict[str, Any],
    *,
    backend: str,
    host: str,
    port: int,
    budget: int,
    seed: int,
) -> OpenAIMiniWobDriver:
    model_id = str(model["model_id"])
    model_path = str(model.get("snapshot_path") or model.get("model_path") or model.get("model_path_or_hf_id") or model_id)
    served_model_name = str(model.get("served_model_name") or served_model_alias(model_id))
    client = OpenAICompatibleClient(api_base_url=f"http://{host}:{port}")

    def _generate(prompt_text: str) -> str:
        payload = client.generate(
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
            max_tokens=96,
        )
        return str(payload["text"])

    return OpenAIMiniWobDriver(
        LLMDriverConfig(
            driver_id=f"llm-driver/miniwob/{model_id}",
            driver_version="0.1.0",
            model_backend=backend,
            backend_engine=backend,
            backend_version="",
            model_id=model_id,
            model_family=str(model.get("model_family") or "unknown"),
            model_revision=str(model.get("model_revision") or model.get("checkpoint_id") or model.get("model_path_hash") or ""),
            model_path_or_hf_id=model_path,
            tokenizer_path=str(model.get("tokenizer_path") or model_path),
            policy_version="llm_miniwob_react_v1",
            prompt_template=(
                "You are a MiniWoB browser-task agent.\n"
                "Emit strict JSON only.\n"
                "Supported actions:\n"
                '1. {"action":"click"}\n'
                '2. {"action":"type","text":"..."}\n'
                '3. {"action":"drag"}\n'
                "Do not include markdown fences or extra explanation.\n"
                "Observation:\n{obs}\n"
                "Task context:\n{task_context}\n"
            ),
            action_parser_version="miniwob-action-parser-v1",
            budget=budget,
            seed=seed,
        ),
        generate_fn=_generate,
        base_url=f"http://{host}:{port}",
        served_model_name=served_model_name,
    )


def run_llm_task(
    task: RealMiniWobTaskSpec,
    *,
    output_root: Path,
    backend: str,
    upstream_package_version: str,
    browser_version: str,
    driver_version: str,
    telemetry_mode: str,
    seed: int,
    budget: int,
    driver: LLMDriver,
    env: Any,
    initial_observation: Any,
    initial_info: dict[str, Any],
) -> dict[str, Any]:
    run_context = RunContext.create(
        output_root=output_root,
        task_family="miniwob",
        telemetry_mode=telemetry_mode,
        environment_id=normalize_env_id_for_backend(task.env_id, backend),
        model_id=driver.metadata.model_id,
        model_version=driver.metadata.model_version,
        policy_version=driver.metadata.policy_version,
        replay_class="R0",
    )
    bus = EventBus(run_context)
    episode_id = f"{task.task_id}-episode-{seed:04d}"
    bus.emit(
        EventType.RUN_START,
        episode_id=episode_id,
        step_id=0,
        task_id=task.task_id,
        payload={"backend": backend, "seed": seed, "implementation_source": "real_upstream"},
    )
    bus.emit(
        EventType.ENV_RESET,
        episode_id=episode_id,
        step_id=1,
        task_id=task.task_id,
        payload=json_safe_payload({"observation": initial_observation, "info": initial_info}),
    )

    action_rows: list[dict[str, Any]] = []
    obs = initial_observation
    info = dict(initial_info)
    reward = 0.0
    terminated = False
    truncated = False
    step_count = 0
    start_ns = time.perf_counter_ns()
    next_step_id = 2
    model_action: dict[str, Any] = {"events": []}

    for turn in range(budget):
        model_action, parsed = _driver_action(
            driver,
            task,
            observation=obs,
            info=info,
            backend=backend,
        )
        for event in model_action["events"]:
            kind = event.get("event")
            if kind == "model_request_submit":
                event_type = EventType.LLM_REQUEST_SUBMIT
            elif kind == "model_request_start":
                event_type = EventType.LLM_REQUEST_START
            elif kind == "model_request_end":
                event_type = EventType.LLM_REQUEST_END
            else:
                continue
            bus.emit(
                event_type,
                episode_id=episode_id,
                step_id=next_step_id,
                task_id=task.task_id,
                queue_wait_ms=float(event.get("queue_wait_ms") or 0.0),
                payload={key: value for key, value in event.items() if key != "event"},
            )
            next_step_id += 1

        translated_action = translate_action_for_backend(parsed.canonical_action, backend=backend)
        action_rows.append(
            {
                "turn": turn,
                "canonical_action": parsed.canonical_action,
                "translated_action": translated_action,
                "parse_status": parsed.parse_status,
                "invalid_action": parsed.invalid_action,
            }
        )
        bus.emit(
            EventType.ENV_STEP_START,
            episode_id=episode_id,
            step_id=next_step_id,
            task_id=task.task_id,
            payload={"action": translated_action},
        )
        next_step_id += 1
        obs, reward, terminated, truncated, info = env.step(translated_action)
        step_count += 1
        bus.emit(
            EventType.ENV_STEP_END,
            episode_id=episode_id,
            step_id=next_step_id,
            task_id=task.task_id,
            payload=json_safe_payload(
                {
                    "observation": obs,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                }
            ),
        )
        next_step_id += 1
        if terminated or truncated:
            break

    bus.emit(
        EventType.RUN_END,
        episode_id=episode_id,
        step_id=next_step_id,
        task_id=task.task_id,
        payload={"reward": reward, "terminated": terminated, "truncated": truncated},
    )
    env.close()

    action_log_path = run_context.paths.root / "action_log.jsonl"
    action_log_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in action_rows),
        encoding="utf-8",
    )
    manifest = {
        "task_family": "miniwob",
        "task_id": task.task_id,
        "env_id": normalize_env_id_for_backend(task.env_id, backend),
        "backend": backend,
        "implementation_source": "real_upstream",
        "replay_class": "R0",
        "upstream_package_version": upstream_package_version,
        "browser_version": browser_version,
        "driver_version": driver_version,
        "task_manifest_hash": task.task_manifest_hash,
        **driver.metadata_row(),
    }
    manifest_path = run_context.write_manifest(manifest)
    validation = validate_jsonl(run_context.paths.events_jsonl) if run_context.telemetry_enabled() else None
    final_model_event = next(
        (event for event in reversed(model_action["events"]) if event.get("event") == "model_request_end"),
        {},
    )
    summary = {
        "run_id": run_context.run_id,
        "implementation_source": "real_upstream",
        "backend": backend,
        "replay_class": "R0",
        "task_id": task.task_id,
        "task_success": bool(reward > 0 or terminated),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "duration_ms": (time.perf_counter_ns() - start_ns) / 1_000_000.0,
        "env_steps": step_count,
        "upstream_package_version": upstream_package_version,
        "browser_version": browser_version,
        "driver_version": driver_version,
        "task_manifest_hash": task.task_manifest_hash,
        "obs_hash": _stable_hash(obs),
        "action_hash": _stable_hash(action_rows),
        "screenshot_hash": info.get("screenshot_hash"),
        "evidence_validation_pass": True if validation is None else validation.valid,
        "invalid_action": bool(action_rows[-1]["invalid_action"]) if action_rows else True,
        "parse_status": str(action_rows[-1]["parse_status"]) if action_rows else "empty",
        "model_latency_ms": float(final_model_event.get("model_latency_ms") or 0.0),
        "prompt_tokens": int(final_model_event.get("prompt_tokens") or 0),
        "completion_tokens": int(final_model_event.get("completion_tokens") or 0),
        "total_tokens": int(final_model_event.get("total_tokens") or 0),
        "budget": budget,
        "seed": seed,
        "trace_path": str(run_context.paths.events_jsonl),
        "summary_path": str(run_context.paths.root / "summary.json"),
        "manifest_path": str(manifest_path),
        **driver.metadata_row(),
    }
    (run_context.paths.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _run_task_worker(spec: MiniWobWorkerSpec) -> dict[str, Any]:
    info = backend_info(spec.backend, spec.upstream_root)
    env, obs, reset_info = make_real_env(
        spec.task,
        backend=spec.backend,
        upstream_root=spec.upstream_root,
        headless=spec.headless,
        seed=spec.seed,
    )
    model = _load_registry_model(spec.registry_path, spec.model_id)
    driver = make_openai_miniwob_driver(
        model,
        backend="vllm",
        host=spec.host,
        port=spec.port,
        budget=spec.budget,
        seed=spec.seed,
    )
    return run_llm_task(
        spec.task,
        output_root=spec.output_root,
        backend=spec.backend,
        upstream_package_version=info.upstream_package_version,
        browser_version=info.browser_version,
        driver_version=info.driver_version,
        telemetry_mode=spec.telemetry_mode,
        seed=spec.seed,
        budget=spec.budget,
        driver=driver,
        env=env,
        initial_observation=obs,
        initial_info=reset_info,
    )


def _write_aggregate(output_root: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_root / "llm_miniwob_summary.csv"
    fieldnames = [
        "task_id",
        "backend",
        "telemetry_mode",
        "task_success",
        "reward",
        "env_steps",
        "model_backend",
        "model_id",
        "policy_version",
        "budget",
        "seed",
        "model_latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "invalid_action",
        "parse_status",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def select_tasks(
    tasks: list[RealMiniWobTaskSpec],
    *,
    limit: int | None = None,
    task_ids: tuple[str, ...] = (),
) -> list[RealMiniWobTaskSpec]:
    selected = list(tasks)
    if task_ids:
        by_id: dict[str, RealMiniWobTaskSpec] = {}
        for task in tasks:
            for alias in _task_aliases(task):
                by_id.setdefault(alias, task)
        selected = [by_id[task_id] for task_id in task_ids if task_id in by_id]
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_llm_slice(
    config: MiniWobRealConfig,
    *,
    driver: LLMDriver,
    budget: int,
    seed: int,
    concurrency: int = 1,
    repeat: int = 1,
    limit: int | None = None,
    task_ids: tuple[str, ...] = (),
    registry_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 18000,
) -> list[dict[str, Any]]:
    selected_tasks = select_tasks(load_real_task_specs(config.task_selection_path), limit=limit, task_ids=task_ids)
    if not selected_tasks:
        raise ValueError(
            "no MiniWoB tasks selected for LLM run "
            f"(task_selection_path={config.task_selection_path}, limit={limit}, task_ids={task_ids})"
        )
    expanded_tasks: list[RealMiniWobTaskSpec] = []
    for _ in range(max(1, repeat)):
        expanded_tasks.extend(selected_tasks)

    if concurrency <= 1:
        rows = []
        info = backend_info(config.backend, config.upstream_root)
        for task in expanded_tasks:
            env, obs, reset_info = make_real_env(
                task,
                backend=config.backend,
                upstream_root=config.upstream_root,
                headless=config.headless,
                seed=seed,
            )
            rows.append(
                run_llm_task(
                    task,
                    output_root=config.output_root,
                    backend=config.backend,
                    upstream_package_version=info.upstream_package_version,
                    browser_version=info.browser_version,
                    driver_version=info.driver_version,
                    telemetry_mode=config.telemetry_mode,
                    seed=seed,
                    budget=budget,
                    driver=driver,
                    env=env,
                    initial_observation=obs,
                    initial_info=reset_info,
                )
            )
    else:
        if registry_path is None:
            raise ValueError("registry_path is required when concurrency > 1")
        worker_specs = [
            MiniWobWorkerSpec(
                task=task,
                output_root=config.output_root,
                backend=config.backend,
                upstream_root=config.upstream_root,
                headless=config.headless,
                telemetry_mode=config.telemetry_mode,
                registry_path=registry_path,
                model_id=driver.metadata.model_id,
                host=host,
                port=port,
                budget=budget,
                seed=seed,
            )
            for task in expanded_tasks
        ]
        with ProcessPoolExecutor(max_workers=max(1, concurrency)) as pool:
            rows = list(pool.map(_run_task_worker, worker_specs))

    _write_aggregate(config.output_root, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/miniwob_real"))
    parser.add_argument("--telemetry-mode", choices=("off", "basic", "full"), default="full")
    parser.add_argument("--backend", choices=("browsergym_miniwob", "farama_selenium"), default="browsergym_miniwob")
    parser.add_argument(
        "--model-backend",
        choices=("vllm", "sglang"),
        default=os.environ.get("NIPS_MODEL_BACKEND", "vllm"),
        help="OpenAI-compatible model serving backend used for driver metadata.",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--registry", type=Path, default=Path("manifests/models/local_and_hf_models.yaml"))
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = MiniWobRealConfig.from_env(
        output_root=args.output_root,
        telemetry_mode=args.telemetry_mode,
    )
    model = _load_registry_model(args.registry, args.model_id)
    driver = make_openai_miniwob_driver(
        model,
        backend=args.model_backend,
        host=args.host,
        port=args.port,
        budget=args.budget,
        seed=args.seed,
    )
    rows = run_llm_slice(
        config,
        driver=driver,
        budget=args.budget,
        seed=args.seed,
        concurrency=args.concurrency,
        repeat=args.repeat,
        limit=args.limit,
        task_ids=tuple(args.task_id),
        registry_path=args.registry,
        host=args.host,
        port=args.port,
    )
    print(json.dumps({"task_count": len(rows), "summary_path": str(config.output_root / "llm_miniwob_summary.csv")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
