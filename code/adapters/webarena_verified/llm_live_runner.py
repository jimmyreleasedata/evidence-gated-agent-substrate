"""Generic OpenAI-compatible WebArena live runner helpers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from time import monotonic
from typing import Any
import urllib.request
from urllib.parse import urlparse

import yaml

from adapters.webarena_verified.real_live_capture import (
    AgentResponseArtifact,
    BrowserCaptureArtifact,
    LiveCaptureConfig,
    run_live_capture,
)
from runtime.agent_driver.agent_driver.base import ModelActionRecord
from runtime.agent_driver.agent_driver.llm_driver import LLMDriver, LLMDriverConfig
from runtime.rl_closure.backends import served_model_alias


def _openai_post(base_url: str, payload: dict[str, Any], *, timeout_s: float = 180.0) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True, slots=True)
class ParsedWebArenaAction:
    action_type: str
    parse_status: str
    invalid_action: bool
    agent_response: dict[str, Any]
    target_url: str | None = None


def _default_agent_response(error_details: str) -> dict[str, Any]:
    return {
        "task_type": "RETRIEVE",
        "status": "UNKNOWN_ERROR",
        "retrieved_data": None,
        "error_details": error_details,
    }


def parse_browser_action(action_text: str, *, allowed_origins: set[str]) -> ParsedWebArenaAction:
    try:
        payload = json.loads(action_text)
    except json.JSONDecodeError:
        return ParsedWebArenaAction(
            action_type="invalid",
            parse_status="invalid_json",
            invalid_action=True,
            agent_response=_default_agent_response("invalid_llm_action_json"),
        )
    if not isinstance(payload, dict):
        return ParsedWebArenaAction(
            action_type="invalid",
            parse_status="invalid_shape",
            invalid_action=True,
            agent_response=_default_agent_response("invalid_llm_action_shape"),
        )

    action = str(payload.get("action") or "").strip().lower()
    if action == "answer":
        agent_response = {
            "task_type": str(payload.get("task_type") or "RETRIEVE"),
            "status": str(payload.get("status") or "UNKNOWN_ERROR"),
            "retrieved_data": payload.get("retrieved_data"),
        }
        error_details = payload.get("error_details")
        if error_details not in {None, ""}:
            agent_response["error_details"] = error_details
        return ParsedWebArenaAction(
            action_type="answer",
            parse_status="ok",
            invalid_action=False,
            agent_response=agent_response,
        )

    if action == "goto":
        target_url = str(payload.get("url") or "").strip()
        origin = f"{urlparse(target_url).scheme}://{urlparse(target_url).netloc}" if target_url else ""
        if not target_url or origin not in allowed_origins:
            return ParsedWebArenaAction(
                action_type="goto",
                parse_status="invalid_target",
                invalid_action=True,
                agent_response=_default_agent_response("llm_goto_target_not_allowed"),
                target_url=target_url or None,
            )
        return ParsedWebArenaAction(
            action_type="goto",
            parse_status="ok",
            invalid_action=False,
            agent_response=_default_agent_response("llm_goto_not_executed_in_smoke_runner"),
            target_url=target_url,
        )

    return ParsedWebArenaAction(
        action_type=action or "invalid",
        parse_status="unsupported_action",
        invalid_action=True,
        agent_response=_default_agent_response("unsupported_llm_action"),
    )


def _capture_observation(task: dict[str, Any], capture: BrowserCaptureArtifact) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id")),
        "intent": str(task.get("intent") or ""),
        "sites": list(task.get("sites") or []),
        "start_url": capture.start_url,
        "final_url": capture.final_url,
        "title": capture.title,
        "browser_version": capture.browser_version,
    }


class OpenAIWebArenaDriver(LLMDriver):
    def __init__(self, config: LLMDriverConfig, *, base_url: str, served_model_name: str) -> None:
        super().__init__(config, generate_fn=lambda _: "")
        self._base_url = base_url.rstrip("/")
        self._served_model_name = served_model_name

    def act(self, obs: Any, task_context: dict[str, Any], budget: int) -> dict[str, Any]:
        prompt_text = self._render_prompt(obs, task_context)
        started = monotonic()
        try:
            completion = _openai_post(
                self._base_url,
                {
                    "model": self._served_model_name,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": 0.0,
                    "max_tokens": 128,
                },
            )
            action_text = str(completion.get("choices", [{}])[0].get("message", {}).get("content") or "")
            usage = dict(completion.get("usage") or {})
            prompt_tokens = int(usage.get("prompt_tokens") or max(len(prompt_text.split()), 1))
            completion_tokens = int(usage.get("completion_tokens") or (max(len(action_text.split()), 1) if action_text else 0))
        except Exception as exc:  # noqa: BLE001
            completion = {}
            action_text = json.dumps(
                {
                    "action": "answer",
                    "task_type": "RETRIEVE",
                    "status": "UNKNOWN_ERROR",
                    "retrieved_data": None,
                    "error_details": f"model_request_failed:{type(exc).__name__}",
                }
            )
            prompt_tokens = max(len(prompt_text.split()), 1)
            completion_tokens = max(len(action_text.split()), 1)
        latency_ms = round((monotonic() - started) * 1000.0, 3)
        parsed_action, parse_status, invalid_action = self._parse_action(action_text)
        record = ModelActionRecord(
            metadata=self.metadata,
            obs_payload=str(obs),
            prompt_text=prompt_text,
            action_text=action_text,
            parsed_action=str(parsed_action),
            parse_status=parse_status,
            invalid_action=invalid_action,
            model_latency_ms=latency_ms,
            queue_wait_ms=0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return {
            "action_text": action_text,
            "parsed_action": parsed_action,
            "parse_status": parse_status,
            "invalid_action": invalid_action,
            "events": record.to_event_rows(),
            "budget": budget,
        }


def build_llm_agent_artifact(
    task: dict[str, Any],
    task_root: Path,
    live_config: LiveCaptureConfig,
    capture: BrowserCaptureArtifact,
    *,
    driver: OpenAIWebArenaDriver,
) -> AgentResponseArtifact:
    observation = _capture_observation(task, capture)
    task_context = {
        "task_id": str(task.get("task_id")),
        "intent": str(task.get("intent") or ""),
        "site": ",".join(task.get("sites", [])),
        "start_url": capture.start_url,
    }
    action = driver.act(observation, task_context, budget=driver.metadata.budget)
    allowed_origins = {
        f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        for url in live_config.wa_urls.values()
        if url and str(url).lower() != "todo"
    }
    parsed = parse_browser_action(str(action["action_text"]), allowed_origins=allowed_origins)
    (task_root / "llm_observation.json").write_text(
        json.dumps(observation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return AgentResponseArtifact(
        agent_response=parsed.agent_response,
        driver_metadata=driver.metadata_row(),
        model_events=list(action.get("events") or []),
        action_log_rows=[
            {
                "phase": "llm_driver",
                "action": parsed.action_type,
                "status": parsed.parse_status,
                "invalid_action": parsed.invalid_action,
                "target": parsed.target_url,
            }
        ],
        summary_overrides={
            "model_latency_ms": next(
                (
                    event.get("model_latency_ms")
                    for event in reversed(list(action.get("events") or []))
                    if event.get("event") == "model_request_end"
                ),
                0.0,
            ),
            "prompt_tokens": next(
                (
                    event.get("prompt_tokens")
                    for event in reversed(list(action.get("events") or []))
                    if event.get("event") == "model_request_end"
                ),
                0,
            ),
            "completion_tokens": next(
                (
                    event.get("completion_tokens")
                    for event in reversed(list(action.get("events") or []))
                    if event.get("event") == "model_request_end"
                ),
                0,
            ),
            "total_tokens": next(
                (
                    event.get("total_tokens")
                    for event in reversed(list(action.get("events") or []))
                    if event.get("event") == "model_request_end"
                ),
                0,
            ),
            "invalid_action": parsed.invalid_action,
            "parse_status": parsed.parse_status,
        },
    )


def _load_registry_model(registry_path: Path, model_id: str) -> dict[str, Any]:
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    for model in payload.get("models", []):
        if model.get("model_id") == model_id:
            return dict(model)
    raise KeyError(f"model_id not found in registry: {model_id}")


def make_openai_driver(
    model: dict[str, Any],
    *,
    backend: str,
    host: str,
    port: int,
    budget: int,
    seed: int,
) -> OpenAIWebArenaDriver:
    model_id = str(model["model_id"])
    model_path = str(model.get("snapshot_path") or model.get("model_path") or model.get("model_path_or_hf_id") or model_id)
    served_model_name = str(model.get("served_model_name") or served_model_alias(model_id))
    return OpenAIWebArenaDriver(
        LLMDriverConfig(
            driver_id=f"llm-driver/webarena/{model_id}",
            driver_version="0.1.0",
            model_backend=backend,
            backend_engine=backend,
            backend_version="",
            model_id=model_id,
            model_family=str(model.get("model_family") or "unknown"),
            model_revision=str(model.get("model_revision") or model.get("checkpoint_id") or model.get("model_path_hash") or ""),
            model_path_or_hf_id=model_path,
            tokenizer_path=str(model.get("tokenizer_path") or model_path),
            policy_version="llm_webarena_react_v1",
            prompt_template=(
                "You are a browser-task agent.\n"
                "Given the observation and task context, emit strict JSON only.\n"
                "Supported actions:\n"
                '1. {"action":"answer","task_type":"NAVIGATE"|"RETRIEVE","status":"SUCCESS"|"UNKNOWN_ERROR","retrieved_data":<value or null>,"error_details":<string or null>}\n'
                '2. {"action":"goto","url":"https://..."}\n'
                "Do not include markdown fences.\n"
                "Observation:\n{obs}\n"
                "Task context:\n{task_context}\n"
            ),
            action_parser_version="webarena-action-parser-v1",
            budget=budget,
            seed=seed,
        ),
        base_url=f"http://{host}:{port}",
        served_model_name=served_model_name,
    )


def _write_summary(output_root: Path, summaries: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = output_root / "llm_webarena_summary.csv"
    fieldnames = [
        "task_id",
        "site",
        "task_success",
        "evidence_validation_pass",
        "model_backend",
        "model_id",
        "policy_version",
        "model_latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "invalid_action",
        "parse_status",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {name: summary.get(name) for name in fieldnames}
            site_value = summary.get("site")
            if isinstance(site_value, str):
                row["site"] = site_value
            else:
                row["site"] = ",".join(site_value or [])
            writer.writerow(row)

    md_path = output_root / "llm_webarena_summary.md"
    total = len(summaries)
    passed = sum(1 for summary in summaries if summary.get("task_success"))
    evidence_pass = sum(1 for summary in summaries if summary.get("evidence_validation_pass"))
    p95 = 0.0
    if summaries:
        latencies = sorted(float(summary.get("model_latency_ms") or 0.0) for summary in summaries)
        p95 = latencies[min(len(latencies) - 1, max(0, int(round(0.95 * len(latencies))) - 1))]
    md_path.write_text(
        "\n".join(
            [
                "# LLM WebArena Summary",
                "",
                f"- tasks: {total}",
                f"- task_success: {passed}/{total}",
                f"- evidence_validation_pass: {evidence_pass}/{total}",
                f"- model_latency_ms_mean: {round(mean(float(summary.get('model_latency_ms') or 0.0) for summary in summaries), 3) if summaries else 0.0}",
                f"- model_latency_ms_p95: {round(p95, 3)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--telemetry-mode", default="full")
    args = parser.parse_args(argv)

    config = LiveCaptureConfig.from_env(
        output_root=args.output_root,
        available_tasks_path=args.tasks,
        telemetry_mode=args.telemetry_mode,
    )
    model = _load_registry_model(args.registry, args.model_id)
    driver = make_openai_driver(
        model,
        backend=args.backend,
        host=args.host,
        port=args.port,
        budget=args.budget,
        seed=args.seed,
    )
    summaries = run_live_capture(
        config,
        limit=args.limit or None,
        agent_response_fn=lambda task, task_root, live_config, capture: build_llm_agent_artifact(
            task,
            task_root,
            live_config,
            capture,
            driver=driver,
        ),
    )
    _write_summary(args.output_root, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
