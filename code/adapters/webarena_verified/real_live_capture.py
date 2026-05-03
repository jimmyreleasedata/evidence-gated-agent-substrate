"""Minimal real live-capture runner for WebArena Verified."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tarfile
from typing import Any, Callable
from urllib.parse import urlparse

import yaml

from runner.event_bus import EventBus
from runner.run_context import RunContext
from trace.schema.events import EventType
from trace.validators.schema_validator import validate_jsonl


DEFAULT_ASSET_ROOT = Path("artifact_release_root/nips_upstream_assets")
LIVE_BACKEND = "webarena_verified_live_capture"


def _env_any(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_proxy() -> str | None:
    return _env_any(
        "WEBARENA_PLAYWRIGHT_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    )


def _site_placeholder(site: str) -> str:
    return f"__{site.upper()}__"


def _render_url(url_template: str, wa_urls: dict[str, str]) -> str:
    rendered = url_template
    for site, base_url in wa_urls.items():
        rendered = rendered.replace(_site_placeholder(site), base_url.rstrip("/"))
    return rendered


def _expected_task_type(task: dict[str, Any]) -> str:
    for evaluator in task.get("eval", []):
        expected = evaluator.get("expected", {})
        task_type = expected.get("task_type")
        if isinstance(task_type, str) and task_type.strip():
            return task_type.strip().upper()
    return "NAVIGATE"


def _default_agent_response(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": _expected_task_type(task),
        "status": "UNKNOWN_ERROR",
        "retrieved_data": None,
        "error_details": "baseline_live_capture_no_policy",
    }


@dataclass(slots=True)
class BrowserCaptureArtifact:
    start_url: str
    final_url: str
    title: str | None
    browser_version: str
    driver_version: str
    action_log: list[dict[str, Any]]
    screenshot_paths: list[Path]
    network_trace_path: Path
    browser_trace_path: Path | None
    gcp_public_host: str


@dataclass(slots=True)
class AgentResponseArtifact:
    agent_response: dict[str, Any]
    driver_metadata: dict[str, Any] | None = None
    model_events: list[dict[str, Any]] | None = None
    action_log_rows: list[dict[str, Any]] | None = None
    summary_overrides: dict[str, Any] | None = None


@dataclass(slots=True)
class LiveCaptureConfig:
    output_root: Path
    dataset_path: Path
    available_task_ids_path: Path
    REDACTED_BROWSER_STATE_LABEL_path: Path
    extra_headers_path: Path
    upstream_root: Path
    upstream_commit: str
    evaluator_version: str
    wa_urls: dict[str, str]
    telemetry_mode: str = "full"
    pbs_job_id: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        output_root: Path,
        available_tasks_path: Path | None = None,
        telemetry_mode: str = "full",
    ) -> "LiveCaptureConfig":
        asset_root = Path(_env_any("NIPS_UPSTREAM_ASSET_ROOT") or DEFAULT_ASSET_ROOT).expanduser().resolve(strict=False)
        upstream_root = Path(
            _env_any("NIPS_WEBARENA_VERIFIED_ROOT")
            or asset_root / "webarena_verified" / "repo"
        ).expanduser().resolve(strict=False)
        dataset_path = Path(
            _env_any("NIPS_WEBARENA_VERIFIED_DATASET")
            or asset_root / "webarena_verified" / "datasets" / "webarena-verified-hard.json"
        ).expanduser().resolve(strict=False)
        available_task_ids_path = (
            available_tasks_path.expanduser().resolve(strict=False)
            if available_tasks_path is not None
            else Path(
                _env_any("NIPS_WEBARENA_VERIFIED_AVAILABLE_TASKS")
                or asset_root / "webarena_verified" / "gcp_live" / "webarena_verified_gcp_available_task_ids.txt"
            ).expanduser().resolve(strict=False)
        )
        REDACTED_BROWSER_STATE_LABEL_path = Path(
            _env_any("NIPS_BENCH_WEBARENA_STORAGE_STATE")
            or asset_root / "webarena_verified" / "gcp_live" / "webarena_REDACTED_BROWSER_STATE_LABEL.json"
        ).expanduser().resolve(strict=False)
        extra_headers_path = Path(
            _env_any("REDACTED_REQUEST_METADATA_ENV")
            or asset_root / "webarena_verified" / "gcp_live" / "webarena_extra_headers.json"
        ).expanduser().resolve(strict=False)

        bootstrap_metadata_path = upstream_root.parent / "bootstrap_metadata.json"
        bootstrap_metadata = _load_json(bootstrap_metadata_path) if bootstrap_metadata_path.exists() else {}
        evaluator_version = _env_any("NIPS_WEBARENA_VERIFIED_EVALUATOR_VERSION") or str(
            bootstrap_metadata.get("evaluator_version") or "official"
        )
        upstream_commit = _env_any("NIPS_WEBARENA_VERIFIED_COMMIT") or str(
            bootstrap_metadata.get("repo_commit") or ""
        )
        if not upstream_commit:
            raise ValueError("missing required WebArena upstream commit")

        wa_urls = {
            "shopping": _env_any("WA_SHOPPING") or "",
            "shopping_admin": _env_any("WA_SHOPPING_ADMIN") or "",
            "reddit": _env_any("WA_REDDIT") or "",
            "gitlab": _env_any("WA_GITLAB") or "",
            "wikipedia": _env_any("WA_WIKIPEDIA") or "",
            "map": _env_any("WA_MAP") or "",
            "homepage": _env_any("WA_HOMEPAGE") or "",
        }
        for required in ("shopping", "reddit", "gitlab", "homepage"):
            if not wa_urls[required]:
                raise ValueError(f"missing required live URL for {required}")

        return cls(
            output_root=output_root,
            dataset_path=dataset_path,
            available_task_ids_path=available_task_ids_path,
            REDACTED_BROWSER_STATE_LABEL_path=REDACTED_BROWSER_STATE_LABEL_path,
            extra_headers_path=extra_headers_path,
            upstream_root=upstream_root,
            upstream_commit=upstream_commit,
            evaluator_version=evaluator_version,
            wa_urls=wa_urls,
            telemetry_mode=telemetry_mode,
            pbs_job_id=_env_any("PBS_JOBID"),
        )


def load_live_capture_tasks(config: LiveCaptureConfig, *, limit: int | None = None) -> list[dict[str, Any]]:
    dataset = _load_json(config.dataset_path)
    if not isinstance(dataset, list):
        raise ValueError(f"invalid WebArena dataset payload: {config.dataset_path}")
    wanted = [
        int(line.strip())
        for line in config.available_task_ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    selected = [task for task in dataset if int(task.get("task_id")) in set(wanted)]
    selected.sort(key=lambda item: wanted.index(int(item["task_id"])))
    if limit is not None:
        selected = selected[:limit]
    return selected


def _load_extra_headers(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"extra headers file must contain a JSON object: {path}")
    return {str(key): str(value) for key, value in payload.items()}


def _load_webarena_api(config: LiveCaptureConfig):
    src_root = config.upstream_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from webarena_verified.api.internal.data_reader import WebArenaVerifiedDataReader
    from webarena_verified.api.internal.evaluator import WebArenaVerifiedEvaluator
    from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
    from webarena_verified.types.data import TaskSubset
    from webarena_verified.types.task import WebArenaSite

    full_dataset_path = config.dataset_path
    if "hard" in full_dataset_path.name:
        candidate = full_dataset_path.with_name(full_dataset_path.name.replace("hard", "full"))
        if candidate.exists():
            full_dataset_path = candidate

    env_map = {
        WebArenaSite.SHOPPING: EnvironmentConfig(urls=[config.wa_urls["shopping"]], active_url_idx=0),
        WebArenaSite.REDDIT: EnvironmentConfig(urls=[config.wa_urls["reddit"]], active_url_idx=0),
        WebArenaSite.GITLAB: EnvironmentConfig(urls=[config.wa_urls["gitlab"]], active_url_idx=0),
    }
    if config.wa_urls.get("shopping_admin") and config.wa_urls["shopping_admin"].lower() != "todo":
        env_map[WebArenaSite.SHOPPING_ADMIN] = EnvironmentConfig(urls=[config.wa_urls["shopping_admin"]], active_url_idx=0)
    if config.wa_urls.get("wikipedia") and config.wa_urls["wikipedia"].lower() != "todo":
        env_map[WebArenaSite.WIKIPEDIA] = EnvironmentConfig(urls=[config.wa_urls["wikipedia"]], active_url_idx=0)
    if config.wa_urls.get("map") and config.wa_urls["map"].lower() != "todo":
        env_map[WebArenaSite.MAP] = EnvironmentConfig(urls=[config.wa_urls["map"]], active_url_idx=0)
    subset_task_ids = [
        int(line.strip())
        for line in config.available_task_ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    subset = TaskSubset(
        description="Evaluation Environment live capture available tasks",
        task_ids=subset_task_ids,
        checksum=TaskSubset.compute_checksum(subset_task_ids),
    )
    wa_config = WebArenaVerifiedConfig(test_data_file=full_dataset_path, environments=env_map)
    reader = WebArenaVerifiedDataReader(wa_config, subset=subset)
    return WebArenaVerifiedEvaluator(config=wa_config, reader=reader)


async def _capture_with_playwright_async(task: dict[str, Any], task_root: Path, config: LiveCaptureConfig) -> BrowserCaptureArtifact:
    from playwright.async_api import async_playwright

    task_root.mkdir(parents=True, exist_ok=True)
    screenshot_dir = task_root / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    start_url = _render_url(str(task["start_urls"][0]), config.wa_urls)
    proxy_server = _resolve_proxy()
    launch_kwargs: dict[str, Any] = {"headless": True}
    if proxy_server:
        launch_kwargs["proxy"] = {"server": proxy_server}

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(**launch_kwargs)
        browser_version = browser.version
        context = await browser.new_context(
            REDACTED_BROWSER_STATE_LABEL=str(config.REDACTED_BROWSER_STATE_LABEL_path),
            record_har_path=str(task_root / "network.har"),
            ignore_https_errors=True,
        )
        await context.tracing.start(screenshots=True, snapshots=True)
        extra_headers = _load_extra_headers(config.extra_headers_path)
        if extra_headers:
            await context.set_extra_http_headers(extra_headers)

        page = await context.new_page()
        action_log: list[dict[str, Any]] = []
        await page.goto(start_url, wait_until="domcontentloaded", timeout=45_000)
        action_log.append({"step": 1, "action": "goto", "target": start_url, "status": "ok"})
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            action_log.append({"step": 2, "action": "wait_for_load_state", "target": "networkidle", "status": "timeout"})
        screenshot_path = screenshot_dir / "step-000.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        title = await page.title()
        final_url = page.url
        browser_trace_path = task_root / "browser_trace.zip"
        await context.tracing.stop(path=str(browser_trace_path))
        await context.close()
        await browser.close()

    return BrowserCaptureArtifact(
        start_url=start_url,
        final_url=final_url,
        title=title,
        browser_version=browser_version,
        driver_version="playwright",
        action_log=action_log,
        screenshot_paths=[screenshot_path],
        network_trace_path=task_root / "network.har",
        browser_trace_path=browser_trace_path,
        gcp_public_host=urlparse(start_url).hostname or start_url,
    )


def capture_with_playwright(task: dict[str, Any], task_root: Path, config: LiveCaptureConfig) -> BrowserCaptureArtifact:
    return asyncio.run(_capture_with_playwright_async(task, task_root, config))


def evaluate_live_task(task: dict[str, Any], task_root: Path, config: LiveCaptureConfig) -> dict[str, Any]:
    evaluator = _load_webarena_api(config)
    result = evaluator.evaluate_task(
        task_id=int(task["task_id"]),
        agent_response=task_root / "agent_response.json",
        network_trace=task_root / "network.har",
    )
    payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)  # type: ignore[arg-type]
    payload["evaluator_version"] = config.evaluator_version
    payload["evaluator_id"] = payload.get("evaluator_id") or "webarena_verified_real_evaluator"
    payload["passed"] = bool(payload.get("score", 0.0) == 1.0) if "passed" not in payload else bool(payload["passed"])
    return payload


def _write_action_log(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _agent_response_builder_accepts_capture(builder: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):
        return False
    positional = 0
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
    return positional >= 4


def _normalize_agent_response_artifact(value: Any) -> AgentResponseArtifact:
    if isinstance(value, AgentResponseArtifact):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"agent response builder returned unsupported type: {type(value).__name__}")
    return AgentResponseArtifact(agent_response=value)


def _build_agent_response_artifact(
    builder: Callable[..., Any],
    task: dict[str, Any],
    task_root: Path,
    config: LiveCaptureConfig,
    capture: BrowserCaptureArtifact,
) -> AgentResponseArtifact:
    if _agent_response_builder_accepts_capture(builder):
        return _normalize_agent_response_artifact(builder(task, task_root, config, capture))
    return _normalize_agent_response_artifact(builder(task, task_root, config))


_MODEL_EVENT_MAP = {
    "model_request_submit": EventType.LLM_REQUEST_SUBMIT,
    "model_request_start": EventType.LLM_REQUEST_START,
    "model_request_end": EventType.LLM_REQUEST_END,
}


def _emit_model_events(
    bus: EventBus,
    *,
    episode_id: str,
    step_id: int,
    task_id: str,
    artifact: AgentResponseArtifact,
) -> int:
    for event in artifact.model_events or []:
        event_type = _MODEL_EVENT_MAP.get(str(event.get("event") or ""))
        if event_type is None:
            continue
        bus.emit(
            event_type,
            episode_id=episode_id,
            step_id=step_id,
            task_id=task_id,
            queue_wait_ms=float(event.get("queue_wait_ms") or 0.0),
            payload=dict(event),
        )
        step_id += 1
    return step_id


def _write_capture_manifest(
    config: LiveCaptureConfig,
    task_root: Path,
    task: dict[str, Any],
    capture: BrowserCaptureArtifact,
    evaluator_output: dict[str, Any],
    run_context: RunContext,
    *,
    evidence_validation_pass: bool,
    REDACTED_BROWSER_STATE_LABEL_hash: str,
    extra_headers_hash: str,
    agent_artifact: AgentResponseArtifact | None = None,
) -> tuple[Path, Path]:
    trace_hash = _sha256_file(run_context.paths.events_jsonl)
    hashes = {
        "task_json": _sha256_file(task_root / "task.json"),
        "run_manifest": None,
        "events_jsonl": trace_hash,
        "action_log": _sha256_file(task_root / "action_log.jsonl"),
        "agent_response": _sha256_file(task_root / "agent_response.json"),
        "evaluator_output": _sha256_file(task_root / "evaluator_output.json"),
        "network_trace": _sha256_file(capture.network_trace_path),
        "REDACTED_BROWSER_STATE_LABEL": REDACTED_BROWSER_STATE_LABEL_hash,
        "extra_headers": extra_headers_hash,
        "screenshots": {path.name: _sha256_file(path) for path in capture.screenshot_paths},
    }
    if capture.browser_trace_path and capture.browser_trace_path.exists():
        hashes["browser_trace"] = _sha256_file(capture.browser_trace_path)

    task_id = str(task["task_id"])
    driver_metadata = dict(agent_artifact.driver_metadata or {}) if agent_artifact is not None else {}
    summary_overrides = dict(agent_artifact.summary_overrides or {}) if agent_artifact is not None else {}
    manifest_payload = {
            "backend": LIVE_BACKEND,
            "implementation_source": "real_upstream_live",
            "task_id": task_id,
            "site": ",".join(task.get("sites", [])),
            "start_url": capture.start_url,
            "final_url": capture.final_url,
            "upstream_commit": config.upstream_commit,
            "evaluator_version": config.evaluator_version,
            "browser_version": capture.browser_version,
            "driver_version": capture.driver_version,
            "REDACTED_BROWSER_STATE_LABEL_hash": REDACTED_BROWSER_STATE_LABEL_hash,
            "extra_headers_hash": extra_headers_hash,
            "trace_hash": trace_hash,
            "artifact_hash": _sha256_file(capture.network_trace_path),
            "task_success": bool(evaluator_output.get("passed")),
            "evidence_validation_pass": evidence_validation_pass,
            "error_class": evaluator_output.get("error_class"),
            "pbs_job_id": config.pbs_job_id,
            "gcp_public_host": capture.gcp_public_host,
            "capture_timestamp": run_context.make_event(
                event_type=EventType.RUN_END,
                episode_id=f"{task_id}-episode-0001",
                step_id=9999,
                task_id=task_id,
            ).timestamp_wall,
            "paths": {
                "task_json": str(task_root / "task.json"),
                "action_log_jsonl": str(task_root / "action_log.jsonl"),
                "agent_response_json": str(task_root / "agent_response.json"),
                "evaluator_output_json": str(task_root / "evaluator_output.json"),
                "network_har": str(capture.network_trace_path),
                "browser_trace": None if capture.browser_trace_path is None else str(capture.browser_trace_path),
            },
        }
    manifest_payload.update(driver_metadata)
    manifest_payload.update(summary_overrides)
    manifest = run_context.write_manifest(manifest_payload)
    hashes["run_manifest"] = _sha256_file(manifest)
    hashes_path = _write_json(task_root / "hashes.json", hashes)
    return manifest, hashes_path


def run_live_capture(
    config: LiveCaptureConfig,
    *,
    limit: int | None = None,
    tasks_override: list[dict[str, Any]] | None = None,
    browser_capture_fn: Callable[[dict[str, Any], Path, LiveCaptureConfig], BrowserCaptureArtifact] | None = None,
    agent_response_fn: Callable[..., dict[str, Any] | AgentResponseArtifact] | None = None,
    evaluator_fn: Callable[[dict[str, Any], Path, LiveCaptureConfig], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    browser_capture = browser_capture_fn or capture_with_playwright
    agent_response_builder = agent_response_fn or (lambda task, _task_root, _config: _default_agent_response(task))
    evaluator = evaluator_fn or evaluate_live_task

    config.output_root.mkdir(parents=True, exist_ok=True)
    tasks = list(tasks_override) if tasks_override is not None else load_live_capture_tasks(config, limit=limit)
    REDACTED_BROWSER_STATE_LABEL_hash = _sha256_file(config.REDACTED_BROWSER_STATE_LABEL_path)
    extra_headers_hash = _sha256_file(config.extra_headers_path)
    summaries: list[dict[str, Any]] = []
    captured_tasks: list[dict[str, Any]] = []

    for task in tasks:
        task_id = str(task["task_id"])
        task_root = config.output_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        _write_json(task_root / "task.json", task)

        run_context = RunContext.create(
            output_root=config.output_root,
            task_family="webarena_verified",
            telemetry_mode=config.telemetry_mode,
            run_id=task_id,
            environment_id=LIVE_BACKEND,
            model_id="baseline-live-runner",
            model_version="playwright",
            policy_version="minimal_live_baseline_v1",
            verifier_id="webarena_verified_real_evaluator",
            verifier_version=config.evaluator_version,
        )
        bus = EventBus(run_context)
        episode_id = f"{task_id}-episode-0001"
        step_id = 0
        bus.emit(
            EventType.RUN_START,
            episode_id=episode_id,
            step_id=step_id,
            task_id=task_id,
            payload={
                "mode": "live_capture",
                "implementation_source": "real_upstream_live",
                "upstream_commit": config.upstream_commit,
                "REDACTED_BROWSER_STATE_LABEL_hash": REDACTED_BROWSER_STATE_LABEL_hash,
                "extra_headers_hash": extra_headers_hash,
            },
        )
        step_id += 1
        try:
            capture = browser_capture(task, task_root, config)
            agent_artifact = _build_agent_response_artifact(agent_response_builder, task, task_root, config, capture)
            _write_json(task_root / "agent_response.json", agent_artifact.agent_response)
            action_log_rows = [*capture.action_log, *(agent_artifact.action_log_rows or [])]
            _write_action_log(task_root / "action_log.jsonl", action_log_rows)
            driver_metadata = dict(agent_artifact.driver_metadata or {})
            if driver_metadata:
                run_context.model_id = str(driver_metadata.get("model_id") or run_context.model_id or "")
                run_context.model_version = str(driver_metadata.get("model_version") or run_context.model_version or "")
                run_context.policy_version = str(driver_metadata.get("policy_version") or run_context.policy_version or "")
            bus.emit(
                EventType.ENV_RESET,
                episode_id=episode_id,
                step_id=step_id,
                task_id=task_id,
                payload={"start_url": capture.start_url, "site": task.get("sites", [])},
            )
            step_id += 1
            first_action = capture.action_log[0]["action"] if capture.action_log else "goto"
            bus.emit(
                EventType.ENV_STEP_START,
                episode_id=episode_id,
                step_id=step_id,
                task_id=task_id,
                payload={"action": first_action, "target": capture.start_url},
            )
            step_id += 1
            bus.emit(
                EventType.ENV_STEP_END,
                episode_id=episode_id,
                step_id=step_id,
                task_id=task_id,
                payload={
                    "observation": {"url": capture.final_url, "title": capture.title},
                    "info": {"browser_version": capture.browser_version, "driver_version": capture.driver_version},
                },
            )
            step_id += 1
            step_id = _emit_model_events(
                bus,
                episode_id=episode_id,
                step_id=step_id,
                task_id=task_id,
                artifact=agent_artifact,
            )
            bus.emit(
                EventType.VERIFIER_START,
                episode_id=episode_id,
                step_id=step_id,
                task_id=task_id,
                payload={"evaluator_version": config.evaluator_version},
            )
            step_id += 1
            evaluator_output = evaluator(task, task_root, config)
            _write_json(task_root / "evaluator_output.json", evaluator_output)
            bus.emit(
                EventType.VERIFIER_END,
                episode_id=episode_id,
                step_id=step_id,
                task_id=task_id,
                payload=evaluator_output,
            )
            step_id += 1
            bus.emit(
                EventType.RUN_END,
                episode_id=episode_id,
                step_id=step_id,
                task_id=task_id,
                payload={"passed": bool(evaluator_output.get("passed")), "score": evaluator_output.get("score", 0.0)},
            )
            validation = validate_jsonl(run_context.paths.events_jsonl)
            manifest_path, _ = _write_capture_manifest(
                config,
                task_root,
                task,
                capture,
                evaluator_output,
                run_context,
                evidence_validation_pass=validation.valid,
                REDACTED_BROWSER_STATE_LABEL_hash=REDACTED_BROWSER_STATE_LABEL_hash,
                extra_headers_hash=extra_headers_hash,
                agent_artifact=agent_artifact,
            )
            summary = {
                "task_id": task_id,
                "site": ",".join(task.get("sites", [])),
                "start_url": capture.start_url,
                "implementation_source": "real_upstream_live",
                "backend": LIVE_BACKEND,
                "upstream_commit": config.upstream_commit,
                "evaluator_version": config.evaluator_version,
                "browser_version": capture.browser_version,
                "driver_version": capture.driver_version,
                "REDACTED_BROWSER_STATE_LABEL_hash": REDACTED_BROWSER_STATE_LABEL_hash,
                "extra_headers_hash": extra_headers_hash,
                "trace_hash": _sha256_file(run_context.paths.events_jsonl),
                "artifact_hash": _sha256_file(capture.network_trace_path),
                "task_success": bool(evaluator_output.get("passed")),
                "evidence_validation_pass": validation.valid,
                "error_class": evaluator_output.get("error_class"),
                "pbs_job_id": config.pbs_job_id,
                "gcp_public_host": capture.gcp_public_host,
                "capture_timestamp": run_context.make_event(
                    event_type=EventType.RUN_END,
                    episode_id=episode_id,
                    step_id=step_id + 1,
                    task_id=task_id,
                ).timestamp_wall,
                "summary_path": str(task_root / "summary.json"),
                "manifest_path": str(manifest_path),
                "events_path": str(run_context.paths.events_jsonl),
            }
            summary.update(driver_metadata)
            summary.update(dict(agent_artifact.summary_overrides or {}))
        except Exception as exc:
            bus.emit(
                EventType.FAILURE,
                episode_id=episode_id,
                step_id=step_id,
                task_id=task_id,
                error_class=type(exc).__name__,
                payload={"message": str(exc)},
            )
            validation = validate_jsonl(run_context.paths.events_jsonl)
            manifest_path = run_context.write_manifest(
                {
                    "backend": LIVE_BACKEND,
                    "implementation_source": "real_upstream_live",
                    "task_id": task_id,
                    "upstream_commit": config.upstream_commit,
                    "evaluator_version": config.evaluator_version,
                    "REDACTED_BROWSER_STATE_LABEL_hash": REDACTED_BROWSER_STATE_LABEL_hash,
                    "extra_headers_hash": extra_headers_hash,
                    "error_class": type(exc).__name__,
                    "pbs_job_id": config.pbs_job_id,
                }
            )
            summary = {
                "task_id": task_id,
                "implementation_source": "real_upstream_live",
                "backend": LIVE_BACKEND,
                "upstream_commit": config.upstream_commit,
                "evaluator_version": config.evaluator_version,
                "REDACTED_BROWSER_STATE_LABEL_hash": REDACTED_BROWSER_STATE_LABEL_hash,
                "extra_headers_hash": extra_headers_hash,
                "task_success": False,
                "evidence_validation_pass": False,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "pbs_job_id": config.pbs_job_id,
                "summary_path": str(task_root / "summary.json"),
                "manifest_path": str(manifest_path),
                "events_path": str(run_context.paths.events_jsonl),
            }

        _write_json(task_root / "summary.json", summary)
        summaries.append(summary)
        captured_tasks.append(
            {
                "task_id": task_id,
                "trace_path": str(run_context.paths.events_jsonl),
                "evaluator_result_path": str(task_root / "evaluator_output.json"),
                "task_manifest_path": str(task_root / "task.json"),
                "trace_bundle_ref": str(task_root),
            }
        )

    _write_json(
        config.output_root / "capture_summary.json",
        {
            "task_count": len(summaries),
            "successful_tasks": sum(1 for row in summaries if row.get("task_success")),
            "evidence_valid_tasks": sum(1 for row in summaries if row.get("evidence_validation_pass")),
            "output_root": str(config.output_root),
        },
    )
    (config.output_root / "captured_tasks.yaml").write_text(
        yaml.safe_dump({"family": "webarena_verified", "suite_version": "0.1.0", "replay_class": "R1", "tasks": captured_tasks}, sort_keys=False),
        encoding="utf-8",
    )
    return summaries


def build_capture_bundle(output_root: Path, bundle_path: Path | None = None) -> Path:
    root = output_root.expanduser().resolve(strict=False)
    target = bundle_path or (root / "webarena_real_eval_trace_bundle.tar.gz")
    with tarfile.open(target, "w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if path == target or not path.is_file():
                continue
            tar.add(path, arcname=str(path.relative_to(root)))
    return target
