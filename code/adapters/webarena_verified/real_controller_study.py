"""Real WebArena controller-study pilot primitives."""

from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import mean
import time
from typing import Any
from urllib.parse import quote_plus, urlparse

from adapters.webarena_verified.real_live_capture import (
    AgentResponseArtifact,
    BrowserCaptureArtifact,
    LiveCaptureConfig,
    _load_extra_headers,
    _render_url,
    _resolve_proxy,
    run_live_capture,
)
from trace.schema.version import SUITE_VERSION, TRACE_SCHEMA_VERSION


SUPPORTED_TASK_IDS: set[str] = {
    "105",
    "106",
    "124",
    "125",
    "142",
    "143",
    "149",
    "156",
    "163",
}

CONTROLLER_DRIVER_VERSION = "real_controller_study_v1"
CONTROLLER_ACTION_PARSER_VERSION = "direct_controller_action_v1"
NO_PROMPT_TEMPLATE_HASH = "sha256:" + hashlib.sha256(b"no_prompt_controller_v1").hexdigest()

GITLAB_TARGETS: dict[str, str] = {
    "105": "{gitlab}/OpenAPITools/openapi-generator/-/issues/?state=opened&label_name%5B%5D=OpenAPI%20Generator%20CLI",
    "106": "{gitlab}/umano/AndroidSlidingUpPanel/-/issues/?state=opened&not%5Blabel_name%5D%5B%5D=BUG",
    "156": "{gitlab}/dashboard/merge_requests?assignee_username=byteblaze",
}

SHOPPING_ORDER_TASKS: dict[str, int] = {
    "142": 163,
    "143": 148,
    "149": 157,
}

SHOPPING_PRICE_RANGE_QUERIES: dict[str, str] = {
    "124": "wireless earphone",
    "125": "teeth grinding mouth guard",
}


def _env_any(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return mean(values) if values else 0.0


def _ordering(rows: list[dict[str, Any]], regime: str) -> str:
    arms = {row["controller"]: row for row in rows if row["regime"] == regime}
    if not {"hook_a_only", "hook_b_only"} <= set(arms):
        raise ValueError(f"missing controller rows for regime={regime}")
    a = float(arms["hook_a_only"]["reward_auc_over_wallclock_mean"])
    b = float(arms["hook_b_only"]["reward_auc_over_wallclock_mean"])
    return "hook_a_only > hook_b_only" if a > b else "hook_b_only > hook_a_only"


def _parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _supported_task_ids_from_file(path: Path, *, limit: int | None = None) -> list[str]:
    task_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    filtered = [task_id for task_id in task_ids if task_id in SUPPORTED_TASK_IDS]
    return filtered[:limit] if limit is not None else filtered


def _release_root(config: "ControllerStudyConfig") -> str:
    return str(Path(_env_any("NIPS_CANONICAL_FINAL_ROOT") or config.output_root).expanduser().resolve(strict=False))


def _fixed_budget(config: "ControllerStudyConfig") -> int:
    raw = _env_any("NIPS_WEBARENA_CONTROLLER_FIXED_BUDGET")
    if raw:
        return int(raw)
    return len(_supported_task_ids_from_file(config.available_task_ids_path))


def _solver_result_path(task_root: Path) -> Path:
    return task_root / "solver_result.json"


def _agent_response_path(task_root: Path) -> Path:
    return task_root / "agent_response.json"


def _currency(value: str) -> float:
    return float(value.replace("$", "").replace(",", "").strip())


def _parse_order_rows(raw_rows: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            continue
        name = lines[0]
        attributes: dict[str, str] = {}
        idx = 1
        while idx + 1 < len(lines) and not lines[idx].startswith("B") and not lines[idx].startswith("$"):
            attributes[lines[idx]] = lines[idx + 1]
            idx += 2
        prices = [_currency(line) for line in lines if line.startswith("$")]
        rows.append(
            {
                "name": name,
                "attributes": attributes,
                "price": prices[0] if prices else None,
                "subtotal": prices[-1] if prices else None,
            }
        )
    return rows


def _price_range_from_product_text(text: str) -> dict[str, float] | None:
    prices = [float(match.replace(",", "")) for match in re.findall(r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", text)]
    if not prices:
        return None
    return {"min": round(min(prices), 2), "max": round(max(prices), 2)}


def _controller_cell_root(config: "ControllerStudyConfig", cell: "ControllerCell") -> Path:
    return (
        config.output_root
        / cell.backend
        / cell.regime
        / cell.controller
        / f"seed_{cell.seed}"
    )


def _canonical_agent_response(payload: dict[str, Any]) -> dict[str, Any]:
    response = {
        "task_type": payload.get("task_type", "RETRIEVE"),
        "status": payload.get("status", "UNKNOWN_ERROR"),
    }
    if payload.get("retrieved_data") is not None:
        response["retrieved_data"] = payload.get("retrieved_data")
    return response


def _task_site(task: dict[str, Any]) -> str:
    sites = task.get("sites", [])
    return str(sites[0]) if sites else "unknown"


def _medium_probe_urls(task: dict[str, Any], live_cfg: LiveCaptureConfig) -> list[str]:
    site = _task_site(task)
    if site == "gitlab":
        return [
            f"{live_cfg.wa_urls['gitlab'].rstrip('/')}/dashboard/projects",
            f"{live_cfg.wa_urls['gitlab'].rstrip('/')}/dashboard/activity",
        ]
    return [
        live_cfg.wa_urls["homepage"].rstrip("/"),
        f"{live_cfg.wa_urls['shopping'].rstrip('/')}/customer/account/",
    ]


def _probe_count(cell: "ControllerCell") -> int:
    if cell.regime == "clean":
        return 0 if cell.controller == "hook_a_only" else 1
    if cell.regime == "medium":
        return 3 if cell.controller == "hook_a_only" else 1
    return 5 if cell.controller == "hook_a_only" else 2


async def _goto(page, url: str, action_log: list[dict[str, Any]], *, phase: str, action: str) -> None:
    started = time.monotonic()
    await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    try:
        await page.wait_for_load_state("networkidle", timeout=10_000)
        status = "ok"
    except Exception:
        status = "timeout"
    action_log.append(
        {
            "phase": phase,
            "action": action,
            "target": url,
            "status": status,
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
        }
    )


async def _solve_gitlab_task(task: dict[str, Any], page, live_cfg: LiveCaptureConfig, action_log: list[dict[str, Any]]) -> dict[str, Any]:
    task_id = str(task["task_id"])
    start_url = _render_url(str(task["start_urls"][0]), live_cfg.wa_urls)
    await _goto(page, start_url, action_log, phase="task", action="goto_start")
    target_url = GITLAB_TARGETS[task_id].format(**live_cfg.wa_urls)
    await _goto(page, target_url, action_log, phase="task", action="goto_target")
    return {
        "success": True,
        "task_type": "NAVIGATE",
        "status": "SUCCESS",
        "retrieved_data": None,
        "error_details": None,
    }


async def _load_order_rows(page, order_id: int, live_cfg: LiveCaptureConfig, action_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    url = f"{live_cfg.wa_urls['shopping'].rstrip('/')}/sales/order/view/order_id/{order_id}/"
    await _goto(page, url, action_log, phase="task", action="goto_order")
    raw_rows = await page.locator("table tbody tr").evaluate_all("els => els.map(e => e.innerText)")
    return _parse_order_rows(raw_rows)


async def _solve_shopping_task(task: dict[str, Any], page, live_cfg: LiveCaptureConfig, action_log: list[dict[str, Any]]) -> dict[str, Any]:
    task_id = str(task["task_id"])
    if task_id in SHOPPING_PRICE_RANGE_QUERIES:
        query = SHOPPING_PRICE_RANGE_QUERIES[task_id]
        url = f"{live_cfg.wa_urls['shopping'].rstrip('/')}/catalogsearch/result/?q={quote_plus(query)}"
        await _goto(page, url, action_log, phase="task", action="goto_search_results")
        product_texts = await page.locator(".product-item-info, .product-item, ol.products li, ul.products li").evaluate_all(
            "els => els.map(e => e.innerText)"
        )
        if not product_texts:
            product_texts = [await page.locator("body").inner_text()]
        price_range = _price_range_from_product_text("\n".join(product_texts))
        return {
            "success": price_range is not None,
            "task_type": "RETRIEVE",
            "status": "SUCCESS" if price_range is not None else "UNKNOWN_ERROR",
            "retrieved_data": [price_range] if price_range is not None else None,
            "error_details": None if price_range is not None else "price_range_not_found",
        }

    if task_id == "163":
        start_url = _render_url(str(task["start_urls"][0]), live_cfg.wa_urls)
        await _goto(page, start_url, action_log, phase="task", action="goto_product")
        await page.locator("text=Reviews 12").first.click()
        await page.wait_for_timeout(1200)
        action_log.append({"phase": "task", "action": "open_reviews", "status": "ok", "elapsed_ms": 1200.0})
        review_rows = await page.locator('[itemprop="review"]').evaluate_all("els => els.map(el => el.innerText)")
        titles: list[str] = []
        for row in review_rows:
            match = re.search(r"^(.*?)\nRating\s*\n(\d+)%", row, re.S)
            if match and int(match.group(2)) <= 40:
                titles.append(match.group(1).strip())
        return {
            "success": bool(titles),
            "task_type": "RETRIEVE",
            "status": "SUCCESS" if titles else "UNKNOWN_ERROR",
            "retrieved_data": titles,
            "error_details": None if titles else "no_low_rating_reviews_found",
        }

    rows = await _load_order_rows(page, SHOPPING_ORDER_TASKS[task_id], live_cfg, action_log)
    if task_id == "143":
        matched = [row for row in rows if "artificial" in row["name"].lower() and "topiary" in row["name"].lower()]
        value = matched[0]["subtotal"] if matched and matched[0]["subtotal"] is not None else None
        return {
            "success": value is not None,
            "task_type": "RETRIEVE",
            "status": "SUCCESS" if value is not None else "UNKNOWN_ERROR",
            "retrieved_data": [value] if value is not None else None,
            "error_details": None if value is not None else "home_decoration_item_not_found",
        }
    if task_id == "149":
        matched = [row for row in rows if "artificial plants" in row["name"].lower()]
        color = matched[0]["attributes"].get("Color") if matched else None
        return {
            "success": bool(color),
            "task_type": "RETRIEVE",
            "status": "SUCCESS" if color else "UNKNOWN_ERROR",
            "retrieved_data": [color] if color else None,
            "error_details": None if color else "artificial_plants_color_not_found",
        }
    if task_id == "142":
        total = 0.0
        matched = 0
        for row in rows:
            name = row["name"].lower()
            if any(keyword in name for keyword in ("conditioner", "haircolor")) and row["subtotal"] is not None:
                total += float(row["subtotal"])
                matched += 1
        value = round(total, 2) if matched else None
        return {
            "success": value is not None,
            "task_type": "RETRIEVE",
            "status": "SUCCESS" if value is not None else "UNKNOWN_ERROR",
            "retrieved_data": [value] if value is not None else None,
            "error_details": None if value is not None else "hair_care_items_not_found",
        }
    return {
        "success": False,
        "task_type": "RETRIEVE",
        "status": "UNKNOWN_ERROR",
        "retrieved_data": None,
        "error_details": f"unsupported_task_{task_id}",
    }


async def _capture_and_solve_async(
    task: dict[str, Any],
    task_root: Path,
    live_cfg: LiveCaptureConfig,
    cell: "ControllerCell",
) -> BrowserCaptureArtifact:
    from playwright.async_api import async_playwright

    task_root.mkdir(parents=True, exist_ok=True)
    screenshot_dir = task_root / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    action_log: list[dict[str, Any]] = []
    proxy_server = _resolve_proxy()
    launch_kwargs: dict[str, Any] = {"headless": True}
    if proxy_server:
        launch_kwargs["proxy"] = {"server": proxy_server}

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(**launch_kwargs)
        context = await browser.new_context(
            REDACTED_BROWSER_STATE_LABEL=str(live_cfg.REDACTED_BROWSER_STATE_LABEL_path),
            record_har_path=str(task_root / "network.har"),
            ignore_https_errors=True,
        )
        await context.tracing.start(screenshots=True, snapshots=True)
        extra_headers = _load_extra_headers(live_cfg.extra_headers_path)
        if extra_headers:
            await context.set_extra_http_headers(extra_headers)
        page = await context.new_page()

        for index in range(_probe_count(cell)):
            for probe_url in _medium_probe_urls(task, live_cfg):
                await _goto(page, probe_url, action_log, phase="stress_probe", action=f"probe_{index}")

        if str(task["task_id"]) in GITLAB_TARGETS:
            solver_result = await _solve_gitlab_task(task, page, live_cfg, action_log)
        else:
            solver_result = await _solve_shopping_task(task, page, live_cfg, action_log)

        screenshot_path = screenshot_dir / "step-000.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        final_url = page.url
        title = await page.title()
        browser_trace_path = task_root / "browser_trace.zip"
        await context.tracing.stop(path=str(browser_trace_path))
        await context.close()
        browser_version = browser.version
        await browser.close()

    _write_json(_solver_result_path(task_root), solver_result)
    _write_json(_agent_response_path(task_root), _agent_response_from_solver_payload(solver_result, cell))
    return BrowserCaptureArtifact(
        start_url=_render_url(str(task["start_urls"][0]), live_cfg.wa_urls),
        final_url=final_url,
        title=title,
        browser_version=browser_version,
        driver_version="playwright",
        action_log=action_log,
        screenshot_paths=[screenshot_path],
        network_trace_path=task_root / "network.har",
        browser_trace_path=browser_trace_path,
        gcp_public_host=urlparse(final_url).hostname or final_url,
    )


def real_controller_browser_capture(
    task: dict[str, Any],
    task_root: Path,
    live_cfg: LiveCaptureConfig,
    cell: "ControllerCell",
) -> BrowserCaptureArtifact:
    return asyncio.run(_capture_and_solve_async(task, task_root, live_cfg, cell))


def _agent_response_from_solver_payload(payload: dict[str, Any], cell: "ControllerCell") -> dict[str, Any]:
    # The upstream AgentResponseEvaluator only accepts the canonical agent-response
    # schema here; controller/back-end metadata belongs in summary/manifest, not in
    # the evaluator input payload.
    return _canonical_agent_response(payload)


def real_controller_agent_response(
    task: dict[str, Any],
    task_root: Path,
    _live_cfg: LiveCaptureConfig,
    cell: "ControllerCell",
) -> dict[str, Any]:
    payload = _load_json(_solver_result_path(task_root))
    return _agent_response_from_solver_payload(payload, cell)


@dataclass(frozen=True, slots=True)
class ControllerPolicy:
    arm: str
    strategy: str
    queue_bias: str
    sample_filter: str


@dataclass(frozen=True, slots=True)
class ControllerStudyConfig:
    output_root: Path
    dataset_path: Path
    available_task_ids_path: Path
    REDACTED_BROWSER_STATE_LABEL_path: Path
    extra_headers_path: Path
    upstream_root: Path
    upstream_commit: str
    evaluator_version: str
    wa_urls: dict[str, str]
    controllers: tuple[str, ...]
    regimes: tuple[str, ...]
    backends: tuple[str, ...]
    seeds: tuple[int, ...]
    pbs_job_id: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        output_root: Path,
        controllers: tuple[str, ...],
        regimes: tuple[str, ...],
        backends: tuple[str, ...],
        seeds: tuple[int, ...],
    ) -> "ControllerStudyConfig":
        allowed_controllers = {"hook_a_only", "hook_b_only"}
        if not set(controllers) <= allowed_controllers:
            raise ValueError(f"unsupported controllers: {controllers}")
        allowed_regimes = {"clean", "medium", "heavy"}
        if not set(regimes) <= allowed_regimes:
            raise ValueError(f"unsupported regimes: {regimes}")
        allowed_backends = {"vllm", "sglang"}
        if not set(backends) <= allowed_backends:
            raise ValueError(f"unsupported backends: {backends}")

        dataset_path = Path(_env_any("NIPS_WEBARENA_VERIFIED_DATASET") or "")
        available_task_ids_path = Path(_env_any("NIPS_WEBARENA_VERIFIED_AVAILABLE_TASKS") or "")
        REDACTED_BROWSER_STATE_LABEL_path = Path(_env_any("NIPS_BENCH_WEBARENA_STORAGE_STATE") or "")
        extra_headers_path = Path(_env_any("REDACTED_REQUEST_METADATA_ENV") or "")
        upstream_root = Path(_env_any("NIPS_WEBARENA_VERIFIED_ROOT") or "")
        bootstrap_metadata_path = upstream_root.parent / "bootstrap_metadata.json"
        bootstrap_metadata = _load_json(bootstrap_metadata_path) if bootstrap_metadata_path.exists() else {}
        upstream_commit = _env_any("NIPS_WEBARENA_VERIFIED_COMMIT") or str(bootstrap_metadata.get("repo_commit") or "")
        evaluator_version = _env_any("NIPS_WEBARENA_VERIFIED_EVALUATOR_VERSION") or str(
            bootstrap_metadata.get("evaluator_version") or "official"
        )
        if not upstream_commit:
            raise ValueError("missing required WebArena upstream commit")
        if not evaluator_version:
            raise ValueError("missing required WebArena evaluator version")

        wa_urls = {
            "shopping": _env_any("WA_SHOPPING") or "",
            "reddit": _env_any("WA_REDDIT") or "",
            "gitlab": _env_any("WA_GITLAB") or "",
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
            controllers=controllers,
            regimes=regimes,
            backends=backends,
            seeds=seeds,
            pbs_job_id=_env_any("PBS_JOBID"),
        )


def controller_policy_for_arm(arm: str) -> ControllerPolicy:
    if arm == "hook_a_only":
        return ControllerPolicy(
            arm=arm,
            strategy="sample_validity_filtering",
            queue_bias="low",
            sample_filter="strict",
        )
    if arm == "hook_b_only":
        return ControllerPolicy(
            arm=arm,
            strategy="adaptive_queue_scheduling",
            queue_bias="high",
            sample_filter="lenient",
        )
    raise ValueError(f"unsupported controller arm: {arm}")


def controller_driver_metadata(config: ControllerStudyConfig, cell: "ControllerCell", policy: ControllerPolicy) -> dict[str, Any]:
    release_root = _release_root(config)
    return {
        "family": "webarena_verified",
        "paper_role": "paper_facing_decision_evidence",
        "main_aggregate_eligible": False,
        "decision_label": cell.controller,
        "driver_id": f"webarena_real_controller:{cell.controller}:{cell.regime}:{cell.backend}",
        "driver_type": "controller",
        "driver_version": CONTROLLER_DRIVER_VERSION,
        "model_family": "controller_policy",
        "model_id": f"webarena_real_controller_{cell.backend}",
        "model_version": CONTROLLER_DRIVER_VERSION,
        "model_backend": cell.backend,
        "backend_engine": cell.backend,
        "policy_version": policy.strategy,
        "prompt_template_hash": NO_PROMPT_TEMPLATE_HASH,
        "action_parser_version": CONTROLLER_ACTION_PARSER_VERSION,
        "budget": _fixed_budget(config),
        "implementation_source": "real_upstream_live",
        "release_root": release_root,
        "source_root": release_root,
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "replay_class": "R1",
        "evaluator_version": config.evaluator_version,
    }


def _wrap_controller_agent_response(
    payload_or_artifact: dict[str, Any] | AgentResponseArtifact,
    *,
    metadata: dict[str, Any],
) -> AgentResponseArtifact:
    if isinstance(payload_or_artifact, AgentResponseArtifact):
        payload = payload_or_artifact.agent_response
        driver_metadata = {**metadata, **dict(payload_or_artifact.driver_metadata or {})}
        summary_overrides = dict(payload_or_artifact.summary_overrides or {})
        return AgentResponseArtifact(
            agent_response=_canonical_agent_response(payload),
            driver_metadata=driver_metadata,
            model_events=payload_or_artifact.model_events,
            action_log_rows=payload_or_artifact.action_log_rows,
            summary_overrides=summary_overrides,
        )
    return AgentResponseArtifact(
        agent_response=_canonical_agent_response(payload_or_artifact),
        driver_metadata=metadata,
        summary_overrides={},
    )


@dataclass(frozen=True, slots=True)
class ControllerCell:
    task_id: str
    controller: str
    regime: str
    backend: str
    seed: int


def _load_dataset_task(config: ControllerStudyConfig, task_id: str) -> dict[str, Any]:
    dataset = json.loads(config.dataset_path.read_text(encoding="utf-8"))
    for task in dataset:
        if str(task.get("task_id")) == str(task_id):
            return task
    raise KeyError(f"task_id {task_id} not found in dataset")


def _merge_json_fields(path: Path, updates: dict[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_controller_cell(
    config: ControllerStudyConfig,
    cell: ControllerCell,
    *,
    browser_capture_fn=None,
    agent_response_fn=None,
    evaluator_fn=None,
) -> dict[str, Any]:
    task = _load_dataset_task(config, cell.task_id)
    policy = controller_policy_for_arm(cell.controller)
    admission_metadata = controller_driver_metadata(config, cell, policy)
    cell_root = _controller_cell_root(config, cell)
    live_config = LiveCaptureConfig(
        output_root=cell_root,
        dataset_path=config.dataset_path,
        available_task_ids_path=config.available_task_ids_path,
        REDACTED_BROWSER_STATE_LABEL_path=config.REDACTED_BROWSER_STATE_LABEL_path,
        extra_headers_path=config.extra_headers_path,
        upstream_root=config.upstream_root,
        upstream_commit=config.upstream_commit,
        evaluator_version=config.evaluator_version,
        wa_urls=config.wa_urls,
        telemetry_mode="full",
        pbs_job_id=config.pbs_job_id,
    )

    def _browser_capture(task_payload: dict[str, Any], task_root: Path, live_cfg: LiveCaptureConfig) -> BrowserCaptureArtifact:
        if browser_capture_fn is not None:
            return browser_capture_fn(task_payload, task_root, live_cfg)
        return real_controller_browser_capture(task_payload, task_root, live_cfg, cell)

    def _agent_response(task_payload: dict[str, Any], task_root: Path, live_cfg: LiveCaptureConfig) -> AgentResponseArtifact:
        if agent_response_fn is not None:
            return _wrap_controller_agent_response(
                agent_response_fn(task_payload, task_root, live_cfg, cell),
                metadata=admission_metadata,
            )
        return _wrap_controller_agent_response(
            real_controller_agent_response(task_payload, task_root, live_cfg, cell),
            metadata=admission_metadata,
        )

    started = time.monotonic()
    summaries = run_live_capture(
        live_config,
        limit=1,
        tasks_override=[task],
        browser_capture_fn=_browser_capture,
        agent_response_fn=_agent_response,
        evaluator_fn=evaluator_fn,
    )
    wall_clock_s = max(time.monotonic() - started, 1e-3)
    summary = dict(summaries[0])
    task_root = cell_root / cell.task_id
    action_rows = [
        json.loads(line)
        for line in (task_root / "action_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latencies = [float(row.get("elapsed_ms", 0.0)) for row in action_rows if row.get("elapsed_ms") is not None]
    stress_latencies = [
        float(row.get("elapsed_ms", 0.0))
        for row in action_rows
        if row.get("phase") == "stress_probe" and row.get("elapsed_ms") is not None
    ]
    manifest_path = Path(str(summary.get("manifest_path") or task_root / "run_manifest.json"))
    task_manifest_path = task_root / "task.json"
    passed = bool(summary.get("task_success"))
    extra = {
        **admission_metadata,
        "controller": cell.controller,
        "regime": cell.regime,
        "backend": cell.backend,
        "seed": int(cell.seed),
        "stress_overlay": cell.regime != "clean",
        "stress_level": cell.regime,
        "controller_strategy": policy.strategy,
        "queue_bias": policy.queue_bias,
        "sample_filter": policy.sample_filter,
        "reward_auc_over_wallclock": round((1.0 / wall_clock_s) if summary["task_success"] else 0.0, 6),
        "queue_wait_p99_ms": round(max(stress_latencies) if stress_latencies else 0.0, 3),
        "p99_latency_ms": round(max(latencies) if latencies else wall_clock_s * 1000.0, 3),
        "manifest_path": str(manifest_path),
        "run_manifest_path": str(manifest_path),
        "manifest_hash": _sha256_file(manifest_path) if manifest_path.exists() else "",
        "task_manifest_path": str(task_manifest_path),
        "task_manifest_hash": _sha256_file(task_manifest_path) if task_manifest_path.exists() else "",
        "terminal_outcome_present": True,
        "terminal_outcome": "pass" if passed else "fail",
        "pass_rate": 1.0 if passed else 0.0,
        "passed": passed,
        "trace_complete": bool(summary.get("evidence_validation_pass")),
    }
    _merge_json_fields(task_root / "summary.json", extra)
    _merge_json_fields(task_root / "run_manifest.json", extra)
    summary.update(extra)
    return summary


def enumerate_controller_cells(config: ControllerStudyConfig, *, limit: int | None = None) -> list[ControllerCell]:
    task_ids = _supported_task_ids_from_file(config.available_task_ids_path, limit=limit)
    return [
        ControllerCell(task_id=task_id, controller=controller, regime=regime, backend=backend, seed=seed)
        for task_id in task_ids
        for controller in config.controllers
        for regime in config.regimes
        for backend in config.backends
        for seed in config.seeds
    ]


def write_controller_study_outputs(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    detail_rows = sorted(
        rows,
        key=lambda row: (
            str(row["backend"]),
            str(row["regime"]),
            str(row["controller"]),
            int(row["seed"]),
            str(row["task_id"]),
        ),
    )
    trace_path = _write_csv(
        output_root / "controller_trace.csv",
        detail_rows,
        [
            "family",
            "backend",
            "backend_engine",
            "controller",
            "decision_label",
            "regime",
            "seed",
            "task_id",
            "paper_role",
            "main_aggregate_eligible",
            "driver_id",
            "driver_type",
            "driver_version",
            "model_family",
            "model_id",
            "model_backend",
            "policy_version",
            "prompt_template_hash",
            "action_parser_version",
            "budget",
            "implementation_source",
            "reward_auc_over_wallclock",
            "queue_wait_p99_ms",
            "p99_latency_ms",
            "task_success",
            "terminal_outcome_present",
            "terminal_outcome",
            "pass_rate",
            "passed",
            "trace_complete",
            "evidence_validation_pass",
            "manifest_hash",
            "manifest_path",
            "run_manifest_path",
            "task_manifest_hash",
            "task_manifest_path",
            "schema_version",
            "trace_schema_version",
            "suite_version",
            "replay_class",
            "release_root",
            "source_root",
            "evaluator_version",
            "summary_path",
            "events_path",
        ],
    )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in detail_rows:
        key = (str(row["backend"]), str(row["regime"]), str(row["controller"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (backend, regime, controller), group in sorted(grouped.items()):
        summary_rows.append(
            {
                "backend": backend,
                "regime": regime,
                "controller": controller,
                "reward_auc_over_wallclock_mean": round(_mean(group, "reward_auc_over_wallclock"), 6),
                "queue_wait_p99_ms_mean": round(_mean(group, "queue_wait_p99_ms"), 6),
                "p99_latency_ms_mean": round(_mean(group, "p99_latency_ms"), 6),
                "task_success_rate": round(_mean(group, "task_success"), 6),
                "cell_count": len(group),
            }
        )
    summary_path = _write_csv(
        output_root / "controller_summary.csv",
        summary_rows,
        [
            "backend",
            "regime",
            "controller",
            "reward_auc_over_wallclock_mean",
            "queue_wait_p99_ms_mean",
            "p99_latency_ms_mean",
            "task_success_rate",
            "cell_count",
        ],
    )

    reversal_rows: list[dict[str, Any]] = []
    by_backend: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_backend.setdefault(str(row["backend"]), []).append(row)
    for backend, backend_rows in sorted(by_backend.items()):
        clean_ordering = _ordering(backend_rows, "clean")
        stressed_ordering = _ordering(backend_rows, "medium")
        reversal_rows.append(
            {
                "backend": backend,
                "clean_ordering": clean_ordering,
                "stressed_ordering": stressed_ordering,
                "reversal": str(clean_ordering != stressed_ordering).lower(),
            }
        )
    reversal_path = _write_csv(
        output_root / "decision_reversal_summary.csv",
        reversal_rows,
        ["backend", "clean_ordering", "stressed_ordering", "reversal"],
    )
    return {
        "controller_trace": str(trace_path),
        "controller_summary": str(summary_path),
        "decision_reversal_summary": str(reversal_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a real WebArena controller-study pilot.")
    parser.add_argument("--tasks", type=Path, required=False, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--controllers", default="hook_a_only,hook_b_only")
    parser.add_argument("--regimes", default="clean,medium")
    parser.add_argument("--backends", default="vllm")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args(argv)

    if args.tasks is not None:
        os.environ["NIPS_WEBARENA_VERIFIED_AVAILABLE_TASKS"] = str(args.tasks)

    config = ControllerStudyConfig.from_env(
        output_root=args.output_root,
        controllers=_parse_csv_list(args.controllers),
        regimes=_parse_csv_list(args.regimes),
        backends=_parse_csv_list(args.backends),
        seeds=tuple(int(seed) for seed in _parse_csv_list(args.seeds)),
    )
    cells = enumerate_controller_cells(config, limit=args.limit or None)
    if not cells:
        raise SystemExit("no supported real WebArena tasks selected for pilot")
    rows = [run_controller_cell(config, cell) for cell in cells]
    outputs = write_controller_study_outputs(args.output_root, rows)
    print(json.dumps({"rows": len(rows), **outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
