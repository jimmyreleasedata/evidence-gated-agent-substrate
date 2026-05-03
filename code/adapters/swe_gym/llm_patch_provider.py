"""Generic LLM patch-provider helpers for SWE real-upstream traffic."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import urllib.request
from urllib.error import HTTPError, URLError

from adapters.swe_gym.real_config import SweRealConfig
from adapters.swe_gym.real_swebench_harness import load_image_manifest
from adapters.swe_gym.real_task_loader import PatchCondition, RealSweTaskSpec, load_real_task_specs
from adapters.swe_gym.real_verifier import run_real_swe_task
from adapters.webarena_verified.llm_live_runner import _load_registry_model
from runtime.agent_driver.patch_provider import PatchProviderDriver
from runtime.agent_driver.agent_driver.llm_driver import LLMDriverConfig
from runtime.model_backend_client import OpenAICompatibleClient
from runtime.rl_closure.backends import served_model_alias


@dataclass(frozen=True, slots=True)
class ParsedPatchResponse:
    patch_text: str
    parse_status: str
    invalid_patch: bool


def extract_patch_from_response(response_text: str) -> ParsedPatchResponse:
    text = response_text.strip()
    if "diff --git " in text and not text.lstrip().startswith("```"):
        start = text.find("diff --git ")
        text = text[start:]
        lines = [line for line in text.splitlines() if line.strip() != "```"]
        text = "\n".join(lines).strip()
    elif "```diff" in text:
        _, _, remainder = text.partition("```diff")
        patch_text, _, _ = remainder.partition("```")
        text = patch_text.strip()
    elif "```" in text:
        _, _, remainder = text.partition("```")
        patch_text, _, _ = remainder.partition("```")
        text = patch_text.strip()

    if not text.startswith("diff --git "):
        return ParsedPatchResponse(patch_text="", parse_status="missing_diff", invalid_patch=True)
    if not text.endswith("\n"):
        text = f"{text}\n"
    return ParsedPatchResponse(patch_text=text, parse_status="ok", invalid_patch=False)


def build_patch_prompt(
    *,
    instance_id: str,
    repo: str,
    base_commit: str,
    problem_statement: str,
    repo_context: str | None = None,
) -> str:
    context_block = repo_context.strip() if repo_context and repo_context.strip() else "No additional repo context provided."
    return (
        "You are a patch-generation assistant for SWE-bench style tasks.\n"
        "Return a unified diff only. Do not include markdown fences.\n"
        f"Instance: {instance_id}\n"
        f"Repository: {repo}\n"
        f"Base commit: {base_commit}\n"
        "Problem statement:\n"
        f"{problem_statement.strip()}\n"
        "Repository context:\n"
        f"{context_block}\n"
    )


def make_patch_driver(
    model: dict[str, Any],
    *,
    host: str,
    port: int,
    budget: int,
    seed: int,
    backend: str,
) -> PatchProviderDriver:
    model_id = str(model["model_id"])
    model_path = str(model.get("snapshot_path") or model.get("model_path") or model.get("model_path_or_hf_id") or model_id)
    served_model_name = str(model.get("served_model_name") or served_model_alias(model_id))
    client = OpenAICompatibleClient(api_base_url=f"http://{host}:{port}")

    def _generate(prompt_text: str) -> str:
        payload = client.generate(
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
            max_tokens=1024,
        )
        return str(payload["text"])

    return PatchProviderDriver(
        LLMDriverConfig(
            driver_id=f"llm-patch/{model_id}",
            driver_version="0.1.0",
            model_backend=backend,
            backend_engine=backend,
            backend_version="",
            model_id=model_id,
            model_family=str(model.get("model_family") or "unknown"),
            model_revision=str(model.get("model_revision") or model.get("checkpoint_id") or model.get("model_path_hash") or ""),
            model_path_or_hf_id=model_path,
            tokenizer_path=str(model.get("tokenizer_path") or model_path),
            policy_version="llm_swe_patch_v1",
            prompt_template="{obs}",
            action_parser_version="swe-patch-parser-v1",
            budget=budget,
            seed=seed,
        ),
        generate_fn=_generate,
    )


def make_vllm_patch_driver(model: dict[str, Any], *, host: str, port: int, budget: int, seed: int) -> PatchProviderDriver:
    return make_patch_driver(
        model,
        host=host,
        port=port,
        budget=budget,
        seed=seed,
        backend="vllm",
    )


def load_registry_model(registry_path: Path, model_id: str) -> dict[str, Any]:
    return _load_registry_model(registry_path, model_id)


def generate_llm_patch(
    *,
    driver: PatchProviderDriver,
    instance_id: str,
    repo: str,
    base_commit: str,
    problem_statement: str,
    repo_context: str | None = None,
) -> dict[str, Any]:
    prompt = build_patch_prompt(
        instance_id=instance_id,
        repo=repo,
        base_commit=base_commit,
        problem_statement=problem_statement,
        repo_context=repo_context,
    )
    action = driver.act(prompt, {"instance_id": instance_id, "repo": repo}, budget=driver.metadata.budget)
    parsed = extract_patch_from_response(str(action["action_text"]))
    terminal_event = {}
    for event in action.get("events") or []:
        if str(event.get("event") or "") == "model_request_end":
            terminal_event = dict(event)
    return {
        "patch_text": parsed.patch_text,
        "patch_hash": None if not parsed.patch_text else hashlib.sha256(parsed.patch_text.encode("utf-8")).hexdigest(),
        "patch_length": len(parsed.patch_text),
        "parse_status": parsed.parse_status,
        "invalid_patch": parsed.invalid_patch,
        "driver_metadata": driver.metadata_row(),
        "model_latency_ms": terminal_event.get("model_latency_ms"),
        "prompt_tokens": terminal_event.get("prompt_tokens"),
        "completion_tokens": terminal_event.get("completion_tokens"),
        "total_tokens": terminal_event.get("total_tokens"),
        "invalid_action": terminal_event.get("invalid_action"),
        "events": list(action.get("events") or []),
        "raw_response_path": None,
        "raw_response": str(action["action_text"]),
        "serialized": json.dumps({"patch_text": parsed.patch_text, "parse_status": parsed.parse_status}, sort_keys=True),
    }


def _issue_number_from_instance(instance_id: str, repo: str) -> int:
    if "__" in instance_id and "-" in instance_id:
        suffix = instance_id.rsplit("-", 1)[-1]
        return int(suffix)
    suffix = instance_id.rsplit("-", 1)[-1]
    return int(suffix)


def fetch_problem_statement(repo: str, instance_id: str, *, timeout_s: float = 30.0) -> str:
    issue_number = _issue_number_from_instance(instance_id, repo)
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues/{issue_number}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "nips-bench-llm-patch-provider",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        return f"[problem statement unavailable for {instance_id}: {exc}]"
    title = str(payload.get("title") or "").strip()
    body = str(payload.get("body") or "").strip()
    comments = payload.get("comments")
    lines = []
    if title:
        lines.append(title)
    if body:
        lines.append(body)
    if comments not in (None, 0):
        lines.append(f"[github_comments_count={comments}]")
    return "\n\n".join(lines).strip()


def build_repo_context(task: RealSweTaskSpec) -> str:
    payload = {
        "repo": task.repo,
        "base_commit": task.base_commit,
        "fail_to_pass": list(task.fail_to_pass),
        "pass_to_pass": list(task.pass_to_pass),
        "test_command": task.test_command,
        "build_command": task.build_command,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def task_with_llm_patch(task: RealSweTaskSpec, patch_text: str) -> RealSweTaskSpec:
    patch_conditions = list(task.patch_conditions)
    patch_conditions = [condition for condition in patch_conditions if condition.name != "llm_patch"]
    patch_conditions.append(PatchCondition(name="llm_patch", kind="git_apply", patch=patch_text or ""))
    return replace(task, patch_conditions=patch_conditions)


def select_tasks(
    tasks: list[RealSweTaskSpec],
    *,
    limit: int | None = None,
    instance_ids: tuple[str, ...] = (),
) -> list[RealSweTaskSpec]:
    selected = list(tasks)
    if instance_ids:
        by_id = {task.instance_id: task for task in tasks}
        selected = [by_id[instance_id] for instance_id in instance_ids if instance_id in by_id]
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_llm_patch_task(
    task: RealSweTaskSpec,
    *,
    driver: PatchProviderDriver,
    config: SweRealConfig,
    image_manifest: dict[str, Any],
    output_root: Path,
    include_controls: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    problem_statement = fetch_problem_statement(task.repo, task.instance_id)
    repo_context = build_repo_context(task)
    generated = generate_llm_patch(
        driver=driver,
        instance_id=task.instance_id,
        repo=task.repo,
        base_commit=task.base_commit,
        problem_statement=problem_statement,
        repo_context=repo_context,
    )
    llm_task = task_with_llm_patch(task, generated["patch_text"])
    rows: list[dict[str, Any]] = []
    llm_row = run_real_swe_task(
        llm_task,
        "llm_patch",
        config,
        image_manifest,
        driver_metadata=generated["driver_metadata"],
        generated_patch_metadata={
            "patch_hash": generated["patch_hash"],
            "patch_length": generated["patch_length"],
            "patch_candidate_parse_status": generated["parse_status"],
            "patch_candidate_invalid_patch": generated["invalid_patch"],
            "patch_generation_latency_ms": generated.get("model_latency_ms"),
            "prompt_tokens": generated.get("prompt_tokens"),
            "completion_tokens": generated.get("completion_tokens"),
            "total_tokens": generated.get("total_tokens"),
            "invalid_action": generated.get("invalid_action"),
            "problem_statement_source": "github_issue",
        },
    )
    run_root = Path(llm_row["summary_path"]).parent
    (run_root / "generated_patch.diff").write_text(generated["patch_text"], encoding="utf-8")
    (run_root / "raw_model_response.txt").write_text(generated["raw_response"], encoding="utf-8")
    rows.append(llm_row)

    for condition_name in include_controls:
        if any(condition.name == condition_name for condition in task.patch_conditions):
            rows.append(run_real_swe_task(task, condition_name, config, image_manifest))
    return rows


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = output_root / "llm_swe_patch_summary.csv"
    fieldnames = [
        "instance_id",
        "patch_condition",
        "passed",
        "implementation_source",
        "driver_id",
        "driver_type",
        "model_backend",
        "model_id",
        "policy_version",
        "patch_hash",
        "patch_length",
        "patch_apply_success",
        "patch_parse_status",
        "build_ms",
        "test_ms",
        "verifier_service_ms",
        "verifier_queue_wait_ms",
        "summary_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})

    md_path = output_root / "llm_swe_patch_summary.md"
    llm_rows = [row for row in rows if row.get("patch_condition") == "llm_patch"]
    pass_rate = (sum(1 for row in llm_rows if row.get("passed")) / len(llm_rows)) if llm_rows else 0.0
    md_path.write_text(
        "\n".join(
            [
                "# LLM SWE Patch Summary",
                "",
                f"- rows: `{len(rows)}`",
                f"- llm_patch_rows: `{len(llm_rows)}`",
                f"- llm_patch_pass_rate: `{pass_rate:.3f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="")
    parser.add_argument("--controls", default="oracle,noop")
    parser.add_argument("--instance-ids", default="")
    args = parser.parse_args()

    config = SweRealConfig.from_env(output_root=args.output_root, telemetry_mode="full")
    tasks = load_real_task_specs(config.task_selection_path)
    instance_ids = tuple(part.strip() for part in args.instance_ids.split(",") if part.strip())
    tasks = select_tasks(tasks, limit=args.limit, instance_ids=instance_ids)
    image_manifest = load_image_manifest(config.image_manifest_path)
    model = load_registry_model(args.registry, args.model_id)
    backend = str(args.backend or model.get("backend") or os.environ.get("NIPS_MODEL_BACKEND") or "vllm")
    driver = make_patch_driver(model, host=args.host, port=args.port, budget=args.budget, seed=args.seed, backend=backend)
    include_controls = tuple(part.strip() for part in args.controls.split(",") if part.strip())

    rows: list[dict[str, Any]] = []
    for task in tasks:
        rows.extend(
            run_llm_patch_task(
                task,
                driver=driver,
                config=config,
                image_manifest=image_manifest,
                output_root=args.output_root,
                include_controls=include_controls,
            )
        )
    _write_summary(args.output_root, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
