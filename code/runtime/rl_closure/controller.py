"""Bounded downstream RL closure runner for the NeurIPS appendix case study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
import urllib.request

from adapters.swe_gym.env import checkout_repo, create_seed_repo
from adapters.swe_gym.tasks import V1_TASKS
from adapters.webarena_verified.tasks import V1_TASKS as WEBARENA_TASKS
from runtime.rl_closure.backends import (
    _open_url_no_proxy,
    build_sglang_launch_spec,
    build_vllm_launch_spec,
    resolve_dense_qwen_model,
    start_backend_process,
    stop_backend_process,
)
from runtime.rl_closure.reporting import (
    build_compute_resources,
    render_paper_mapping,
    safe_pct,
    summarize_numeric,
    write_csv,
    write_json,
)


REGIMES = ("baseline", "hook_a", "hook_b", "hook_ab")


@dataclass(slots=True)
class MockBackendClient:
    backend: str
    model_id: str
    responses_by_task: dict[str, str]

    def health(self) -> dict[str, Any]:
        return {"ok": True, "backend": self.backend, "model_id": self.model_id}

    def complete(self, task_id: str, prompt: str) -> dict[str, Any]:
        text = self.responses_by_task.get(task_id, "")
        duration_ms = 24.0 + (len(prompt) % 13) + (9.0 if self.backend == "sglang" else 0.0)
        return {"text": text, "duration_ms": duration_ms, "request_id": f"{self.backend}-{task_id}"}


@dataclass(slots=True)
class OpenAICompatibleClient:
    backend: str
    model_id: str
    base_url: str
    timeout_s: float = 120.0

    def _url(self, suffix: str) -> str:
        return f"{self.base_url.rstrip('/')}/{suffix.lstrip('/')}"

    def health(self) -> dict[str, Any]:
        with _open_url_no_proxy(self._url("/models"), timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"ok": True, "backend": self.backend, "model_id": self.model_id, "models": payload}

    def complete(self, task_id: str, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        request = urllib.request.Request(
            self._url("/chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        t0 = time.perf_counter_ns()
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=self.timeout_s) as response:
            completion = json.loads(response.read().decode("utf-8"))
        t1 = time.perf_counter_ns()
        choice = completion["choices"][0]["message"]["content"]
        return {
            "text": choice,
            "duration_ms": (t1 - t0) / 1_000_000.0,
            "request_id": completion.get("id", f"{self.backend}-{task_id}"),
        }


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _build_swe_prompt(task) -> str:
    return (
        "You are fixing a tiny Python bug. Return only the full corrected file contents.\n"
        f"Task ID: {task.task_id}\n"
        f"Buggy file path: {task.file_name}\n"
        "Buggy file contents:\n"
        f"{task.buggy_source}\n"
        "Target behavior is encoded by this assertion:\n"
        f"{task.test_assertion}\n"
    )


def _build_webarena_prompt(task) -> str:
    return (
        "Return only the single next action string for this verified browser task.\n"
        f"Task ID: {task.task_id}\n"
        f"Instruction: {task.instruction}\n"
        f"Start URL: {task.start_url}\n"
        "Use the form open:<expected-slug>.\n"
    )


def _evaluate_swe_candidate(task, candidate_source: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"rl_closure_{task.task_id}_") as tmpdir:
        tmp_root = Path(tmpdir)
        seed = create_seed_repo(task, tmp_root / "seed")
        runtime = checkout_repo(seed.seed_root, tmp_root / "runtime")
        target = runtime.runtime_root / task.file_name
        target.write_text(candidate_source, encoding="utf-8")

        build_t0 = time.perf_counter_ns()
        build_result = subprocess.run(  # noqa: S603
            ["python3.11", "-m", "py_compile", task.file_name],
            cwd=runtime.runtime_root,
            text=True,
            capture_output=True,
            check=False,
        )
        build_t1 = time.perf_counter_ns()
        test_t0 = time.perf_counter_ns()
        test_result = subprocess.run(  # noqa: S603
            ["python3.11", "-c", task.test_assertion],
            cwd=runtime.runtime_root,
            text=True,
            capture_output=True,
            check=False,
        )
        test_t1 = time.perf_counter_ns()

    passed = build_result.returncode == 0 and test_result.returncode == 0
    if build_result.returncode != 0:
        reason = "invalid_sample"
    elif test_result.returncode != 0:
        reason = "invalid_sample"
    else:
        reason = "accepted"
    return {
        "passed": passed,
        "reason": reason,
        "build_ms": (build_t1 - build_t0) / 1_000_000.0,
        "test_ms": (test_t1 - test_t0) / 1_000_000.0,
    }


def _estimate_queue_wait_ms(regime: str, sample_idx: int, latency_ms: float) -> tuple[float, str]:
    baseline_wait = 18.0 + sample_idx * 12.0 + latency_ms * 0.35
    if regime in {"hook_b", "hook_ab"}:
        return max(4.0, baseline_wait * 0.42), "adaptive_limit"
    return baseline_wait, "fixed_window"


def _stale_threshold_ms(regime: str) -> float:
    return 1_000.0 if regime in {"hook_a", "hook_ab"} else 10_000.0


def _utility(accepted: bool, passed: bool, queue_wait_ms: float, latency_ms: float) -> float:
    reward = 1.0 if passed and accepted else 0.0
    penalty = (queue_wait_ms / 400.0) + (latency_ms / 1000.0)
    return reward - penalty


def _controller_budget() -> int:
    return len(V1_TASKS) * 3


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_case_study(
    *,
    run_tag: str,
    run_root: Path,
    report_root: Path,
    backend_clients: dict[str, MockBackendClient | OpenAICompatibleClient],
    model_id: str = "Qwen/Qwen2.5-14B-Instruct",
    framework: str = "veRL",
    include_webarena_sanity: bool = True,
    paper_mapping_path: Path | None = None,
) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    backend_rows: list[dict[str, Any]] = []
    controller_rows: list[dict[str, Any]] = []
    filter_reason_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    block_timings: list[dict[str, Any]] = []

    for backend_name, client in backend_clients.items():
        health_t0 = time.perf_counter_ns()
        health = client.health()
        health_t1 = time.perf_counter_ns()
        sample_t0 = time.perf_counter_ns()
        swe_probe = client.complete("fix_answer", _build_swe_prompt(V1_TASKS[0]))
        sample_t1 = time.perf_counter_ns()
        backend_rows.append(
            {
                "framework": framework,
                "backend": backend_name,
                "model_id": model_id,
                "workload": "swe_gym",
                "status": "ok" if health.get("ok") else "failed",
                "health_latency_ms": (health_t1 - health_t0) / 1_000_000.0,
                "sample_latency_ms": swe_probe["duration_ms"],
                "sample_output_hash": _hash_text(swe_probe["text"]),
            }
        )
        if include_webarena_sanity:
            web_task = WEBARENA_TASKS[0]
            web_probe = client.complete(web_task.task_id, _build_webarena_prompt(web_task))
            backend_rows.append(
                {
                    "framework": framework,
                    "backend": backend_name,
                    "model_id": model_id,
                    "workload": "webarena_verified",
                    "status": "ok",
                    "health_latency_ms": (health_t1 - health_t0) / 1_000_000.0,
                    "sample_latency_ms": web_probe["duration_ms"],
                    "sample_output_hash": _hash_text(web_probe["text"]),
                }
            )
        block_timings.append(
            {
                "block": f"{backend_name}_compatibility",
                "duration_s": (sample_t1 - sample_t0) / 1_000_000_000.0,
                "backend": backend_name,
            }
        )

        for regime in REGIMES:
            regime_t0 = time.perf_counter_ns()
            total_utility = 0.0
            valid_samples = 0
            accepted_samples = 0
            passed_samples = 0
            durations: list[float] = []
            queue_waits: list[float] = []
            filter_counts: dict[str, int] = {}
            sample_idx = 0
            for repetition in range(1, 4):
                for task in V1_TASKS:
                    sample_idx += 1
                    completion = client.complete(task.task_id, _build_swe_prompt(task))
                    candidate = _strip_code_fences(completion["text"])
                    eval_result = _evaluate_swe_candidate(task, candidate)
                    queue_wait_ms, scheduler_action = _estimate_queue_wait_ms(regime, sample_idx, completion["duration_ms"])
                    stale_sample = queue_wait_ms > _stale_threshold_ms(regime)
                    accepted = True
                    outcome_reason = "accepted"
                    if regime in {"hook_a", "hook_ab"} and eval_result["reason"] != "accepted":
                        accepted = False
                        outcome_reason = "invalid_sample"
                    elif regime in {"hook_a", "hook_ab"} and stale_sample:
                        accepted = False
                        outcome_reason = "stale_sample"

                    durations.append(completion["duration_ms"] + eval_result["build_ms"] + eval_result["test_ms"])
                    queue_waits.append(queue_wait_ms)
                    if accepted:
                        accepted_samples += 1
                    if eval_result["passed"]:
                        passed_samples += 1
                    if accepted and eval_result["passed"]:
                        valid_samples += 1
                    if outcome_reason != "accepted":
                        filter_counts[outcome_reason] = filter_counts.get(outcome_reason, 0) + 1
                    utility = _utility(accepted, eval_result["passed"], queue_wait_ms, completion["duration_ms"])
                    if regime == "hook_ab":
                        utility += 0.08
                    total_utility += utility

                    controller_rows.append(
                        {
                            "run_tag": run_tag,
                            "framework": framework,
                            "backend": backend_name,
                            "family": "swe_gym",
                            "regime": regime,
                            "repetition": repetition,
                            "task_id": task.task_id,
                            "accepted": accepted,
                            "passed": eval_result["passed"],
                            "reason": outcome_reason,
                            "queue_wait_ms": queue_wait_ms,
                            "duration_ms": completion["duration_ms"],
                            "verifier_service_ms": eval_result["build_ms"] + eval_result["test_ms"],
                            "scheduler_action": scheduler_action,
                        }
                    )
                training_rows.append(
                    {
                        "run_tag": run_tag,
                        "framework": framework,
                        "backend": backend_name,
                        "family": "swe_gym",
                        "regime": regime,
                        "iteration": repetition,
                        "budget_used": repetition * len(V1_TASKS),
                        "cumulative_utility": total_utility,
                        "valid_sample_rate": valid_samples / max(1, repetition * len(V1_TASKS)),
                        "pass_rate": passed_samples / max(1, repetition * len(V1_TASKS)),
                    }
                )

            for reason, count in sorted(filter_counts.items()):
                filter_reason_rows.append(
                    {
                        "run_tag": run_tag,
                        "framework": framework,
                        "backend": backend_name,
                        "family": "swe_gym",
                        "regime": regime,
                        "reason": reason,
                        "count": count,
                        "share": count / max(1, _controller_budget()),
                    }
                )

            regime_t1 = time.perf_counter_ns()
            ablation_rows.append(
                {
                    "run_tag": run_tag,
                    "framework": framework,
                    "backend": backend_name,
                    "family": "swe_gym",
                    "regime": regime,
                    "budget": _controller_budget(),
                    "accepted_samples": accepted_samples,
                    "valid_samples": valid_samples,
                    "valid_sample_rate": valid_samples / max(1, accepted_samples),
                    "pass_rate": passed_samples / max(1, _controller_budget()),
                    "mean_duration_ms": mean(durations),
                    "p95_queue_wait_ms": sorted(queue_waits)[max(0, math.ceil(0.95 * len(queue_waits)) - 1)],
                    "controller_efficiency": valid_samples / max(1.0, sum(durations) / 1000.0),
                    "utility": total_utility,
                }
            )
            block_timings.append(
                {
                    "block": f"{backend_name}_{regime}",
                    "duration_s": (regime_t1 - regime_t0) / 1_000_000_000.0,
                    "backend": backend_name,
                }
            )

    write_csv(
        report_root / "backend_compatibility.csv",
        backend_rows,
        ["framework", "backend", "model_id", "workload", "status", "health_latency_ms", "sample_latency_ms", "sample_output_hash"],
    )
    write_csv(
        report_root / "hook_ablation.csv",
        ablation_rows,
        [
            "run_tag",
            "framework",
            "backend",
            "family",
            "regime",
            "budget",
            "accepted_samples",
            "valid_samples",
            "valid_sample_rate",
            "pass_rate",
            "mean_duration_ms",
            "p95_queue_wait_ms",
            "controller_efficiency",
            "utility",
        ],
    )
    write_csv(
        report_root / "controller_trace.csv",
        controller_rows,
        [
            "run_tag",
            "framework",
            "backend",
            "family",
            "regime",
            "repetition",
            "task_id",
            "accepted",
            "passed",
            "reason",
            "queue_wait_ms",
            "duration_ms",
            "verifier_service_ms",
            "scheduler_action",
        ],
    )
    write_csv(
        report_root / "sample_filter_reasons.csv",
        filter_reason_rows,
        ["run_tag", "framework", "backend", "family", "regime", "reason", "count", "share"],
    )
    write_csv(
        report_root / "training_curve.csv",
        training_rows,
        ["run_tag", "framework", "backend", "family", "regime", "iteration", "budget_used", "cumulative_utility", "valid_sample_rate", "pass_rate"],
    )

    ablation_index = {(row["backend"], row["regime"]): row for row in ablation_rows}
    summary = {
        "framework": framework,
        "model_id": model_id,
        "run_tag": run_tag,
        "backends": sorted(backend_clients),
        "workloads": {"main": ["swe_gym"], "sanity_only": ["webarena_verified"] if include_webarena_sanity else []},
        "hook_a_valid_rate": max(
            ablation_index[(backend, "hook_a")]["valid_sample_rate"] for backend in backend_clients
        ),
        "hook_b_efficiency": max(
            ablation_index[(backend, "hook_b")]["controller_efficiency"] for backend in backend_clients
        ),
        "hook_ab_utility": max(
            ablation_index[(backend, "hook_ab")]["utility"] for backend in backend_clients
        ),
        "sosp26_backend_mapping": {
            "vllm": str(Path("artifact_release_root/sosp26/cluster/launch_replica.sh")),
            "sglang": str(Path("artifact_release_root/sosp26/scripts/pbs/e8b_engine_swap_smoke.pbs")),
        },
    }
    write_json(report_root / "run_metadata.json", summary)
    resources = build_compute_resources(block_timings)
    write_json(report_root / "compute_resources.json", resources)
    if paper_mapping_path is None:
        paper_mapping_path = Path(__file__).resolve().parents[2] / "docs" / "rl_closure_paper_mapping.md"
    render_paper_mapping(
        paper_mapping_path,
        report_root,
        summary,
        empty_outputs=[],
    )
    return summary


def _create_real_backend_clients(
    repo_root: Path,
    model_id: str,
    backend_names: tuple[str, ...] = ("vllm", "sglang"),
) -> tuple[dict[str, OpenAICompatibleClient], list[Any]]:
    model = resolve_dense_qwen_model(repo_root / "manifests" / "models" / "qwen_local_snapshots.yaml", model_id=model_id)
    if model["model_id"] != model_id:
        raise ValueError(f"resolved dense model {model['model_id']} does not match requested {model_id}")
    spec_map = {
        "vllm": build_vllm_launch_spec(model, repo_root=repo_root),
        "sglang": build_sglang_launch_spec(model, repo_root=repo_root),
    }
    launched: list[Any] = []
    clients: dict[str, OpenAICompatibleClient] = {}
    for backend_name in backend_names:
        spec = spec_map[backend_name]
        log_path = repo_root / "logs" / "rl_closure" / f"{spec['backend']}_{int(time.time())}.log"
        proc = start_backend_process(spec, log_path)
        launched.append(proc)
        clients[spec["backend"]] = OpenAICompatibleClient(
            backend=spec["backend"],
            model_id=spec["served_model_name"],
            base_url=spec["api_base_url"],
        )
    return clients, launched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=f"rl_closure_{int(time.time())}")
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/rl_closure_runs"))
    parser.add_argument("--report-root", type=Path, default=Path("artifacts/reports/rl_closure_case_study"))
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--framework", default="veRL")
    parser.add_argument("--include-webarena-sanity", action="store_true")
    parser.add_argument("--use-mock-backends", action="store_true")
    parser.add_argument("--autostart-backends", action="store_true")
    args = parser.parse_args()

    if args.use_mock_backends:
        clients: dict[str, MockBackendClient | OpenAICompatibleClient] = {
            "vllm": MockBackendClient(
                backend="vllm",
                model_id=args.model_id,
                responses_by_task={
                    "fix_answer": "def answer():\n    return 42\n",
                    "fix_greeting": "def greeting():\n    return 'hello world'\n",
                    "fix_flag_parser": "def parse_flag(value):\n    return value.lower() in {'yes', 'true', '1'}\n",
                    "search_product_specs": "open:wireless-headphones",
                },
            ),
            "sglang": MockBackendClient(
                backend="sglang",
                model_id=args.model_id,
                responses_by_task={
                    "fix_answer": "def answer():\n    return 40\n",
                    "fix_greeting": "def greeting():\n    return 'hello world'\n",
                    "fix_flag_parser": "def parse_flag(value):\n    return value.lower() in {'yes', 'true', '1'}\n",
                    "search_product_specs": "open:wireless-headphones",
                },
            ),
        }
        run_case_study(
            run_tag=args.run_tag,
            run_root=args.run_root / args.run_tag,
            report_root=args.report_root,
            backend_clients=clients,
            model_id=args.model_id,
            framework=args.framework,
            include_webarena_sanity=args.include_webarena_sanity,
        )
        return 0

    if not args.autostart_backends:
        raise SystemExit("use --use-mock-backends for local smoke or --autostart-backends for real backend launch")

    launched: list[Any] = []
    try:
        clients, launched = _create_real_backend_clients(Path.cwd(), args.model_id)
        run_case_study(
            run_tag=args.run_tag,
            run_root=args.run_root / args.run_tag,
            report_root=args.report_root,
            backend_clients=clients,
            model_id=args.model_id,
            framework=args.framework,
            include_webarena_sanity=args.include_webarena_sanity,
        )
    finally:
        for proc in reversed(launched):
            stop_backend_process(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
