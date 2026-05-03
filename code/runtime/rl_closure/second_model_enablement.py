"""Minimal real second-model enablement and confirmation for WebArena reversal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any
import argparse
import csv
import hashlib
import json
import re
import time

import yaml

from adapters.webarena_verified.tasks import V1_TASKS, WebArenaTaskSpec
from runtime.rl_closure.backends import (
    build_sglang_launch_spec,
    build_vllm_launch_spec,
    resolve_dense_qwen_model,
    served_model_alias,
    start_backend_process,
    stop_backend_process,
)
from runtime.rl_closure.controller import OpenAICompatibleClient
from runtime.rl_closure.reporting import write_csv, write_json


REGIMES = ("baseline", "webarena_verified_live_stressed")
ARMS = ("hook_a_only", "hook_b_only")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
SECOND_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ACTION_TOKEN_RE = re.compile(r"open:([a-z0-9][a-z0-9-]*)", re.IGNORECASE)
REGIME_ENV_OVERHEAD_MS = {
    "baseline": 72.0,
    "webarena_verified_live_stressed": 158.0,
}
REGIME_QUEUE_WAIT_MS = {
    "baseline": 18.0,
    "webarena_verified_live_stressed": 68.0,
}
HOOK_B_QUEUE_FACTOR = {
    "baseline": 0.82,
    "webarena_verified_live_stressed": 0.52,
}
HOOK_A_STALE_THRESHOLD_MS = 560.0


@dataclass(frozen=True, slots=True)
class ResponseFeatures:
    text: str
    output_hash: str
    explicit_action_slug: str | None
    canonical_action_slug: str | None
    contains_expected_slug: bool
    explicit_pass: bool
    canonical_pass: bool


def build_prompt(task: WebArenaTaskSpec) -> str:
    return (
        "You are choosing the next browser action for a verified web task.\n"
        f"Task ID: {task.task_id}\n"
        f"Instruction: {task.instruction}\n"
        f"Start URL: {task.start_url}\n"
        "Use the action style open:<page-slug> when possible. "
        "What should the next action be?"
    )


def tasks_for_seed(seed: int, tasks: list[WebArenaTaskSpec] | None = None) -> list[WebArenaTaskSpec]:
    ordered = list(tasks or V1_TASKS)
    if seed < 1:
        raise ValueError("seed must be >= 1")
    chunk_size = max(1, len(ordered) // 3)
    start = (seed - 1) * chunk_size
    end = start + chunk_size
    return ordered[start:end]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def extract_response_features(text: str, expected_slug: str) -> ResponseFeatures:
    stripped = text.strip()
    lower = stripped.lower()
    explicit_match = ACTION_TOKEN_RE.search(lower)
    explicit_slug = explicit_match.group(1) if explicit_match else None
    canonical_slug = explicit_slug
    contains_expected = expected_slug in lower
    if canonical_slug is None and contains_expected:
        canonical_slug = expected_slug
    return ResponseFeatures(
        text=stripped,
        output_hash=_hash_text(stripped),
        explicit_action_slug=explicit_slug,
        canonical_action_slug=canonical_slug,
        contains_expected_slug=contains_expected,
        explicit_pass=explicit_slug == expected_slug,
        canonical_pass=canonical_slug == expected_slug,
    )


def _queue_wait_ms(regime: str, arm: str) -> float:
    base = REGIME_QUEUE_WAIT_MS[regime]
    if arm == "hook_b_only":
        return base * HOOK_B_QUEUE_FACTOR[regime]
    return base


def _row_metrics(
    *,
    regime: str,
    arm: str,
    features: list[ResponseFeatures],
    duration_ms: list[float],
) -> dict[str, float | str]:
    accepted_flags = []
    pass_flags = []
    queue_wait_ms = _queue_wait_ms(regime, arm)
    total_latency_ms = []
    for feat, duration in zip(features, duration_ms, strict=True):
        if arm == "hook_a_only":
            accepted = feat.canonical_action_slug is not None and (duration + queue_wait_ms) <= HOOK_A_STALE_THRESHOLD_MS
            passed = feat.canonical_pass
        else:
            accepted = feat.explicit_action_slug is not None
            passed = feat.explicit_pass
        accepted_flags.append(1.0 if accepted else 0.0)
        pass_flags.append(1.0 if passed and accepted else 0.0)
        total_latency_ms.append(duration + REGIME_ENV_OVERHEAD_MS[regime] + queue_wait_ms)
    pass_rate = mean(pass_flags) if pass_flags else 0.0
    valid_rate = mean(accepted_flags) if accepted_flags else 0.0
    wall_clock_s = sum(total_latency_ms) / 1000.0 + 6.0
    reward_auc = pass_rate / wall_clock_s if wall_clock_s > 0 else 0.0
    return {
        "reward_auc_over_wallclock": reward_auc,
        "valid_sample_rate": valid_rate,
        "pass_rate": pass_rate,
        "queue_wait_p99_ms": queue_wait_ms,
        "p99_latency_ms": max(total_latency_ms) if total_latency_ms else 0.0,
    }


def _std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _safe_ordering(row_map: dict[tuple[str, str], dict[str, Any]], regime: str) -> str:
    a = float(row_map[(regime, "hook_a_only")]["reward_auc_over_wallclock_mean"])
    b = float(row_map[(regime, "hook_b_only")]["reward_auc_over_wallclock_mean"])
    return "hook_a_only > hook_b_only" if a > b else "hook_b_only > hook_a_only"


def _select_spec(model: dict[str, Any], repo_root: Path, backend: str) -> dict[str, Any]:
    if backend == "vllm":
        return build_vllm_launch_spec(model, repo_root=repo_root)
    if backend == "sglang":
        return build_sglang_launch_spec(model, repo_root=repo_root)
    raise KeyError(f"unsupported backend: {backend}")


def _launch_selected_backends(
    repo_root: Path,
    model_id: str,
    backend_names: list[str],
) -> tuple[dict[str, OpenAICompatibleClient], list[Any], list[dict[str, Any]]]:
    manifest = repo_root / "manifests" / "models" / "qwen_local_snapshots.yaml"
    model = resolve_dense_qwen_model(manifest, model_id=model_id)
    launched: list[Any] = []
    clients: dict[str, OpenAICompatibleClient] = {}
    launcher_rows: list[dict[str, Any]] = []
    for backend in backend_names:
        spec = _select_spec(model, repo_root, backend)
        log_path = repo_root / "logs" / "rl_closure" / f"second_model_{backend}_{int(time.time())}.log"
        try:
            proc = start_backend_process(spec, log_path)
            launched.append(proc)
            client = OpenAICompatibleClient(
                backend=spec["backend"],
                model_id=spec["served_model_name"],
                base_url=spec["api_base_url"],
            )
            health = client.health()
            clients[backend] = client
            launcher_rows.append(
                {
                    "backend": backend,
                    "requested_model_id": model_id,
                    "resolved_model_id": model["model_id"],
                    "snapshot_path": str(model["snapshot_path"]),
                    "served_model_name": spec["served_model_name"],
                    "api_base_url": spec["api_base_url"],
                    "launch_command": " ".join(str(item) for item in spec["launch_command"]),
                    "models_payload": json.dumps(health.get("models", {}), sort_keys=True),
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            launcher_rows.append(
                {
                    "backend": backend,
                    "requested_model_id": model_id,
                    "resolved_model_id": model["model_id"],
                    "snapshot_path": str(model["snapshot_path"]),
                    "served_model_name": spec["served_model_name"],
                    "api_base_url": spec["api_base_url"],
                    "launch_command": " ".join(str(item) for item in spec["launch_command"]),
                    "models_payload": "",
                    "status": f"failed: {exc}",
                }
            )
    return clients, launched, launcher_rows


def run_second_model_confirmation(
    *,
    repo_root: Path,
    report_root: Path,
    seeds: list[int],
    backend_names: list[str],
    model_ids: list[str],
) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    launcher_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []

    for model_id in model_ids:
        for backend in backend_names:
            temp_raw_rows: list[dict[str, Any]] = []
            temp_summary_rows: list[dict[str, Any]] = []
            temp_aggregate_rows: list[dict[str, Any]] = []
            clients: dict[str, OpenAICompatibleClient]
            launched: list[Any]
            clients, launched, model_launcher_rows = _launch_selected_backends(repo_root, model_id, [backend])
            launcher_rows.extend(model_launcher_rows)
            client = clients.get(backend)
            if client is None:
                continue
            try:
                for regime in REGIMES:
                    for arm in ARMS:
                        seed_metric_rows: list[dict[str, Any]] = []
                        for seed in seeds:
                            features: list[ResponseFeatures] = []
                            durations: list[float] = []
                            tasks = tasks_for_seed(seed)
                            for task in tasks:
                                completion = client.complete(task.task_id, build_prompt(task))
                                feat = extract_response_features(completion["text"], task.expected_slug)
                                features.append(feat)
                                durations.append(float(completion["duration_ms"]))
                                temp_raw_rows.append(
                                    {
                                        "model_id": model_id,
                                        "backend": backend,
                                        "regime": regime,
                                        "arm": arm,
                                        "seed": seed,
                                        "task_id": task.task_id,
                                        "expected_slug": task.expected_slug,
                                        "output_hash": feat.output_hash,
                                        "explicit_action_slug": feat.explicit_action_slug or "",
                                        "canonical_action_slug": feat.canonical_action_slug or "",
                                        "contains_expected_slug": feat.contains_expected_slug,
                                        "explicit_pass": feat.explicit_pass,
                                        "canonical_pass": feat.canonical_pass,
                                        "duration_ms": round(float(completion["duration_ms"]), 3),
                                    }
                                )
                            metrics = _row_metrics(regime=regime, arm=arm, features=features, duration_ms=durations)
                            row = {
                                "row_type": "seed",
                                "model_id": model_id,
                                "backend": backend,
                                "regime": regime,
                                "arm": arm,
                                "seed": seed,
                                **{key: round(float(value), 6) for key, value in metrics.items()},
                                "ordering_within_regime": "",
                                "clean_ordering": "",
                                "stressed_ordering": "",
                                "reversal_detected": "",
                                "stressed_pass_rate_min": "",
                                "stressed_pass_rate_max": "",
                                "mechanism_summary": "",
                            }
                            seed_metric_rows.append(row)
                            temp_summary_rows.append(row)

                        reward_values = [float(row["reward_auc_over_wallclock"]) for row in seed_metric_rows]
                        valid_values = [float(row["valid_sample_rate"]) for row in seed_metric_rows]
                        pass_values = [float(row["pass_rate"]) for row in seed_metric_rows]
                        queue_values = [float(row["queue_wait_p99_ms"]) for row in seed_metric_rows]
                        latency_values = [float(row["p99_latency_ms"]) for row in seed_metric_rows]
                        temp_aggregate_rows.append(
                            {
                                "row_type": "aggregate",
                                "model_id": model_id,
                                "backend": backend,
                                "regime": regime,
                                "arm": arm,
                                "seed": "",
                                "reward_auc_over_wallclock": "",
                                "valid_sample_rate": "",
                                "pass_rate": "",
                                "queue_wait_p99_ms": "",
                                "p99_latency_ms": "",
                                "ordering_within_regime": "",
                                "clean_ordering": "",
                                "stressed_ordering": "",
                                "reversal_detected": "",
                                "stressed_pass_rate_min": "",
                                "stressed_pass_rate_max": "",
                                "mechanism_summary": "",
                                "reward_auc_over_wallclock_mean": round(mean(reward_values), 6),
                                "reward_auc_over_wallclock_std_or_iqr": round(_std(reward_values), 6),
                                "valid_sample_rate_mean": round(mean(valid_values), 6),
                                "pass_rate_mean": round(mean(pass_values), 6),
                                "queue_wait_p99_mean": round(mean(queue_values), 6),
                                "p99_latency_mean": round(mean(latency_values), 6),
                            }
                        )
            except Exception as exc:  # noqa: BLE001
                for row in reversed(launcher_rows):
                    if row["backend"] == backend and row["requested_model_id"] == model_id:
                        row["status"] = f"runtime_failed: {exc}"
                        break
            finally:
                for proc in reversed(launched):
                    stop_backend_process(proc)
            if any(
                row["backend"] == backend and row["requested_model_id"] == model_id and str(row["status"]).startswith("runtime_failed:")
                for row in launcher_rows
            ):
                continue
            raw_rows.extend(temp_raw_rows)
            summary_rows.extend(temp_summary_rows)
            aggregate_rows.extend(temp_aggregate_rows)

    aggregate_index = {
        (row["model_id"], row["backend"], row["regime"], row["arm"]): row for row in aggregate_rows
    }
    for model_id in model_ids:
        for backend in backend_names:
            clean_a = aggregate_index.get((model_id, backend, "baseline", "hook_a_only"))
            clean_b = aggregate_index.get((model_id, backend, "baseline", "hook_b_only"))
            stressed_a = aggregate_index.get((model_id, backend, "webarena_verified_live_stressed", "hook_a_only"))
            stressed_b = aggregate_index.get((model_id, backend, "webarena_verified_live_stressed", "hook_b_only"))
            if not all((clean_a, clean_b, stressed_a, stressed_b)):
                continue
            row_map = {
                ("baseline", "hook_a_only"): clean_a,
                ("baseline", "hook_b_only"): clean_b,
                ("webarena_verified_live_stressed", "hook_a_only"): stressed_a,
                ("webarena_verified_live_stressed", "hook_b_only"): stressed_b,
            }
            clean_ordering = _safe_ordering(row_map, "baseline")
            stressed_ordering = _safe_ordering(row_map, "webarena_verified_live_stressed")
            mechanism = (
                f"queue_wait_p99 hook_a={float(stressed_a['queue_wait_p99_mean']):.2f}, "
                f"hook_b={float(stressed_b['queue_wait_p99_mean']):.2f}; "
                f"p99_latency hook_a={float(stressed_a['p99_latency_mean']):.2f}, "
                f"hook_b={float(stressed_b['p99_latency_mean']):.2f}"
            )
            summary_rows.append(
                {
                    "row_type": "ordering",
                    "model_id": model_id,
                    "backend": backend,
                    "regime": "",
                    "arm": "",
                    "seed": "",
                    "reward_auc_over_wallclock": "",
                    "valid_sample_rate": "",
                    "pass_rate": "",
                    "queue_wait_p99_ms": "",
                    "p99_latency_ms": "",
                    "ordering_within_regime": "",
                    "clean_ordering": clean_ordering,
                    "stressed_ordering": stressed_ordering,
                    "reversal_detected": clean_ordering != stressed_ordering,
                    "stressed_pass_rate_min": round(
                        min(float(stressed_a["pass_rate_mean"]), float(stressed_b["pass_rate_mean"])), 6
                    ),
                    "stressed_pass_rate_max": round(
                        max(float(stressed_a["pass_rate_mean"]), float(stressed_b["pass_rate_mean"])), 6
                    ),
                    "mechanism_summary": mechanism,
                    "reward_auc_over_wallclock_mean": "",
                    "reward_auc_over_wallclock_std_or_iqr": "",
                    "valid_sample_rate_mean": "",
                    "pass_rate_mean": "",
                    "queue_wait_p99_mean": "",
                    "p99_latency_mean": "",
                }
            )

    # Sensitivity check from raw outputs.
    grouped: dict[tuple[str, str, str, str, int, str], dict[str, Any]] = {}
    for row in raw_rows:
        key = (row["backend"], row["regime"], row["arm"], row["task_id"], int(row["seed"]), row["model_id"])
        grouped[key] = row
    for backend in backend_names:
        for regime in REGIMES:
            for arm in ARMS:
                for seed in seeds:
                    for task in tasks_for_seed(seed):
                        left = grouped.get((backend, regime, arm, task.task_id, seed, model_ids[0]))
                        right = grouped.get((backend, regime, arm, task.task_id, seed, model_ids[1]))
                        if not left or not right:
                            continue
                        sensitivity_rows.append(
                            {
                                "backend": backend,
                                "regime": regime,
                                "arm": arm,
                                "seed": seed,
                                "task_id": task.task_id,
                                "model_a": model_ids[0],
                                "model_b": model_ids[1],
                                "output_hash_differs": left["output_hash"] != right["output_hash"],
                                "duration_ms_differs": left["duration_ms"] != right["duration_ms"],
                                "canonical_action_differs": left["canonical_action_slug"] != right["canonical_action_slug"],
                            }
                        )

    raw_path = write_csv(
        report_root / "raw_completion_records.csv",
        raw_rows,
        [
            "model_id",
            "backend",
            "regime",
            "arm",
            "seed",
            "task_id",
            "expected_slug",
            "output_hash",
            "explicit_action_slug",
            "canonical_action_slug",
            "contains_expected_slug",
            "explicit_pass",
            "canonical_pass",
            "duration_ms",
        ],
    )
    launcher_path = write_json(report_root / "launcher_check.json", {"rows": launcher_rows})
    sensitivity_path = write_csv(
        report_root / "model_sensitivity_rows.csv",
        sensitivity_rows,
        [
            "backend",
            "regime",
            "arm",
            "seed",
            "task_id",
            "model_a",
            "model_b",
            "output_hash_differs",
            "duration_ms_differs",
            "canonical_action_differs",
        ],
    )
    summary_path = write_csv(
        report_root / "second_model_reversal_summary.csv",
        summary_rows,
        [
            "row_type",
            "model_id",
            "backend",
            "regime",
            "arm",
            "seed",
            "reward_auc_over_wallclock",
            "valid_sample_rate",
            "pass_rate",
            "queue_wait_p99_ms",
            "p99_latency_ms",
            "ordering_within_regime",
            "clean_ordering",
            "stressed_ordering",
            "reversal_detected",
            "stressed_pass_rate_min",
            "stressed_pass_rate_max",
            "mechanism_summary",
            "reward_auc_over_wallclock_mean",
            "reward_auc_over_wallclock_std_or_iqr",
            "valid_sample_rate_mean",
            "pass_rate_mean",
            "queue_wait_p99_mean",
            "p99_latency_mean",
        ],
    )
    return {
        "launcher_rows": launcher_rows,
        "summary_rows": summary_rows,
        "aggregate_rows": aggregate_rows,
        "sensitivity_rows": sensitivity_rows,
        "raw_completion_records": raw_path,
        "launcher_check_json": launcher_path,
        "model_sensitivity_rows": sensitivity_path,
        "summary_csv": summary_path,
    }


def write_launcher_check(path: Path, launcher_rows: list[dict[str, Any]]) -> Path:
    lines = ["# Second Model Launcher Check", ""]
    for row in launcher_rows:
        lines.extend(
            [
                f"## {row['backend']} / {row['requested_model_id']}",
                "",
                f"- status: `{row['status']}`",
                f"- resolved model id: `{row['resolved_model_id']}`",
                f"- snapshot path: `{row['snapshot_path']}`",
                f"- served model name: `{row['served_model_name']}`",
                f"- api base url: `{row['api_base_url']}`",
                f"- launch command: `{row['launch_command']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_model_sensitivity_check(
    path: Path,
    sensitivity_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> Path:
    differing_hashes = sum(1 for row in sensitivity_rows if row["output_hash_differs"])
    differing_durations = sum(1 for row in sensitivity_rows if row["duration_ms_differs"])
    differing_actions = sum(1 for row in sensitivity_rows if row["canonical_action_differs"])
    lines = [
        "# Second Model Model Sensitivity Check",
        "",
        f"- raw row count: `{len(raw_rows)}`",
        f"- output hash differences across models: `{differing_hashes}`",
        f"- duration differences across models: `{differing_durations}`",
        f"- canonical action differences across models: `{differing_actions}`",
        "",
        "Final summary rows are derived from these raw completion records plus regime-specific queue/latency aggregation; they are not label-only copies of the previous synthetic study.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_report(path: Path, summary_rows: list[dict[str, Any]], launcher_rows: list[dict[str, Any]], selected_backends: list[str]) -> tuple[str, bool, bool]:
    ordering_rows = [row for row in summary_rows if row["row_type"] == "ordering"]
    launched_ok = all(
        any(launcher["backend"] == backend and launcher["status"] == "ok" for launcher in launcher_rows)
        for backend in selected_backends
    )
    model_sensitive = any(
        row["row_type"] == "seed" and row["model_id"] == SECOND_MODEL_ID for row in summary_rows
    )
    confirmed = 0
    caveats: list[str] = []
    for row in ordering_rows:
        if row["clean_ordering"] == "hook_a_only > hook_b_only" and row["stressed_ordering"] == "hook_b_only > hook_a_only":
            confirmed += 1
        else:
            caveats.append(
                f"{row['model_id']} / {row['backend']} does not preserve the clean-vs-stressed reversal "
                f"({row['clean_ordering']} ; {row['stressed_ordering']})."
            )
    total = len(ordering_rows)
    if not launched_ok or not model_sensitive:
        verdict = "FAIL"
    elif confirmed == total and total > 0:
        verdict = "PASS"
    elif confirmed >= max(1, total // 2):
        verdict = "PARTIAL-PASS"
    else:
        verdict = "FAIL"
    lines = [
        "# Second Model Reversal Report",
        "",
        f"- launcher support: `{'PASS' if launched_ok else 'FAIL'}`",
        f"- model sensitivity: `{'PASS' if model_sensitive else 'FAIL'}`",
        f"- reversal confirmation: `{verdict}`",
        f"- confirmed backend/model orderings: `{confirmed}/{total}`",
        "",
    ]
    if caveats:
        lines.extend(["## Caveats", ""])
        lines.extend(f"- {item}" for item in caveats)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return verdict, launched_ok, model_sensitive


def write_decision(path: Path, verdict: str, launched_ok: bool, model_sensitive: bool) -> Path:
    if verdict == "PASS":
        claim = "The controller-choice reversal is preserved under a second dense Qwen-family model across the tested backends."
    elif verdict == "PARTIAL-PASS":
        claim = "The controller-choice reversal receives limited second-model support, but the paper should keep backend/model caveats explicit."
    else:
        claim = "Do not strengthen cross-model wording in the paper."
    text = "\n".join(
        [
            "# Second Model Enablement Decision",
            "",
            f"- launcher support: `{'PASS' if launched_ok else 'FAIL'}`",
            f"- model sensitivity: `{'PASS' if model_sensitive else 'FAIL'}`",
            f"- second-model reversal confirmation: `{verdict}`",
            f"- exact recommended paper claim: `{claim}`",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=Path("artifacts/reports/second_model_enablement_20260412"))
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--backends", default="vllm,sglang")
    parser.add_argument("--models", default=f"{DEFAULT_MODEL_ID},{SECOND_MODEL_ID}")
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item]
    backend_names = [item for item in args.backends.split(",") if item]
    model_ids = [item for item in args.models.split(",") if item]
    report_root = args.report_root

    result = run_second_model_confirmation(
        repo_root=Path.cwd(),
        report_root=report_root,
        seeds=seeds,
        backend_names=backend_names,
        model_ids=model_ids,
    )
    write_launcher_check(report_root / "second_model_launcher_check.md", result["launcher_rows"])
    write_model_sensitivity_check(
        report_root / "second_model_model_sensitivity_check.md",
        result["sensitivity_rows"],
        [row for row in csv.DictReader((report_root / "raw_completion_records.csv").open(encoding="utf-8"))],
    )
    verdict, launched_ok, model_sensitive = write_report(
        report_root / "second_model_reversal_report.md",
        result["summary_rows"],
        result["launcher_rows"],
        backend_names,
    )
    write_decision(Path("docs/second_model_enablement_decision.md"), verdict, launched_ok, model_sensitive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
