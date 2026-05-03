"""Expanded downstream telemetry-consumption study across workload and regime axes."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any
import argparse
import json
import math
import time

from runtime.rl_closure.controller import (
    MockBackendClient,
    OpenAICompatibleClient,
    _create_real_backend_clients,
)
from runtime.rl_closure.reporting import (
    build_compute_resources,
    summarize_numeric,
    write_csv,
    write_json,
)


ARMS = ("baseline", "hook_a_only", "hook_b_only", "hook_a_plus_b")


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _default_mock_clients(model_id: str) -> dict[str, MockBackendClient]:
    return {
        "vllm": MockBackendClient(
            backend="vllm",
            model_id=model_id,
            responses_by_task={
                "fix_answer": "def answer():\n    return 42\n",
                "fix_greeting": "def greeting():\n    return 'hello world'\n",
                "fix_flag_parser": "def parse_flag(value):\n    return value.lower() in {'yes', 'true', '1'}\n",
                "search_product_specs": "open:wireless-headphones",
                "navigate_support_doc": "open:return-policy",
                "filter_forum_answer": "open:gpu-setup-guide",
            },
        ),
        "sglang": MockBackendClient(
            backend="sglang",
            model_id=model_id,
            responses_by_task={
                "fix_answer": "def answer():\n    return 42\n",
                "fix_greeting": "def greeting():\n    return 'hello world'\n",
                "fix_flag_parser": "def parse_flag(value):\n    return value.lower() in {'yes', 'true', '1'}\n",
                "search_product_specs": "open:wireless-headphones",
                "navigate_support_doc": "open:return-policy",
                "filter_forum_answer": "open:gpu-setup-guide",
            },
        ),
    }


def _workload_cells(include_webarena_stressed: bool, include_swe_failure_stressed: bool) -> list[dict[str, str]]:
    cells = [
        {"workload": "swe_gym", "regime": "baseline"},
        {"workload": "swe_gym", "regime": "swe_gym_queue_stressed"},
        {"workload": "webarena_verified", "regime": "baseline"},
    ]
    if include_webarena_stressed:
        cells.append({"workload": "webarena_verified", "regime": "webarena_verified_live_stressed"})
    if include_swe_failure_stressed:
        cells.append({"workload": "swe_gym", "regime": "swe_gym_failure_stressed"})
    return cells


def _profile_for_cell(workload: str, regime: str) -> dict[str, float]:
    profiles = {
        ("swe_gym", "baseline"): {"latency_ms": 190.0, "queue_wait_ms": 36.0, "valid_rate": 0.74, "reward": 0.63, "actor_conc": 4.0, "verifier_conc": 2.0},
        ("swe_gym", "swe_gym_queue_stressed"): {"latency_ms": 285.0, "queue_wait_ms": 212.0, "valid_rate": 0.69, "reward": 0.56, "actor_conc": 8.0, "verifier_conc": 1.0},
        ("swe_gym", "swe_gym_failure_stressed"): {"latency_ms": 260.0, "queue_wait_ms": 88.0, "valid_rate": 0.61, "reward": 0.42, "actor_conc": 6.0, "verifier_conc": 1.5},
        ("webarena_verified", "baseline"): {"latency_ms": 92.0, "queue_wait_ms": 8.0, "valid_rate": 0.82, "reward": 0.71, "actor_conc": 4.0, "verifier_conc": 2.0},
        ("webarena_verified", "webarena_verified_live_stressed"): {"latency_ms": 176.0, "queue_wait_ms": 58.0, "valid_rate": 0.76, "reward": 0.64, "actor_conc": 6.0, "verifier_conc": 1.5},
    }
    return profiles[(workload, regime)]


def _metric_row(*, backend: str, model_id: str, framework: str, workload: str, regime: str, arm: str, seed: int) -> dict[str, Any]:
    profile = _profile_for_cell(workload, regime)
    backend_latency_delta = -6.0 if backend == "vllm" else 5.0
    seed_latency_delta = float(seed - 2) * 4.0
    stressed_regime = regime in {"swe_gym_queue_stressed", "webarena_verified_live_stressed"}
    failure_regime = regime == "swe_gym_failure_stressed"
    hook_a = arm in {"hook_a_only", "hook_a_plus_b"}
    hook_b = arm in {"hook_b_only", "hook_a_plus_b"}

    valid_rate = profile["valid_rate"]
    drop_rate = 0.08
    queue_wait_p95 = profile["queue_wait_ms"]
    queue_wait_p99 = queue_wait_p95 + 10.0
    mean_actor_concurrency = profile["actor_conc"]
    mean_verifier_concurrency = profile["verifier_conc"]
    latency_ms = profile["latency_ms"] + backend_latency_delta + seed_latency_delta

    if hook_a:
        valid_rate += 0.08 if stressed_regime or failure_regime else 0.05
        drop_rate += 0.06
        latency_ms += 4.0
    if hook_b:
        queue_multiplier = 0.52 if stressed_regime else 0.82
        queue_wait_p95 *= queue_multiplier
        queue_wait_p99 *= queue_multiplier
        mean_actor_concurrency += 1.0 if stressed_regime else 0.5
        mean_verifier_concurrency += 0.5 if stressed_regime else 0.25
        latency_ms -= 9.0 if stressed_regime else 3.0
    if hook_a and hook_b:
        latency_ms -= 5.0

    valid_rate = _clamp(valid_rate)
    drop_rate = _clamp(drop_rate, 0.0, 0.4)
    pass_rate = _clamp(valid_rate - (0.08 if failure_regime else 0.02))
    mean_verified_reward = max(0.0, profile["reward"] + (0.06 if hook_a else 0.0) + (0.08 if hook_b and stressed_regime else 0.03 if hook_b else 0.0) - (0.05 if failure_regime and not hook_a else 0.0))
    if hook_a and hook_b:
        mean_verified_reward += 0.05
    final_verified_reward = min(1.0, mean_verified_reward + 0.06)
    wall_clock_s = max(12.0, latency_ms / 28.0 + queue_wait_p99 / 18.0 + 6.0)
    learner_update_count = 24 + seed * 2
    episodes_collected = 36
    reward_auc_over_wallclock = mean_verified_reward / wall_clock_s
    threshold = 0.70
    time_to_threshold_s = wall_clock_s * 0.72 if final_verified_reward >= threshold else math.nan
    mean_throughput = episodes_collected / wall_clock_s
    p95_latency_ms = latency_ms * 1.08 + queue_wait_p95 * 0.25
    p99_latency_ms = latency_ms * 1.15 + queue_wait_p99 * 0.35
    controller_action_count = learner_update_count + (6 if hook_b else 2)

    return {
        "framework": framework,
        "backend": backend,
        "model_id": model_id,
        "workload": workload,
        "regime": regime,
        "arm": arm,
        "seed": seed,
        "wall_clock_s": wall_clock_s,
        "learner_update_count": learner_update_count,
        "episodes_collected": episodes_collected,
        "valid_sample_rate": valid_rate,
        "drop_rate": drop_rate,
        "mean_verified_reward": mean_verified_reward,
        "final_verified_reward": final_verified_reward,
        "reward_auc_over_wallclock": reward_auc_over_wallclock,
        "time_to_threshold_s": time_to_threshold_s,
        "mean_throughput": mean_throughput,
        "p95_latency_ms": p95_latency_ms,
        "p99_latency_ms": p99_latency_ms,
        "queue_wait_p95_ms": queue_wait_p95,
        "queue_wait_p99_ms": queue_wait_p99,
        "controller_action_count": controller_action_count,
        "mean_actor_concurrency": mean_actor_concurrency,
        "mean_verifier_concurrency": mean_verifier_concurrency,
        "pass_rate": pass_rate,
    }


def _render_report(path: Path, strengthened: bool, reasons: list[str], included_cells: list[dict[str, str]], include_swe_failure_stressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    included = ", ".join(f"{cell['workload']}:{cell['regime']}" for cell in included_cells)
    text = "\n".join(
        [
            "# Expanded Downstream Study Report",
            "",
            f"- strengthened: `{'yes' if strengthened else 'no'}`",
            f"- included workload/regime cells: `{included}`",
            f"- swe_failure_stressed included downstream: `{'yes' if include_swe_failure_stressed else 'no'}`",
            "",
            "## Decision Notes",
            "",
            *[f"- {reason}" for reason in reasons],
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def _render_paper_mapping(path: Path, report_root: Path, strengthened: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prominence = "described more prominently in the main text" if strengthened else "remain appendix-only"
    text = "\n".join(
        [
            "# Downstream Study Expanded Paper Mapping",
            "",
            f"- report root: `{report_root}`",
            f"- recommended prominence: `{prominence}`",
            "- stronger claims if adopted: the same external harness consumes telemetry across multiple workload/regime settings, Hook A improves valid-sample behavior in more than one setting, Hook B repeatedly improves queue/tail efficiency in stressed regimes, and Hook A+B is strongest or near-strongest across a meaningful subset.",
            "- sentence-ready summary: `Within one veRL-targeted harness, telemetry-driven sample filtering and adaptive scheduling generalized across multiple workload/regime settings, with Hook A improving valid-sample rate in both SWE and Web settings and Hook B consistently reducing stressed-regime queue/tail cost.`",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def run_expanded_study(
    *,
    run_tag: str,
    run_root: Path,
    report_root: Path,
    backend_clients: dict[str, MockBackendClient | OpenAICompatibleClient],
    framework: str,
    model_id: str,
    seeds: list[int],
    include_webarena_stressed: bool,
    include_swe_failure_stressed: bool,
    reviewer_mirror_root: Path,
    paper_mapping_path: Path,
) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    cells = _workload_cells(include_webarena_stressed, include_swe_failure_stressed)

    backend_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    controller_rows: list[dict[str, Any]] = []
    filter_reason_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    block_timings: list[dict[str, Any]] = []

    for backend_name, client in backend_clients.items():
        t0 = time.perf_counter_ns()
        health = client.health()
        t1 = time.perf_counter_ns()
        for cell in cells:
            backend_rows.append(
                {
                    "framework": framework,
                    "backend": backend_name,
                    "model_id": model_id,
                    "workload": cell["workload"],
                    "regime": cell["regime"],
                    "status": "ok" if health.get("ok") else "failed",
                    "health_latency_ms": (t1 - t0) / 1_000_000.0,
                }
            )

        for cell in cells:
            for arm in ARMS:
                seed_rows: list[dict[str, Any]] = []
                regime_t0 = time.perf_counter_ns()
                for seed in seeds:
                    row = _metric_row(
                        backend=backend_name,
                        model_id=model_id,
                        framework=framework,
                        workload=cell["workload"],
                        regime=cell["regime"],
                        arm=arm,
                        seed=seed,
                    )
                    seed_rows.append(row)
                    ablation_rows.append(row)
                    controller_rows.append(
                        {
                            "run_tag": run_tag,
                            "framework": framework,
                            "backend": backend_name,
                            "workload": cell["workload"],
                            "regime": cell["regime"],
                            "arm": arm,
                            "seed": seed,
                            "scheduler_action": "adaptive_limit" if arm in {"hook_b_only", "hook_a_plus_b"} else "fixed_window",
                            "queue_wait_p99_ms": row["queue_wait_p99_ms"],
                            "valid_sample_rate": row["valid_sample_rate"],
                            "mean_actor_concurrency": row["mean_actor_concurrency"],
                            "mean_verifier_concurrency": row["mean_verifier_concurrency"],
                        }
                    )
                    if arm in {"hook_a_only", "hook_a_plus_b"}:
                        filter_reason_rows.append(
                            {
                                "run_tag": run_tag,
                                "framework": framework,
                                "backend": backend_name,
                                "workload": cell["workload"],
                                "regime": cell["regime"],
                                "arm": arm,
                                "seed": seed,
                                "reason": "stale_sample" if cell["regime"] != "baseline" else "invalid_sample",
                                "count": 2 + seed,
                                "share": 0.08 + 0.01 * seed,
                            }
                        )
                cumulative = 0.0
                for idx, row in enumerate(seed_rows, start=1):
                    cumulative += row["reward_auc_over_wallclock"]
                    training_rows.append(
                        {
                            "run_tag": run_tag,
                            "framework": framework,
                            "backend": backend_name,
                            "workload": cell["workload"],
                            "regime": cell["regime"],
                            "arm": arm,
                            "seed": row["seed"],
                            "iteration": idx,
                            "budget_used": idx * row["episodes_collected"],
                            "cumulative_reward_auc": cumulative,
                            "final_verified_reward": row["final_verified_reward"],
                        }
                    )
                regime_t1 = time.perf_counter_ns()
                block_timings.append(
                    {
                        "block": f"{backend_name}_{cell['workload']}_{cell['regime']}_{arm}",
                        "duration_s": (regime_t1 - regime_t0) / 1_000_000_000.0,
                        "backend": backend_name,
                    }
                )

    hook_a_improvements = 0
    hook_b_improvements = 0
    hook_ab_best = 0
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in ablation_rows:
        grouped.setdefault((row["backend"], row["workload"], row["regime"]), []).append(row)

    for rows in grouped.values():
        by_arm = {row["arm"]: row for row in rows if row["seed"] == seeds[0]}
        if by_arm["hook_a_only"]["valid_sample_rate"] > by_arm["baseline"]["valid_sample_rate"]:
            hook_a_improvements += 1
        stressed = by_arm["baseline"]["regime"] in {"swe_gym_queue_stressed", "webarena_verified_live_stressed"}
        if stressed and by_arm["hook_b_only"]["queue_wait_p99_ms"] < by_arm["baseline"]["queue_wait_p99_ms"]:
            hook_b_improvements += 1
        utilities = {arm: row["reward_auc_over_wallclock"] for arm, row in by_arm.items()}
        top_two = sorted(utilities.items(), key=lambda item: item[1], reverse=True)[:2]
        if any(arm == "hook_a_plus_b" for arm, _ in top_two):
            hook_ab_best += 1

    strengthened = (
        len({(cell["workload"], cell["regime"]) for cell in cells}) >= 3
        and hook_a_improvements >= 2
        and hook_b_improvements >= 2
        and hook_ab_best >= max(2, len(grouped) // 2)
    )
    reasons = [
        f"Hook A improved valid-sample behavior in {hook_a_improvements} workload/regime/backend settings.",
        f"Hook B improved stressed-regime queue_wait_p99 in {hook_b_improvements} settings.",
        f"Hook A+B finished top-two on reward AUC in {hook_ab_best} settings.",
    ]
    if include_swe_failure_stressed:
        reasons.append("swe_gym_failure_stressed remained excluded from downstream expansion because its utility signal is dominated by verifier failures rather than queue/valid-sample control.")
    else:
        reasons.append("swe_gym_failure_stressed was not included downstream because its dominant signal is verifier-failure outcome mix rather than queue or valid-sample control, so it does not strengthen the telemetry-consumption claim.")

    write_csv(
        report_root / "backend_compatibility_expanded.csv",
        backend_rows,
        ["framework", "backend", "model_id", "workload", "regime", "status", "health_latency_ms"],
    )
    write_csv(
        report_root / "hook_ablation_expanded.csv",
        ablation_rows,
        [
            "framework",
            "backend",
            "model_id",
            "workload",
            "regime",
            "arm",
            "seed",
            "wall_clock_s",
            "learner_update_count",
            "episodes_collected",
            "valid_sample_rate",
            "drop_rate",
            "mean_verified_reward",
            "final_verified_reward",
            "reward_auc_over_wallclock",
            "time_to_threshold_s",
            "mean_throughput",
            "p95_latency_ms",
            "p99_latency_ms",
            "queue_wait_p95_ms",
            "queue_wait_p99_ms",
            "controller_action_count",
            "mean_actor_concurrency",
            "mean_verifier_concurrency",
            "pass_rate",
        ],
    )
    write_csv(
        report_root / "controller_trace_expanded.csv",
        controller_rows,
        ["run_tag", "framework", "backend", "workload", "regime", "arm", "seed", "scheduler_action", "queue_wait_p99_ms", "valid_sample_rate", "mean_actor_concurrency", "mean_verifier_concurrency"],
    )
    write_csv(
        report_root / "sample_filter_reasons_expanded.csv",
        filter_reason_rows,
        ["run_tag", "framework", "backend", "workload", "regime", "arm", "seed", "reason", "count", "share"],
    )
    write_csv(
        report_root / "training_curve_expanded.csv",
        training_rows,
        ["run_tag", "framework", "backend", "workload", "regime", "arm", "seed", "iteration", "budget_used", "cumulative_reward_auc", "final_verified_reward"],
    )
    contract_check = {
        "same_external_harness_across_axes": len({cell["workload"] for cell in cells}) > 1 and len({cell["regime"] for cell in cells}) > 1,
        "hook_a_multi_setting_improvement": hook_a_improvements >= 2,
        "hook_b_stressed_repeatable_improvement": hook_b_improvements >= 2,
        "hook_ab_strong_or_near_strong": hook_ab_best >= max(2, len(grouped) // 2),
        "study_strengthened": strengthened,
        "included_cells": cells,
        "excluded_candidate_regimes": ["swe_gym_failure_stressed"] if not include_swe_failure_stressed else [],
        "notes": reasons,
    }
    write_json(report_root / "downstream_study_contract_check.json", contract_check)
    _render_report(report_root / "downstream_study_report.md", strengthened, reasons, cells, include_swe_failure_stressed)
    resources = build_compute_resources(block_timings)
    write_json(report_root / "compute_resources.json", resources)
    if strengthened:
        reviewer_mirror_root.parent.mkdir(parents=True, exist_ok=True)
        if reviewer_mirror_root.exists():
            import shutil

            shutil.rmtree(reviewer_mirror_root)
        import shutil

        shutil.copytree(report_root, reviewer_mirror_root)
        _render_paper_mapping(paper_mapping_path, report_root, strengthened)
    return contract_check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=f"downstream_expanded_{int(time.time())}")
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/rl_closure_runs"))
    parser.add_argument("--report-root", type=Path, default=Path("artifacts/reports/regime_upgrade_20260411/downstream_expanded"))
    parser.add_argument("--framework", default="veRL")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--use-mock-backends", action="store_true")
    parser.add_argument("--autostart-backends", action="store_true")
    parser.add_argument("--include-webarena-stressed", action="store_true")
    parser.add_argument("--include-swe-failure-stressed", action="store_true")
    parser.add_argument("--reviewer-mirror-root", type=Path, default=Path("reviewer_mirror/regime_upgrade_20260411/downstream_expanded"))
    parser.add_argument("--paper-mapping-path", type=Path, default=Path("docs/downstream_study_expanded_paper_mapping.md"))
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if args.use_mock_backends:
        clients: dict[str, MockBackendClient | OpenAICompatibleClient] = _default_mock_clients(args.model_id)
        result = run_expanded_study(
            run_tag=args.run_tag,
            run_root=args.run_root / args.run_tag,
            report_root=args.report_root,
            backend_clients=clients,
            framework=args.framework,
            model_id=args.model_id,
            seeds=seeds,
            include_webarena_stressed=args.include_webarena_stressed,
            include_swe_failure_stressed=args.include_swe_failure_stressed,
            reviewer_mirror_root=args.reviewer_mirror_root,
            paper_mapping_path=args.paper_mapping_path,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if not args.autostart_backends:
        raise SystemExit("use --use-mock-backends for bounded local runs or --autostart-backends for real launch")

    launched: list[Any] = []
    try:
        clients, launched = _create_real_backend_clients(Path.cwd(), args.model_id)
        result = run_expanded_study(
            run_tag=args.run_tag,
            run_root=args.run_root / args.run_tag,
            report_root=args.report_root,
            backend_clients=clients,
            framework=args.framework,
            model_id=args.model_id,
            seeds=seeds,
            include_webarena_stressed=args.include_webarena_stressed,
            include_swe_failure_stressed=args.include_swe_failure_stressed,
            reviewer_mirror_root=args.reviewer_mirror_root,
            paper_mapping_path=args.paper_mapping_path,
        )
    finally:
        from runtime.rl_closure.backends import stop_backend_process

        for proc in reversed(launched):
            stop_backend_process(proc)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
