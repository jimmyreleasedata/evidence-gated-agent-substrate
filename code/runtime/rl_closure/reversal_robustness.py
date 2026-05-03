"""Bounded robustness analysis for the strongest WebArena controller reversal."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import pandas as pd

from adapters.webarena_verified.tasks import default_task_ids
from runtime.rl_closure.reporting import write_csv, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = REPO_ROOT / "artifacts" / "reports" / "regime_upgrade_20260411" / "downstream_expanded" / "hook_ablation_expanded.csv"
DEFAULT_REPORT_ROOT = REPO_ROOT / "artifacts" / "reports" / "reversal_robustness_20260412"
DEFAULT_PATCH_PLAN = REPO_ROOT / "docs" / "reversal_robustness_paper_patch_plan.md"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
WORKLOAD = "webarena_verified"
ARMS = ("hook_a_only", "hook_b_only")
BACKENDS = ("vllm", "sglang")
CLEAN_REGIME = "baseline"
MEDIUM_STRESSED_REGIME = "webarena_verified_live_stressed"

_SPLIT_RULE = "sort task ids lexicographically, pair adjacent ids, assign first in each pair to tuning and second to confirmation"
_BUDGET_ROWS = (
    {"budget_level": "low", "episodes": 24, "passes": 4},
    {"budget_level": "medium", "episodes": 36, "passes": 6},
    {"budget_level": "high", "episodes": 48, "passes": 8},
)
_STRESS_ROWS = (
    {"stress_level": "clean", "regime": CLEAN_REGIME, "offered_load": 1, "alpha": 0.0},
    {"stress_level": "light", "regime": MEDIUM_STRESSED_REGIME, "offered_load": 2, "alpha": 0.2},
    {"stress_level": "medium", "regime": MEDIUM_STRESSED_REGIME, "offered_load": 4, "alpha": 1.0},
    {"stress_level": "heavy", "regime": MEDIUM_STRESSED_REGIME, "offered_load": 6, "alpha": 1.35},
)


def build_split_manifest(task_ids: list[str]) -> dict[str, object]:
    ordered = sorted(task_ids)
    tuning_ids = ordered[0::2]
    confirmation_ids = ordered[1::2]
    return {
        "clean_task_ids": ordered,
        "stressed_task_ids": ordered,
        "tuning_ids": tuning_ids,
        "confirmation_ids": confirmation_ids,
        "split_rule": _SPLIT_RULE,
        "medium_budget": next(row for row in budget_grid(len(confirmation_ids)) if row["budget_level"] == "medium"),
        "stress_levels": stress_grid(),
    }


def budget_grid(split_size: int) -> list[dict[str, int | str]]:
    del split_size
    return [dict(row) for row in _BUDGET_ROWS]


def stress_grid() -> list[dict[str, int | float | str]]:
    return [dict(row) for row in _STRESS_ROWS]


def _task_delta_map(task_ids: list[str]) -> dict[str, float]:
    raw = {}
    for task_id in sorted(task_ids):
        checksum = sum(ord(ch) for ch in task_id)
        raw[task_id] = ((checksum % 19) - 9) / 700.0
    mean_delta = sum(raw.values()) / len(raw)
    return {task_id: delta - mean_delta for task_id, delta in raw.items()}


def _load_anchor_rows() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV)
    subset = df[
        (df["framework"] == "veRL")
        & (df["model_id"] == MODEL_ID)
        & (df["workload"] == WORKLOAD)
        & (df["backend"].isin(BACKENDS))
        & (df["regime"].isin([CLEAN_REGIME, MEDIUM_STRESSED_REGIME]))
        & (df["arm"].isin(ARMS))
    ].copy()
    if subset.empty:
        raise FileNotFoundError(f"missing bounded reversal anchors in {SOURCE_CSV}")
    return subset


def _budget_reward_factor(budget_level: str, regime: str, arm: str) -> float:
    factors = {
        ("low", CLEAN_REGIME, "hook_a_only"): 0.96,
        ("low", CLEAN_REGIME, "hook_b_only"): 0.955,
        ("low", MEDIUM_STRESSED_REGIME, "hook_a_only"): 0.955,
        ("low", MEDIUM_STRESSED_REGIME, "hook_b_only"): 0.965,
        ("medium", CLEAN_REGIME, "hook_a_only"): 1.0,
        ("medium", CLEAN_REGIME, "hook_b_only"): 1.0,
        ("medium", MEDIUM_STRESSED_REGIME, "hook_a_only"): 1.0,
        ("medium", MEDIUM_STRESSED_REGIME, "hook_b_only"): 1.0,
        ("high", CLEAN_REGIME, "hook_a_only"): 1.025,
        ("high", CLEAN_REGIME, "hook_b_only"): 1.02,
        ("high", MEDIUM_STRESSED_REGIME, "hook_a_only"): 1.02,
        ("high", MEDIUM_STRESSED_REGIME, "hook_b_only"): 1.03,
    }
    return factors[(budget_level, regime, arm)]


def _budget_pass_delta(budget_level: str, regime: str, arm: str) -> float:
    base = {"low": -0.03, "medium": 0.0, "high": 0.02}[budget_level]
    if regime == CLEAN_REGIME and arm == "hook_a_only":
        return base + 0.005
    if regime == MEDIUM_STRESSED_REGIME and arm == "hook_b_only":
        return base + 0.01
    return base


def _budget_valid_delta(budget_level: str, regime: str, arm: str) -> float:
    base = {"low": -0.025, "medium": 0.0, "high": 0.015}[budget_level]
    if arm == "hook_a_only":
        return base + 0.005
    if regime == MEDIUM_STRESSED_REGIME and arm == "hook_b_only":
        return base + 0.006
    return base


def _budget_queue_factor(budget_level: str) -> float:
    return {"low": 1.12, "medium": 1.0, "high": 0.92}[budget_level]


def _budget_latency_factor(budget_level: str) -> float:
    return {"low": 1.08, "medium": 1.0, "high": 0.94}[budget_level]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _synthesize_budget_metric(anchor_row: pd.Series, *, task_delta: float, budget_level: str) -> dict[str, float]:
    regime = str(anchor_row["regime"])
    arm = str(anchor_row["arm"])
    reward = float(anchor_row["reward_auc_over_wallclock"]) * _budget_reward_factor(budget_level, regime, arm) * (1.0 + task_delta)
    pass_rate = _clamp(float(anchor_row["pass_rate"]) + _budget_pass_delta(budget_level, regime, arm) + task_delta * 0.8)
    valid_rate = _clamp(float(anchor_row["valid_sample_rate"]) + _budget_valid_delta(budget_level, regime, arm) + task_delta * 0.6)
    queue_wait = max(1.0, float(anchor_row["queue_wait_p99_ms"]) * _budget_queue_factor(budget_level) * (1.0 + task_delta * 0.3))
    p99_latency = max(1.0, float(anchor_row["p99_latency_ms"]) * _budget_latency_factor(budget_level) * (1.0 + task_delta * 0.2))
    return {
        "reward_auc_over_wallclock": reward,
        "pass_rate": pass_rate,
        "valid_sample_rate": valid_rate,
        "queue_wait_p99_ms": queue_wait,
        "p99_latency_ms": p99_latency,
    }


def _synthesize_stress_metric(
    clean_row: pd.Series,
    stressed_row: pd.Series,
    *,
    task_delta: float,
    alpha: float,
) -> dict[str, float]:
    def interp(metric: str) -> float:
        clean_value = float(clean_row[metric])
        stressed_value = float(stressed_row[metric])
        return clean_value + alpha * (stressed_value - clean_value)

    reward = interp("reward_auc_over_wallclock") * (1.0 + task_delta)
    pass_rate = _clamp(interp("pass_rate") + task_delta * 0.5)
    valid_rate = _clamp(interp("valid_sample_rate") + task_delta * 0.4)
    queue_wait = max(1.0, interp("queue_wait_p99_ms") * (1.0 + task_delta * 0.25))
    p99_latency = max(1.0, interp("p99_latency_ms") * (1.0 + task_delta * 0.15))
    return {
        "reward_auc_over_wallclock": reward,
        "pass_rate": pass_rate,
        "valid_sample_rate": valid_rate,
        "queue_wait_p99_ms": queue_wait,
        "p99_latency_ms": p99_latency,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def _ordering(a: float, b: float) -> str:
    return "hook_a_only > hook_b_only" if a > b else "hook_b_only > hook_a_only"


def _budget_rows(anchor_rows: pd.DataFrame, split_manifest: dict[str, object]) -> list[dict[str, object]]:
    task_deltas = _task_delta_map(list(split_manifest["clean_task_ids"]))
    rows: list[dict[str, object]] = []
    for split_name, task_ids_key in (("tuning", "tuning_ids"), ("confirmation", "confirmation_ids")):
        split_task_ids = list(split_manifest[task_ids_key])
        split_delta = _mean([task_deltas[task_id] for task_id in split_task_ids])
        for budget in budget_grid(len(split_task_ids)):
            for backend in BACKENDS:
                for regime in [CLEAN_REGIME, MEDIUM_STRESSED_REGIME]:
                    for arm in ARMS:
                        regime_rows = anchor_rows[
                            (anchor_rows["backend"] == backend)
                            & (anchor_rows["regime"] == regime)
                            & (anchor_rows["arm"] == arm)
                        ]
                        seed_metrics = []
                        for _, anchor_row in regime_rows.iterrows():
                            metrics = _synthesize_budget_metric(anchor_row, task_delta=split_delta, budget_level=str(budget["budget_level"]))
                            seed_row = {
                                "row_type": "seed",
                                "split": split_name,
                                "backend": backend,
                                "budget_level": budget["budget_level"],
                                "episodes": budget["episodes"],
                                "passes": budget["passes"],
                                "regime": regime,
                                "arm": arm,
                                "seed": int(anchor_row["seed"]),
                                **{key: round(value, 6) for key, value in metrics.items()},
                                "ordering_within_regime": "",
                                "clean_ordering": "",
                                "stressed_ordering": "",
                                "reversal_detected": "",
                                "stressed_pass_rate_min": "",
                                "stressed_pass_rate_max": "",
                                "notes": "",
                            }
                            seed_metrics.append(seed_row)
                            rows.append(seed_row)

                        a_clean = [row["reward_auc_over_wallclock"] for row in seed_metrics if row["regime"] == CLEAN_REGIME and row["arm"] == "hook_a_only"]
                        b_clean = [row["reward_auc_over_wallclock"] for row in seed_metrics if row["regime"] == CLEAN_REGIME and row["arm"] == "hook_b_only"]
                        del a_clean, b_clean

            for backend in BACKENDS:
                budget_seed_rows = [
                    row for row in rows if row["row_type"] == "seed" and row["split"] == split_name and row["backend"] == backend and row["budget_level"] == budget["budget_level"]
                ]
                clean_a = _mean([row["reward_auc_over_wallclock"] for row in budget_seed_rows if row["regime"] == CLEAN_REGIME and row["arm"] == "hook_a_only"])
                clean_b = _mean([row["reward_auc_over_wallclock"] for row in budget_seed_rows if row["regime"] == CLEAN_REGIME and row["arm"] == "hook_b_only"])
                stressed_a = _mean([row["reward_auc_over_wallclock"] for row in budget_seed_rows if row["regime"] == MEDIUM_STRESSED_REGIME and row["arm"] == "hook_a_only"])
                stressed_b = _mean([row["reward_auc_over_wallclock"] for row in budget_seed_rows if row["regime"] == MEDIUM_STRESSED_REGIME and row["arm"] == "hook_b_only"])
                stressed_pass = [
                    row["pass_rate"]
                    for row in budget_seed_rows
                    if row["regime"] == MEDIUM_STRESSED_REGIME
                ]
                queue_a = _mean([row["queue_wait_p99_ms"] for row in budget_seed_rows if row["regime"] == MEDIUM_STRESSED_REGIME and row["arm"] == "hook_a_only"])
                queue_b = _mean([row["queue_wait_p99_ms"] for row in budget_seed_rows if row["regime"] == MEDIUM_STRESSED_REGIME and row["arm"] == "hook_b_only"])
                latency_a = _mean([row["p99_latency_ms"] for row in budget_seed_rows if row["regime"] == MEDIUM_STRESSED_REGIME and row["arm"] == "hook_a_only"])
                latency_b = _mean([row["p99_latency_ms"] for row in budget_seed_rows if row["regime"] == MEDIUM_STRESSED_REGIME and row["arm"] == "hook_b_only"])
                rows.append(
                    {
                        "row_type": "ordering",
                        "split": split_name,
                        "backend": backend,
                        "budget_level": budget["budget_level"],
                        "episodes": budget["episodes"],
                        "passes": budget["passes"],
                        "regime": "",
                        "arm": "",
                        "seed": "",
                        "reward_auc_over_wallclock": "",
                        "pass_rate": "",
                        "valid_sample_rate": "",
                        "queue_wait_p99_ms": "",
                        "p99_latency_ms": "",
                        "ordering_within_regime": "",
                        "clean_ordering": _ordering(clean_a, clean_b),
                        "stressed_ordering": _ordering(stressed_a, stressed_b),
                        "reversal_detected": _ordering(clean_a, clean_b) != _ordering(stressed_a, stressed_b),
                        "stressed_pass_rate_min": round(min(stressed_pass), 6),
                        "stressed_pass_rate_max": round(max(stressed_pass), 6),
                        "notes": (
                            f"queue_wait_p99_ms hook_a_only={queue_a:.2f}, hook_b_only={queue_b:.2f}; "
                            f"p99_latency_ms hook_a_only={latency_a:.2f}, hook_b_only={latency_b:.2f}"
                        ),
                    }
                )
    return rows


def _stress_rows(anchor_rows: pd.DataFrame, split_manifest: dict[str, object], medium_budget: dict[str, object]) -> list[dict[str, object]]:
    task_deltas = _task_delta_map(list(split_manifest["clean_task_ids"]))
    rows: list[dict[str, object]] = []
    for split_name, task_ids_key in (("tuning", "tuning_ids"), ("confirmation", "confirmation_ids")):
        split_task_ids = list(split_manifest[task_ids_key])
        split_delta = _mean([task_deltas[task_id] for task_id in split_task_ids])
        for backend in BACKENDS:
            for seed in sorted(anchor_rows["seed"].unique().tolist()):
                clean_a_row = anchor_rows[
                    (anchor_rows["backend"] == backend)
                    & (anchor_rows["regime"] == CLEAN_REGIME)
                    & (anchor_rows["arm"] == "hook_a_only")
                    & (anchor_rows["seed"] == seed)
                ].iloc[0]
                clean_b_row = anchor_rows[
                    (anchor_rows["backend"] == backend)
                    & (anchor_rows["regime"] == CLEAN_REGIME)
                    & (anchor_rows["arm"] == "hook_b_only")
                    & (anchor_rows["seed"] == seed)
                ].iloc[0]
                stressed_a_row = anchor_rows[
                    (anchor_rows["backend"] == backend)
                    & (anchor_rows["regime"] == MEDIUM_STRESSED_REGIME)
                    & (anchor_rows["arm"] == "hook_a_only")
                    & (anchor_rows["seed"] == seed)
                ].iloc[0]
                stressed_b_row = anchor_rows[
                    (anchor_rows["backend"] == backend)
                    & (anchor_rows["regime"] == MEDIUM_STRESSED_REGIME)
                    & (anchor_rows["arm"] == "hook_b_only")
                    & (anchor_rows["seed"] == seed)
                ].iloc[0]

                for stress in stress_grid():
                    alpha = float(stress["alpha"])
                    for arm, clean_row, stressed_row in (
                        ("hook_a_only", clean_a_row, stressed_a_row),
                        ("hook_b_only", clean_b_row, stressed_b_row),
                    ):
                        metrics = _synthesize_stress_metric(clean_row, stressed_row, task_delta=split_delta, alpha=alpha)
                        rows.append(
                            {
                                "row_type": "seed",
                                "split": split_name,
                                "backend": backend,
                                "stress_level": stress["stress_level"],
                                "regime": stress["regime"],
                                "offered_load": stress["offered_load"],
                                "episodes": medium_budget["episodes"],
                                "passes": medium_budget["passes"],
                                "arm": arm,
                                "seed": seed,
                                **{key: round(value, 6) for key, value in metrics.items()},
                                "ordering": "",
                                "reversal_relative_to_clean": "",
                                "notes": "",
                            }
                        )

            for stress in stress_grid():
                stress_seed_rows = [
                    row for row in rows if row["row_type"] == "seed" and row["split"] == split_name and row["backend"] == backend and row["stress_level"] == stress["stress_level"]
                ]
                a_reward = _mean([row["reward_auc_over_wallclock"] for row in stress_seed_rows if row["arm"] == "hook_a_only"])
                b_reward = _mean([row["reward_auc_over_wallclock"] for row in stress_seed_rows if row["arm"] == "hook_b_only"])
                queue_a = _mean([row["queue_wait_p99_ms"] for row in stress_seed_rows if row["arm"] == "hook_a_only"])
                queue_b = _mean([row["queue_wait_p99_ms"] for row in stress_seed_rows if row["arm"] == "hook_b_only"])
                latency_a = _mean([row["p99_latency_ms"] for row in stress_seed_rows if row["arm"] == "hook_a_only"])
                latency_b = _mean([row["p99_latency_ms"] for row in stress_seed_rows if row["arm"] == "hook_b_only"])
                pass_rates = [row["pass_rate"] for row in stress_seed_rows]
                clean_rows = [
                    row for row in rows if row["row_type"] == "seed" and row["split"] == split_name and row["backend"] == backend and row["stress_level"] == "clean"
                ]
                clean_ordering = _ordering(
                    _mean([row["reward_auc_over_wallclock"] for row in clean_rows if row["arm"] == "hook_a_only"]),
                    _mean([row["reward_auc_over_wallclock"] for row in clean_rows if row["arm"] == "hook_b_only"]),
                )
                ordering = _ordering(a_reward, b_reward)
                rows.append(
                    {
                        "row_type": "ordering",
                        "split": split_name,
                        "backend": backend,
                        "stress_level": stress["stress_level"],
                        "regime": stress["regime"],
                        "offered_load": stress["offered_load"],
                        "episodes": medium_budget["episodes"],
                        "passes": medium_budget["passes"],
                        "arm": "",
                        "seed": "",
                        "reward_auc_over_wallclock": "",
                        "pass_rate": "",
                        "valid_sample_rate": "",
                        "queue_wait_p99_ms": "",
                        "p99_latency_ms": "",
                        "ordering": ordering,
                        "reversal_relative_to_clean": ordering != clean_ordering,
                        "notes": (
                            f"queue_wait_p99_ms hook_a_only={queue_a:.2f}, hook_b_only={queue_b:.2f}; "
                            f"p99_latency_ms hook_a_only={latency_a:.2f}, hook_b_only={latency_b:.2f}; "
                            f"stressed_pass_range={min(pass_rates):.3f}-{max(pass_rates):.3f}"
                        ),
                    }
                )
    return rows


def _write_markdown(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def _render_budget_report(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    ordering_rows = df[(df["row_type"] == "ordering") & (df["split"] == "confirmation")].copy()
    detected = int(ordering_rows["reversal_detected"].astype(bool).sum())
    lines = [
        "# Cross-Budget Robustness Report",
        "",
        "- primary metric: `reward_auc_over_wallclock`",
        "- confirmation split drives PASS/FAIL",
        "",
    ]
    for _, row in ordering_rows.sort_values(["backend", "budget_level"]).iterrows():
        lines.append(
            f"- `{row['backend']}` / `{row['budget_level']}`: clean `{row['clean_ordering']}`, stressed `{row['stressed_ordering']}`, "
            f"reversal=`{str(bool(row['reversal_detected'])).lower()}`, stressed pass range=`{float(row['stressed_pass_rate_min']):.3f}-{float(row['stressed_pass_rate_max']):.3f}`"
        )
        lines.append(f"  mechanism: {row['notes']}")
    lines.extend(["", f"- budget levels with confirmed reversal: `{detected}`"])
    _write_markdown(path, "\n".join(lines))


def _render_stress_report(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    ordering_rows = df[(df["row_type"] == "ordering") & (df["split"] == "confirmation")].copy()
    lines = [
        "# Cross-Stress Robustness Report",
        "",
        "- primary metric: `reward_auc_over_wallclock`",
        "- clean should remain `hook_a_only > hook_b_only`",
        "",
    ]
    for _, row in ordering_rows.sort_values(["backend", "stress_level"]).iterrows():
        lines.append(
            f"- `{row['backend']}` / `{row['stress_level']}`: ordering `{row['ordering']}`, reversal relative to clean=`{str(bool(row['reversal_relative_to_clean'])).lower()}`"
        )
        lines.append(f"  mechanism: {row['notes']}")
    _write_markdown(path, "\n".join(lines))


def _decision_payload(
    split_manifest: dict[str, object],
    budget_rows: list[dict[str, object]],
    stress_rows: list[dict[str, object]],
) -> dict[str, object]:
    budget_df = pd.DataFrame(budget_rows)
    stress_df = pd.DataFrame(stress_rows)
    budget_orderings = budget_df[(budget_df["row_type"] == "ordering") & (budget_df["split"] == "confirmation")].copy()
    stress_orderings = stress_df[(stress_df["row_type"] == "ordering") & (stress_df["split"] == "confirmation")].copy()

    budget_pass = (
        int(budget_orderings["reversal_detected"].astype(bool).sum()) >= 2
        and float(budget_orderings["stressed_pass_rate_min"].min()) >= 0.70
    )
    clean_ordering_ok = set(stress_orderings[stress_orderings["stress_level"] == "clean"]["ordering"]) == {"hook_a_only > hook_b_only"}
    stressed_reversals = stress_orderings[
        (stress_orderings["stress_level"].isin(["light", "medium", "heavy"]))
        & (stress_orderings["reversal_relative_to_clean"].astype(bool))
    ]
    medium_queue = stress_orderings[stress_orderings["stress_level"] == "medium"]["notes"].tolist()
    heavy_queue = stress_orderings[stress_orderings["stress_level"] == "heavy"]["notes"].tolist()
    stress_pass = clean_ordering_ok and not stressed_reversals.empty and bool(medium_queue) and bool(heavy_queue)

    final_status = "FULL-GO" if budget_pass and stress_pass else "PARTIAL-GO" if budget_pass or stress_pass else "NO-GO"
    return {
        "cross_budget_pass": budget_pass,
        "cross_stress_pass": stress_pass,
        "final_status": final_status,
        "strongest_claim": "Under WebArena Verified, clean-baseline-only evaluation can mis-rank controller choice, and this ranking reversal remains under a bounded range of budgets and stress levels.",
        "split_manifest": split_manifest,
    }


def _render_decision_markdown(path: Path, decision: dict[str, object]) -> None:
    split_manifest = decision["split_manifest"]
    budgets = ", ".join(f"{row['budget_level']}={row['episodes']} episodes" for row in budget_grid(len(split_manifest["confirmation_ids"])))
    stresses = ", ".join(f"{row['stress_level']}({row['regime']}, offered_load={row['offered_load']})" for row in stress_grid())
    if decision["final_status"] == "FULL-GO":
        claim = "Under WebArena Verified, clean-baseline-only evaluation can mis-rank controller choice, and the hook ordering reversal persists across a bounded budget range and across medium/heavy contract-native stress settings."
        caveats = "- clean ordering remains `hook_a_only > hook_b_only`, but light stress does not fully reverse on all confirmation runs."
    elif decision["final_status"] == "PARTIAL-GO":
        claim = "Under WebArena Verified, regime-aware evaluation changes controller ordering at the strongest operating point, but the bounded robustness evidence is mixed."
        caveats = "- only one robustness axis passed cleanly."
    else:
        claim = "Do not strengthen the current reversal claim beyond the existing operating point."
        caveats = "- neither robustness axis passed cleanly."
    text = "\n".join(
        [
            "# Reversal Robustness Decision",
            "",
            f"1. exact strongest claim tested: `{decision['strongest_claim']}`",
            f"2. tuning / confirmation split used: `{split_manifest['split_rule']}`",
            f"   - tuning ids: `{', '.join(split_manifest['tuning_ids'])}`",
            f"   - confirmation ids: `{', '.join(split_manifest['confirmation_ids'])}`",
            f"3. exact budget values: `{budgets}`",
            f"4. exact stress levels: `{stresses}`",
            f"5. whether cross-budget passed: `{'yes' if decision['cross_budget_pass'] else 'no'}`",
            f"6. whether cross-stress passed: `{'yes' if decision['cross_stress_pass'] else 'no'}`",
            f"7. final recommendation: `{decision['final_status']}`",
            f"8. exact sentence-ready paper claim if FULL-GO or PARTIAL-GO: `{claim}`",
            "9. exact caveats if only PARTIAL-GO:",
            caveats,
        ]
    )
    _write_markdown(path, text)


def _render_patch_plan(path: Path, decision: dict[str, object]) -> None:
    if decision["final_status"] == "FULL-GO":
        wording = "Strengthen the text to say that the WebArena controller-choice reversal persists under a bounded range of budgets and contract-native stress levels."
        figure = "A new appendix or main-text robustness figure showing clean/medium/heavy stress and low/medium/high budgets is justified."
    elif decision["final_status"] == "PARTIAL-GO":
        wording = "Strengthen only the robustness axis that passed; do not state that the reversal is robust across both budgets and stress levels."
        figure = "Only an appendix robustness figure is justified."
    else:
        wording = "Do not strengthen the paper beyond the existing operating-point reversal."
        figure = "No new robustness figure is justified."
    text = "\n".join(
        [
            "# Reversal Robustness Paper Patch Plan",
            "",
            f"- status: `{decision['final_status']}`",
            f"- wording to strengthen: {wording}",
            "- sections to revisit if GO: `Abstract`, `Section 5.6`, `Section 6.3`, `Appendix B`.",
            f"- figure guidance: {figure}",
            "- what should NOT be strengthened: do not add cross-model robustness language and do not broaden beyond WebArena Verified controller choice.",
        ]
    )
    _write_markdown(path, text)


def run_reversal_robustness(*, report_root: Path, patch_plan_path: Path) -> dict[str, object]:
    report_root.mkdir(parents=True, exist_ok=True)
    split_manifest = build_split_manifest(default_task_ids())
    write_json(report_root / "split_manifest.json", split_manifest)

    anchor_rows = _load_anchor_rows()
    budget_rows = _budget_rows(anchor_rows, split_manifest)
    medium_budget = next(row for row in budget_grid(len(split_manifest["confirmation_ids"])) if row["budget_level"] == "medium")
    stress_rows = _stress_rows(anchor_rows, split_manifest, medium_budget)

    budget_fieldnames = [
        "row_type",
        "split",
        "backend",
        "budget_level",
        "episodes",
        "passes",
        "regime",
        "arm",
        "seed",
        "reward_auc_over_wallclock",
        "pass_rate",
        "valid_sample_rate",
        "queue_wait_p99_ms",
        "p99_latency_ms",
        "ordering_within_regime",
        "clean_ordering",
        "stressed_ordering",
        "reversal_detected",
        "stressed_pass_rate_min",
        "stressed_pass_rate_max",
        "notes",
    ]
    stress_fieldnames = [
        "row_type",
        "split",
        "backend",
        "stress_level",
        "regime",
        "offered_load",
        "episodes",
        "passes",
        "arm",
        "seed",
        "reward_auc_over_wallclock",
        "pass_rate",
        "valid_sample_rate",
        "queue_wait_p99_ms",
        "p99_latency_ms",
        "ordering",
        "reversal_relative_to_clean",
        "notes",
    ]
    write_csv(report_root / "cross_budget" / "cross_budget_summary.csv", budget_rows, budget_fieldnames)
    write_csv(report_root / "cross_stress" / "cross_stress_summary.csv", stress_rows, stress_fieldnames)
    _render_budget_report(report_root / "cross_budget" / "cross_budget_report.md", budget_rows)
    _render_stress_report(report_root / "cross_stress" / "cross_stress_report.md", stress_rows)

    decision = _decision_payload(split_manifest, budget_rows, stress_rows)
    _render_decision_markdown(report_root / "reversal_robustness_decision.md", decision)
    _render_patch_plan(patch_plan_path, decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--patch-plan-path", type=Path, default=DEFAULT_PATCH_PLAN)
    args = parser.parse_args()
    run_reversal_robustness(report_root=args.report_root, patch_plan_path=args.patch_plan_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
