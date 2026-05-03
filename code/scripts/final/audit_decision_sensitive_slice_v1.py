#!/usr/bin/env python3
"""Audit admitted WebArena Verified controller-decision evidence.

This is an admission/filtering audit. It does not run WebArena, start a live
host, modify paper text, modify reviewer bundles, or regenerate figures.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


TRUE_VALUES = {"1", "true", "yes", "y", "pass", "passed", "ok", "supported"}
FALSE_VALUES = {"0", "false", "no", "n", "fail", "failed", "blocked", "unsupported"}
HOOKS = {"hook_a_only", "hook_b_only"}
BACKENDS = {"vllm", "sglang"}
CLEAN_REGIME_TOKENS = {"clean", "clean_baseline", "baseline", "basic"}
STRESSED_REGIME_TOKENS = {"stressed", "stress", "medium", "heavy", "promoted_stressed", "webarena_stressed"}
SKIP_SCAN_DIRS = {"submission", "submission_package", "logs", "__pycache__", ".git"}

DRIVER_FIELDS = [
    "driver_id",
    "driver_type",
    "driver_version",
    "model_family",
    "model_id",
    "model_backend",
    "backend_engine",
    "policy_version",
    "prompt_template_hash",
    "action_parser_version",
    "budget",
    "implementation_source",
]
MANIFEST_FIELDS_ANY = ["manifest_hash", "manifest_path", "run_manifest_path", "task_manifest_hash", "task_manifest_path"]
MANIFEST_REQUIRED_ANY = {
    "manifest_binding": MANIFEST_FIELDS_ANY,
    "schema_version": ["schema_version", "trace_schema_version"],
    "replay_class": ["replay_class"],
    "release_root": ["release_root", "source_root"],
    "evaluator_version": ["evaluator_version", "verifier_version", "evaluator_id", "verifier_id"],
}
METRIC_FIELDS = [
    "reward_auc_over_wallclock",
    "reward_auc_over_wallclock_mean",
    "reward_auc",
    "decision_metric",
]
PASS_FIELDS = ["pass_rate", "passed", "task_success", "terminal_success", "success"]
TERMINAL_FIELDS = ["terminal_outcome", "terminal_outcome_present", "task_success", "passed", "score", "reward", "terminated", "truncated"]
SOURCE_PRIORITY = [
    "decision_sensitive_admission/controller_trace.csv",
    "decision_sensitive_admission/controller_summary.csv",
    "decision_sensitive_admission/decision_reversal_summary.csv",
    "global/paper_facing_surface.csv",
    "global/excluded_rows.csv",
    "global/global_evidence_index.csv",
    "rounds/closeout_extension_riskfix/model_backend/webarena_model_backend_summary.csv",
    "reports/webarena_stress/webarena_paired_stress_summary.csv",
]
OUTPUT_COLUMNS = [
    "audit_row_id",
    "family",
    "task_id",
    "instance_id",
    "run_id",
    "decision_label",
    "decision_label_provenance",
    "regime",
    "backend_engine",
    "budget",
    "seed",
    "decision_metric",
    "pass_rate",
    "terminal_outcome",
    "evidence_validation_pass",
    "paper_role",
    "source_root",
    "source_file",
    "audit_source_path",
    "admission_status",
    "block_reason",
    "provenance_original_columns",
]


@dataclass(frozen=True)
class AuditResult:
    candidate_count: int
    admitted_count: int
    blocked_count: int
    supported_claim_level: str
    targeted_rerun_required: bool
    summary_path: Path


def truthy(value: object) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def falsey(value: object) -> bool:
    return str(value).strip().lower() in FALSE_VALUES


def blank(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}


def value(row: pd.Series, names: Sequence[str], default: str = "") -> object:
    for name in names:
        if name in row.index and not blank(row.get(name)):
            return row.get(name)
    return default


def normalize_family(raw: object) -> str:
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "webarena" in text:
        return "webarena_verified"
    return text


def normalize_backend(raw: object) -> str:
    text = str(raw or "").strip().lower()
    if "sglang" in text:
        return "sglang"
    if "vllm" in text:
        return "vllm"
    return text


def normalize_regime(raw: object) -> str:
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in CLEAN_REGIME_TOKENS or "clean" in text or "baseline" in text:
        return "clean"
    if text in STRESSED_REGIME_TOKENS or "stress" in text or text in {"medium", "heavy"}:
        return "stressed"
    return text


def derive_decision_label(row: pd.Series) -> tuple[str, str]:
    for name in ["decision_label", "controller", "hook", "controller_hook", "selected_controller"]:
        if name in row.index and not blank(row.get(name)):
            text = str(row.get(name)).strip().lower()
            for hook in HOOKS:
                if hook in text:
                    return hook, name
    for name in ["decision_labels", "source_file", "input_csv", "summary_path"]:
        if name in row.index and not blank(row.get(name)):
            text = str(row.get(name)).strip().lower()
            for hook in HOOKS:
                if hook in text:
                    return hook, name
    return "", ""


def first_numeric(row: pd.Series, names: Sequence[str]) -> float | None:
    raw = value(row, names, "")
    if blank(raw):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def has_any(row: pd.Series, names: Sequence[str]) -> bool:
    return any(name in row.index and not blank(row.get(name)) for name in names)


def discover_input_files(root: Path, *, discovery_mode: str = "compact_first", max_files: int = 2000, time_budget_s: int = 25) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for rel in SOURCE_PRIORITY:
        path = root / rel
        if path.exists() and path.is_file():
            files.append(path)
            seen.add(path.resolve())
    if discovery_mode == "known_only":
        return files
    started = time.monotonic()
    suffixes = {".csv", ".json", ".jsonl"}
    for current, dirs, names in os.walk(root):
        if time.monotonic() - started > time_budget_s:
            break
        dirs[:] = [d for d in dirs if d not in SKIP_SCAN_DIRS]
        current_path = Path(current)
        for name in names:
            if len(files) >= max_files:
                return files
            path = current_path / name
            if path.suffix.lower() not in suffixes:
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            files.append(path)
            seen.add(resolved)
    return files


def read_table(path: Path, root: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, low_memory=False)
        elif path.suffix.lower() == ".jsonl":
            df = pd.read_json(path, lines=True)
        elif path.suffix.lower() == ".json":
            data = pd.read_json(path, typ="series")
            if isinstance(data, pd.Series):
                df = pd.DataFrame([data.to_dict()])
            else:
                df = pd.DataFrame(data)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.copy()
    df["audit_source_path"] = str(path)
    try:
        df["audit_source_relpath"] = str(path.relative_to(root))
    except ValueError:
        df["audit_source_relpath"] = str(path)
    return df


def build_candidates(root: Path, discovery_mode: str = "compact_first") -> tuple[pd.DataFrame, list[str]]:
    inspected: list[str] = []
    frames: list[pd.DataFrame] = []
    for path in discover_input_files(root, discovery_mode=discovery_mode):
        inspected.append(str(path))
        df = read_table(path, root)
        if df.empty:
            continue
        lower_cols = {str(col).lower(): col for col in df.columns}
        df = df.rename(columns={original: lower for lower, original in lower_cols.items()})
        family_col = "family" if "family" in df.columns else "task_family" if "task_family" in df.columns else ""
        text_blob = df.astype(str).agg(" ".join, axis=1).str.lower()
        is_web = text_blob.str.contains("webarena", na=False)
        has_controller_signal = text_blob.str.contains("hook_a_only|hook_b_only|controller|reversal", regex=True, na=False)
        if family_col:
            is_web = is_web | df[family_col].map(normalize_family).eq("webarena_verified")
        selected = df[is_web & has_controller_signal].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return pd.DataFrame(), inspected
    candidates = pd.concat(frames, ignore_index=True, sort=False)
    candidates["audit_row_id"] = [f"candidate_{idx:06d}" for idx in range(len(candidates))]
    return candidates, inspected


def block_reasons(row: pd.Series, root: Path) -> tuple[list[str], dict[str, object]]:
    reasons: list[str] = []
    decision_label, decision_provenance = derive_decision_label(row)
    family = normalize_family(value(row, ["family", "task_family"], ""))
    backend = normalize_backend(value(row, ["backend_engine", "model_backend", "backend"], ""))
    regime = normalize_regime(value(row, ["regime", "mode", "telemetry_mode"], ""))
    metric = first_numeric(row, METRIC_FIELDS)
    source_root = str(value(row, ["source_root", "release_root"], ""))
    source_path = Path(str(value(row, ["audit_source_path"], "")))
    source_ok = source_root.startswith(str(root)) or source_root.startswith(str(root.resolve())) or str(source_path).startswith(str(root))

    if not source_ok:
        reasons.append("source_not_manifest_linked_to_canonical_root")
    if family != "webarena_verified":
        reasons.append("not_webarena_verified")
    for flag, reason in [
        ("preflight_only", "preflight_only"),
        ("fixture_only", "fixture_backed"),
        ("smoke_only", "smoke_only"),
        ("synthetic", "synthetic_or_mock"),
        ("mock", "synthetic_or_mock"),
        ("diagnostic_only", "diagnostic_only_not_promoted"),
    ]:
        if flag in row.index and truthy(row.get(flag)):
            reasons.append(reason)
    paper_role = str(value(row, ["paper_role"], "")).lower()
    if paper_role in {"preflight_only", "fixture_only", "smoke_only", "calibration_only"}:
        reasons.append(f"non_paper_role_{paper_role}")
    if decision_label not in HOOKS:
        reasons.append("missing_hook_decision_label")
    if regime not in {"clean", "stressed"}:
        reasons.append("regime_not_clean_or_promoted_stressed")
    if backend and backend not in BACKENDS:
        reasons.append("backend_not_vllm_or_sglang")
    if not backend:
        reasons.append("missing_backend")
    if blank(value(row, ["budget"], "")):
        reasons.append("missing_fixed_budget")
    if blank(value(row, ["seed", "environment_seed"], "")):
        reasons.append("missing_seed")
    if not has_any(row, TERMINAL_FIELDS):
        reasons.append("missing_terminal_outcome")
    if not has_any(row, PASS_FIELDS):
        reasons.append("missing_pass_rate_or_pass_outcome")
    if metric is None:
        reasons.append("missing_decision_metric")
    if not has_any(row, ["decision_label", "controller", "hook", "controller_hook", "selected_controller", "decision_labels"]):
        reasons.append("missing_decision_label_provenance")
    for field in DRIVER_FIELDS:
        if not has_any(row, [field]):
            reasons.append(f"missing_driver_metadata_{field}")
    for label, aliases in MANIFEST_REQUIRED_ANY.items():
        if not has_any(row, aliases):
            reasons.append(f"missing_manifest_binding_{label}")
    if not truthy(value(row, ["evidence_validation_pass", "validation_pass", "evidence_pass"], "")):
        reasons.append("evidence_validation_pass_false")
    normalized = {
        "audit_row_id": row.get("audit_row_id", ""),
        "family": family,
        "task_id": value(row, ["task_id", "upstream_task_id", "env_id"], ""),
        "instance_id": value(row, ["instance_id"], ""),
        "run_id": value(row, ["run_id", "trace_id"], ""),
        "decision_label": decision_label,
        "decision_label_provenance": decision_provenance,
        "regime": regime,
        "backend_engine": backend,
        "budget": value(row, ["budget"], ""),
        "seed": value(row, ["seed", "environment_seed"], ""),
        "decision_metric": metric if metric is not None else "",
        "pass_rate": value(row, PASS_FIELDS, ""),
        "terminal_outcome": value(row, TERMINAL_FIELDS, ""),
        "evidence_validation_pass": value(row, ["evidence_validation_pass", "validation_pass", "evidence_pass"], ""),
        "paper_role": value(row, ["paper_role"], ""),
        "source_root": source_root,
        "source_file": value(row, ["source_file", "audit_source_relpath"], ""),
        "audit_source_path": value(row, ["audit_source_path"], ""),
        "provenance_original_columns": ";".join(map(str, row.index)),
    }
    return sorted(set(reasons)), normalized


def apply_admission(candidates: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    admitted_rows: list[dict[str, object]] = []
    blocked_rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        reasons, normalized = block_reasons(row, root)
        if reasons:
            blocked = dict(normalized)
            blocked["block_reason"] = ";".join(reasons)
            blocked_rows.append(blocked)
        else:
            admitted = dict(normalized)
            admitted["admission_status"] = "admitted"
            admitted_rows.append(admitted)
    return pd.DataFrame(admitted_rows), pd.DataFrame(blocked_rows)


def compute_claim_level(admitted: pd.DataFrame) -> tuple[str, dict[str, object], pd.DataFrame]:
    if admitted.empty:
        return "NONE", {"reason": "no_admitted_rows"}, pd.DataFrame()
    rows: list[dict[str, object]] = []
    reversals: list[dict[str, object]] = []
    grouped = admitted.copy()
    grouped["decision_metric"] = pd.to_numeric(grouped["decision_metric"], errors="coerce")
    for (backend, budget, seed), group in grouped.groupby(["backend_engine", "budget", "seed"], dropna=False):
        winners: dict[str, str] = {}
        for regime, regime_group in group.groupby("regime"):
            hook_scores = regime_group.groupby("decision_label")["decision_metric"].mean().dropna()
            if {"hook_a_only", "hook_b_only"} <= set(hook_scores.index):
                winner = str(hook_scores.sort_values(ascending=False).index[0])
                winners[regime] = winner
                rows.append(
                    {
                        "backend_engine": backend,
                        "budget": budget,
                        "seed": seed,
                        "regime": regime,
                        "hook_a_metric": hook_scores.get("hook_a_only", ""),
                        "hook_b_metric": hook_scores.get("hook_b_only", ""),
                        "selected_controller": winner,
                    }
                )
        if winners.get("clean") and winners.get("stressed"):
            reversal = winners["clean"] != winners["stressed"]
            reversals.append(
                {
                    "backend_engine": backend,
                    "budget": budget,
                    "seed": seed,
                    "clean_selected": winners["clean"],
                    "stressed_selected": winners["stressed"],
                    "ordering_reverses": reversal,
                }
            )
    comparison = pd.DataFrame(rows + reversals)
    reversal_df = pd.DataFrame(reversals)
    if reversal_df.empty or not reversal_df["ordering_reverses"].astype(bool).any():
        return "NONE", {"reason": "no_admitted_reversal_or_clean_misranking"}, comparison
    backends_with_reversal = set(reversal_df.loc[reversal_df["ordering_reverses"].astype(bool), "backend_engine"].astype(str))
    strong = {"vllm", "sglang"} <= backends_with_reversal
    if strong:
        return "STRONG", {"backends_with_reversal": ",".join(sorted(backends_with_reversal))}, comparison
    return "BOUNDED", {"backends_with_reversal": ",".join(sorted(backends_with_reversal))}, comparison


def write_outputs(
    out_dir: Path,
    admitted: pd.DataFrame,
    blocked: pd.DataFrame,
    comparison: pd.DataFrame,
    inspected: list[str],
    claim_level: str,
    claim_detail: dict[str, object],
) -> AuditResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    admitted_path = out_dir / "decision_slice_admitted.csv"
    blocked_path = out_dir / "decision_slice_blocked.csv"
    admitted.reindex(columns=[col for col in OUTPUT_COLUMNS if col != "block_reason"]).to_csv(admitted_path, index=False)
    blocked.reindex(columns=[col for col in OUTPUT_COLUMNS if col != "admission_status"]).to_csv(blocked_path, index=False)
    if not comparison.empty:
        comparison.to_csv(out_dir / "decision_slice_comparison.csv", index=False)

    clean_exists = not admitted.empty and "clean" in set(admitted["regime"])
    stressed_exists = not admitted.empty and "stressed" in set(admitted["regime"])
    backend_set = set(admitted["backend_engine"].astype(str)) if not admitted.empty else set()
    all_hooks = set(admitted["decision_label"].astype(str)) if not admitted.empty else set()
    fixed_budget = bool(not admitted.empty and admitted["budget"].replace("", pd.NA).notna().all())
    seeds = set(admitted["seed"].astype(str)) if not admitted.empty else set()
    metric_available = bool(not admitted.empty and admitted["decision_metric"].replace("", pd.NA).notna().all())
    terminal_available = bool(not admitted.empty and admitted["terminal_outcome"].replace("", pd.NA).notna().all())
    pass_available = bool(not admitted.empty and admitted["pass_rate"].replace("", pd.NA).notna().all())
    target_required = claim_level != "STRONG"

    summary_lines = [
        "# Decision-Sensitive Controller Slice Admission Audit",
        "",
        f"- number of candidate rows found: {len(admitted) + len(blocked)}",
        f"- number admitted: {len(admitted)}",
        f"- number blocked: {len(blocked)}",
        f"- root paths inspected: {len(inspected)}",
        f"- clean/stressed pairing exists: {'yes' if clean_exists and stressed_exists else 'no'}",
        f"- vLLM/SGLang pairing exists: {'yes' if {'vllm', 'sglang'} <= backend_set else 'no'}",
        f"- all fixed-budget seeds exist: {'yes' if fixed_budget and bool(seeds) else 'no'}",
        f"- pass-rate / terminal outcome is available: {'yes' if terminal_available and pass_available else 'no'}",
        f"- reward_auc_over_wallclock or equivalent decision metric is available: {'yes' if metric_available else 'no'}",
        f"- supported claim level: {claim_level}",
        f"- claim detail: {claim_detail}",
        "",
        "## Root Paths Inspected",
        *[f"- {path}" for path in inspected[:200]],
    ]
    summary_path = out_dir / "decision_slice_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if claim_level == "STRONG":
        language = (
            "Across admitted vLLM/SGLang fixed-budget WebArena Verified controller rows, "
            "the evidence-gated slice supports a clean-versus-stressed reversal in selected controller. "
            "This statement is limited to admitted rows listed in decision_slice_admitted.csv."
        )
    elif claim_level == "BOUNDED":
        language = (
            "Within one admitted WebArena Verified controller slice, clean-baseline-only evaluation can mis-rank "
            "controller choice relative to the promoted stressed-regime evidence. This bounded statement should not "
            "be generalized beyond the admitted backend/budget/seed cells in decision_slice_admitted.csv."
        )
    else:
        language = (
            "No WebArena Verified controller-decision claim is currently supported by admitted rows in the current "
            "canonical final root. The decision-sensitive rows are absent, incomplete, unpaired, or blocked under "
            "the evidence gate; do not write a positive controller-reversal claim."
        )
    (out_dir / "claim_language.md").write_text("# Claim Language\n\n" + language + "\n", encoding="utf-8")

    missing_cells = []
    if not clean_exists:
        missing_cells.append("clean_baseline hook_a_only/hook_b_only admitted rows")
    if not stressed_exists:
        missing_cells.append("promoted stressed hook_a_only/hook_b_only admitted rows")
    if not {"vllm", "sglang"} <= backend_set:
        missing_cells.append("both vLLM and SGLang backend cells")
    if not HOOKS <= all_hooks:
        missing_cells.append("both hook_a_only and hook_b_only controller cells")
    if not metric_available:
        missing_cells.append("reward_auc_over_wallclock or equivalent decision metric")
    if not terminal_available or not pass_available:
        missing_cells.append("terminal outcome plus pass-rate/pass outcome")
    plan_lines = [
        "# Targeted Rerun Plan",
        "",
        "Do not run a full canonical rerun. Only run the missing cells below after explicit approval.",
        "",
        "## Minimal targeted rerun matrix",
    ]
    if missing_cells:
        plan_lines.extend(f"- {cell}" for cell in missing_cells)
    else:
        plan_lines.append("- No missing cells for the currently supported bounded slice; run additional cells only to upgrade to STRONG.")
    plan_lines.extend(
        [
            "",
            "Required cell shape: WebArena Verified, hook_a_only and hook_b_only, clean baseline and promoted stressed regime, fixed budget, paired seed, vLLM/SGLang where possible, terminal outcome, pass outcome, reward_auc_over_wallclock, complete driver metadata, manifest/release binding, evidence_validation_pass=true.",
            "",
            "## Command skeleton for later approval",
            "",
            "Do not run this until live host availability and explicit rerun approval are both present.",
            "",
            "```bash",
            "export NIPS_CANONICAL_FINAL_ROOT=artifact_release_root",
            "export WEBAV_VERIFIED_ROOT=<live_host_verified_root>",
            "export NIPS_WEBARENA_VERIFIED_ROOT=<live_host_verified_root>",
            "export WA_SHOPPING=<anonymous_or_internal_live_host_url>",
            "export WA_REDDIT=<anonymous_or_internal_live_host_url>",
            "export WA_GITLAB=<anonymous_or_internal_live_host_url>",
            "",
            "# Minimal matrix only: WebArena Verified x {hook_a_only,hook_b_only} x {clean_baseline,promoted_stressed}",
            "# x fixed budget x paired seeds x {vLLM,SGLang where available}.",
            "# The concrete runner should write only calibration/decision-sensitive closeout rows first,",
            "# then this audit must be rerun before any paper-facing claim is written.",
            "bash scripts/final/materialize_decision_sensitive_slice_v1.sh",
            "```",
        ]
    )
    (out_dir / "targeted_rerun_plan.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
    return AuditResult(len(admitted) + len(blocked), len(admitted), len(blocked), claim_level, target_required, summary_path)


def audit_decision_sensitive_slice(root: Path, out_dir: Path, discovery_mode: str = "compact_first") -> AuditResult:
    candidates, inspected = build_candidates(root, discovery_mode=discovery_mode)
    if candidates.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        admitted = pd.DataFrame()
        blocked = pd.DataFrame()
        return write_outputs(out_dir, admitted, blocked, pd.DataFrame(), inspected, "NONE", {"reason": "no_candidate_rows_found"})
    admitted, blocked = apply_admission(candidates, root)
    claim_level, claim_detail, comparison = compute_claim_level(admitted)
    return write_outputs(out_dir, admitted, blocked, comparison, inspected, claim_level, claim_detail)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=Path(os.environ.get("NIPS_CANONICAL_FINAL_ROOT", "")))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--discovery-mode", choices=["compact_first", "known_only"], default="compact_first")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.canonical_root:
        raise SystemExit("--canonical-root or NIPS_CANONICAL_FINAL_ROOT is required")
    out_dir = args.out_dir or args.canonical_root / "decision_sensitive_admission"
    result = audit_decision_sensitive_slice(args.canonical_root, out_dir, discovery_mode=args.discovery_mode)
    print(f"admitted_row_count={result.admitted_count}")
    print(f"blocked_row_count={result.blocked_count}")
    print(f"supported_claim_level={result.supported_claim_level}")
    print(f"decision_slice_summary={result.summary_path}")
    print(f"targeted_rerun_required={'yes' if result.targeted_rerun_required else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
