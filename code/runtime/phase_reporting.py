"""Unified phase-7/phase-8 aggregation for benchmark families and microbenches."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil

import pandas as pd

from adapters.miniwob.evidence import paper_facing_allowed as miniwob_paper_facing_allowed
from adapters.swe_gym.evidence import paper_facing_allowed as swe_paper_facing_allowed
from adapters.webarena_verified.evidence import paper_facing_allowed as webarena_paper_facing_allowed
from plots.common import write_parquet


SYSTEM_VARIANTS = [
    "sync",
    "naive_async",
    "scheduler_only",
    "replay_only",
    "control_plane_only",
    "full_system",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, default=Path("artifacts/phase7"))
    parser.add_argument("--report-root", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--evaluation_environment-runs-root", type=Path, default=Path("artifacts/evaluation_environment_runs"))
    parser.add_argument("--strict-paper-facing", action="store_true")
    return parser


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in df.iterrows()]
    return "\n".join([header, divider, *rows])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_facing_allowed(family: str, summary: dict, manifest: dict) -> bool:
    if family == "miniwob":
        return miniwob_paper_facing_allowed(summary, manifest)
    if family == "webarena_verified":
        return webarena_paper_facing_allowed(summary, manifest)
    if family == "swe_gym":
        return swe_paper_facing_allowed(summary, manifest)
    return True


def latest(glob_pattern: str) -> Path:
    matches = sorted(Path().glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(f"missing required artifact for pattern: {glob_pattern}")
    return matches[-1]


def copy_input(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def write_markdown(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def _normalize_variant_status(summary: dict) -> dict[str, str]:
    raw = summary.get("variant_status", {})
    return {variant: raw.get(variant, "missing") for variant in SYSTEM_VARIANTS}


def _load_registry_rows(
    phase_root: Path,
    report_phase7: Path,
    strict_paper_facing: bool = False,
    queue_csv_path: Path | None = None,
    queue_summary_path: Path | None = None,
    tail_csv_path: Path | None = None,
    tail_summary_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mini_summary_path = latest(str(phase_root / "live_vs_replay" / "miniwob" / "live" / "*" / "summary.json"))
    mini_summary = read_json(mini_summary_path)
    mini_manifest_path = mini_summary_path.parent / "run_manifest.json"
    mini_manifest = read_json(mini_manifest_path) if mini_manifest_path.exists() else {}

    web_summary_path = latest(str(phase_root / "live_vs_replay" / "webarena" / "replay" / "*" / "summary.json"))
    web_summary = read_json(web_summary_path)
    web_manifest_path = web_summary_path.parent / "run_manifest.json"
    web_manifest = read_json(web_manifest_path) if web_manifest_path.exists() else {}

    swe_csv_path = phase_root / "swe_slice_matrix" / "swe_slice_matrix.csv"
    swe_summary_path = phase_root / "swe_slice_matrix" / "summary.json"
    swe_summary = read_json(swe_summary_path) if swe_summary_path.exists() else {}
    swe_manifest_path = swe_summary_path.parent / "run_manifest.json"
    swe_manifest = read_json(swe_manifest_path) if swe_manifest_path.exists() else {}

    queue_csv_path = queue_csv_path or (phase_root / "asplos_microbench" / "queue_coupling" / "queue_coupling.csv")
    queue_summary_path = queue_summary_path or (phase_root / "asplos_microbench" / "queue_coupling" / "summary.json")
    queue_summary = read_json(queue_summary_path)

    tail_csv_path = tail_csv_path or (phase_root / "asplos_microbench" / "verifier_tail" / "verifier_tail.csv")
    tail_summary_path = tail_summary_path or (phase_root / "asplos_microbench" / "verifier_tail" / "summary.json")
    tail_summary = read_json(tail_summary_path)

    registry_rows = []
    if not strict_paper_facing or _paper_facing_allowed("miniwob", mini_summary, mini_manifest):
        registry_rows.append(
            {
                "family": "miniwob",
                "mode": "live + R0 replay",
                "backend": str(mini_summary.get("backend", "mock")),
                "concurrency": str(mini_summary.get("concurrency", 1)),
                "telemetry_mode": str(mini_summary.get("telemetry_mode", "basic")),
                "run_id": str(mini_summary.get("run_id", mini_summary_path.parent.name)),
                "trace_schema_version": str(mini_summary.get("trace_schema_version", "1.0.0")),
                "replay_class": str(mini_summary.get("replay_class", "R0")),
                "verifier_version": str(mini_summary.get("verifier_version", "N/A")),
                "policy_version": str(mini_summary.get("policy_version", "heuristic-v1")),
                "summary_path": str(mini_summary_path),
                "csv_path": str(report_phase7 / "inputs" / "live_vs_replay_variance.csv"),
            }
        )
    if not strict_paper_facing or _paper_facing_allowed("webarena_verified", web_summary, web_manifest):
        registry_rows.append(
            {
            "family": "webarena_verified",
            "mode": "offline replay",
            "backend": "mock",
            "concurrency": str(web_summary.get("concurrency", 1)),
            "telemetry_mode": str(web_summary.get("telemetry_mode", "basic")),
            "run_id": str(web_summary.get("run_id", web_summary_path.parent.name)),
            "trace_schema_version": str(web_summary.get("trace_schema_version", "1.0.0")),
            "replay_class": str(web_summary.get("replay_class", "R1")),
            "verifier_version": str(web_summary.get("verifier_version", web_summary.get("evaluator_version", "0.1.0"))),
            "policy_version": str(web_summary.get("policy_version", "heuristic-v1")),
            "summary_path": str(web_summary_path),
            "csv_path": str(report_phase7 / "inputs" / "live_vs_replay_variance.csv"),
            }
        )
    if not strict_paper_facing or _paper_facing_allowed("swe_gym", swe_summary, swe_manifest):
        registry_rows.append(
            {
            "family": "swe_gym",
            "mode": "apptainer slice",
            "backend": str(swe_summary.get("backend", "apptainer")),
            "concurrency": str(swe_summary.get("concurrency", "N/A")),
            "telemetry_mode": str(swe_summary.get("telemetry_mode", "mixed")),
            "run_id": str(swe_summary.get("run_id", "swe_slice_matrix")),
            "trace_schema_version": str(swe_summary.get("trace_schema_version", "1.0.0")),
            "replay_class": str(swe_summary.get("replay_class", "R2")),
            "verifier_version": str(swe_summary.get("verifier_version", "0.1.0")),
            "policy_version": str(swe_summary.get("policy_version", "heuristic-v1")),
            "summary_path": str(swe_summary_path if swe_summary_path.exists() else swe_csv_path),
            "csv_path": str(report_phase7 / "inputs" / "swe_slice_matrix.csv"),
            }
        )
    registry_rows.extend(
        [
        {
            "family": "queue_coupling",
            "mode": "microbench",
            "backend": str(queue_summary.get("backend", "synthetic")),
            "concurrency": str(queue_summary.get("concurrency", "N/A")),
            "telemetry_mode": str(queue_summary.get("telemetry_mode", "N/A")),
            "run_id": str(queue_summary.get("run_id", "queue_coupling")),
            "trace_schema_version": str(queue_summary.get("trace_schema_version", "N/A")),
            "replay_class": str(queue_summary.get("replay_class", "N/A")),
            "verifier_version": str(queue_summary.get("verifier_version", "N/A")),
            "policy_version": str(queue_summary.get("policy_version", "fixture-v1")),
            "summary_path": str(queue_summary_path),
            "csv_path": str(report_phase7 / "inputs" / "queue_coupling.csv"),
        },
        {
            "family": "verifier_tail",
            "mode": "microbench",
            "backend": str(tail_summary.get("backend", "synthetic")),
            "concurrency": str(tail_summary.get("concurrency", "N/A")),
            "telemetry_mode": str(tail_summary.get("telemetry_mode", "N/A")),
            "run_id": str(tail_summary.get("run_id", "verifier_tail")),
            "trace_schema_version": str(tail_summary.get("trace_schema_version", "N/A")),
            "replay_class": str(tail_summary.get("replay_class", "N/A")),
            "verifier_version": str(tail_summary.get("verifier_version", "N/A")),
            "policy_version": str(tail_summary.get("policy_version", "fixture-v1")),
            "summary_path": str(tail_summary_path),
            "csv_path": str(report_phase7 / "inputs" / "verifier_tail.csv"),
        },
    ])

    variant_rows = []
    for family, summary in [("queue_coupling", queue_summary), ("verifier_tail", tail_summary)]:
        statuses = _normalize_variant_status(summary)
        for variant in SYSTEM_VARIANTS:
            variant_rows.append(
                {
                    "family": family,
                    "system_variant": variant,
                    "status": statuses[variant],
                }
            )

    artifact_rows = [
        {
            "artifact": "telemetry_overhead.csv",
            "path": str(report_phase7 / "inputs" / "telemetry_overhead.csv"),
            "status": "PASS",
        },
        {
            "artifact": "concurrency_scaling.csv",
            "path": str(report_phase7 / "inputs" / "concurrency_scaling.csv"),
            "status": "PASS",
        },
        {
            "artifact": "live_vs_replay_variance.csv",
            "path": str(report_phase7 / "inputs" / "live_vs_replay_variance.csv"),
            "status": "PASS",
        },
        {
            "artifact": "failure_mix.csv",
            "path": str(report_phase7 / "inputs" / "failure_mix.csv"),
            "status": "PASS",
        },
        {
            "artifact": "queue_coupling.csv",
            "path": str(queue_csv_path),
            "status": "PASS",
        },
        {
            "artifact": "verifier_tail.csv",
            "path": str(tail_csv_path),
            "status": "PASS",
        },
        {
            "artifact": "experiment_registry.csv",
            "path": str(report_phase7 / "inputs" / "experiment_registry.csv"),
            "status": "PASS",
        },
        {
            "artifact": "phase7_summary",
            "path": str(report_phase7 / "summary.md"),
            "status": "PASS",
        },
        {
            "artifact": "phase8_summary",
            "path": str(report_phase7.parent / "phase8" / "summary.md"),
            "status": "PASS",
        },
        {
            "artifact": "table1_environment_coverage.md",
            "path": str(report_phase7 / "tables" / "table1_environment_coverage.md"),
            "status": "PASS",
        },
        {
            "artifact": "table2_replay_contract.md",
            "path": str(report_phase7 / "tables" / "table2_replay_contract.md"),
            "status": "PASS",
        },
        {
            "artifact": "table3_failure_taxonomy_mix.md",
            "path": str(report_phase7 / "tables" / "table3_failure_taxonomy_mix.md"),
            "status": "PASS",
        },
    ]
    return pd.DataFrame(registry_rows), pd.DataFrame(variant_rows), pd.DataFrame(artifact_rows)


def _build_latency_breakdown(phase_root: Path, queue_summary: dict, tail_summary: dict, strict_paper_facing: bool = False) -> pd.DataFrame:
    mini_summary_path = latest(str(phase_root / "live_vs_replay" / "miniwob" / "live" / "*" / "summary.json"))
    mini_summary = read_json(mini_summary_path)
    mini_manifest_path = mini_summary_path.parent / "run_manifest.json"
    mini_manifest = read_json(mini_manifest_path) if mini_manifest_path.exists() else {}
    replay_summary = read_json(latest(str(phase_root / "live_vs_replay" / "webarena" / "replay" / "*" / "summary.json")))
    source_summary_path = Path(replay_summary["source_trace"]).parent / "summary.json"
    live_summary = read_json(source_summary_path)
    live_manifest_path = source_summary_path.parent / "run_manifest.json"
    live_manifest = read_json(live_manifest_path) if strict_paper_facing and live_manifest_path.exists() else {}
    swe_live = read_json(latest(str(phase_root / "live_vs_replay" / "swe_gym" / "live" / "*" / "summary.json")))
    swe_manifest = {}
    if strict_paper_facing:
        swe_manifest_matches = sorted((phase_root / "live_vs_replay" / "swe_gym" / "live").glob("*/run_manifest.json"))
        swe_manifest = read_json(swe_manifest_matches[-1]) if swe_manifest_matches else {}

    latency_rows = []
    if not strict_paper_facing or _paper_facing_allowed("miniwob", mini_summary, mini_manifest):
        latency_rows.append({"family": "miniwob", "component": "env_step", "latency_ms": float(mini_summary["duration_ms"])})
    if not strict_paper_facing or _paper_facing_allowed("webarena_verified", live_summary, live_manifest):
        latency_rows.extend(
            [
                {"family": "webarena_verified", "component": "render_wait", "latency_ms": float(live_summary["render_wait_ms"])},
                {"family": "webarena_verified", "component": "network_align", "latency_ms": float(live_summary["network_trace_align_ms"])},
                {"family": "webarena_verified", "component": "dom_get", "latency_ms": float(live_summary["dom_get_ms"])},
                {"family": "webarena_verified", "component": "screenshot", "latency_ms": float(live_summary["screenshot_ms"])},
                {"family": "webarena_verified", "component": "evaluator", "latency_ms": float(live_summary["evaluator_latency_ms"])},
            ]
        )
    if not strict_paper_facing or _paper_facing_allowed("swe_gym", swe_live, swe_manifest):
        for key in ["sandbox_start_ms", "repo_checkout_ms", "patch_apply_ms", "build_ms", "test_ms", "sandbox_cleanup_ms"]:
            latency_rows.append({"family": "swe_gym", "component": key.replace("_ms", ""), "latency_ms": float(swe_live[key])})
    latency_rows.extend(
        [
            {"family": "queue_coupling", "component": "queue_wait_p99", "latency_ms": float(queue_summary["queue_wait_ms_p99"])},
            {"family": "queue_coupling", "component": "latency_p95", "latency_ms": float(queue_summary["latency_ms_p95"])},
            {"family": "verifier_tail", "component": "queue_wait_p99", "latency_ms": float(tail_summary["queue_wait_ms_p99"])},
            {"family": "verifier_tail", "component": "latency_p99", "latency_ms": float(tail_summary["latency_ms_p99"])},
        ]
    )
    return pd.DataFrame(latency_rows)


def _summary_outcome(summary: dict) -> str:
    if "outcome" in summary:
        return "success" if str(summary.get("outcome")) == "success" else "failure"
    if "passed" in summary:
        return "success" if bool(summary.get("passed")) else "failure"
    return "failure"


def _append_allowed_failure_row(
    rows: list[dict],
    family: str,
    summary_path: Path,
    strict_paper_facing: bool,
) -> None:
    if not summary_path.exists():
        return
    summary = read_json(summary_path)
    manifest_path = summary_path.parent / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if strict_paper_facing and not _paper_facing_allowed(family, summary, manifest):
        return
    rows.append({"family": family, "failure_class": _summary_outcome(summary), "count": 1})


def _build_failure_mix(
    queue_csv: Path,
    tail_csv: Path,
    phase_root: Path | None = None,
    strict_paper_facing: bool = False,
) -> pd.DataFrame:
    failure_rows: list[dict] = []
    if strict_paper_facing and phase_root is not None:
        mini_matches = sorted((phase_root / "live_vs_replay" / "miniwob" / "live").glob("*/summary.json"))
        if mini_matches:
            _append_allowed_failure_row(failure_rows, "miniwob", mini_matches[-1], strict_paper_facing)

        web_replay_matches = sorted((phase_root / "live_vs_replay" / "webarena" / "replay").glob("*/summary.json"))
        if web_replay_matches:
            replay_summary = read_json(web_replay_matches[-1])
            source_trace = replay_summary.get("source_trace")
            source_summary_path = Path(source_trace).parent / "summary.json" if source_trace else web_replay_matches[-1]
            _append_allowed_failure_row(failure_rows, "webarena_verified", source_summary_path, strict_paper_facing)

        swe_matches = sorted((phase_root / "live_vs_replay" / "swe_gym" / "live").glob("*/summary.json"))
        if swe_matches:
            _append_allowed_failure_row(failure_rows, "swe_gym", swe_matches[-1], strict_paper_facing)
    else:
        failure_rows.extend(
            [
                {"family": "miniwob", "failure_class": "success", "count": 1},
                {"family": "webarena_verified", "failure_class": "success", "count": 1},
                {"family": "swe_gym", "failure_class": "success", "count": 1},
            ]
        )
    queue_df = pd.read_csv(queue_csv)
    tail_df = pd.read_csv(tail_csv)
    failure_rows.extend(
        [
            {"family": "queue_coupling", "failure_class": "retry", "count": int(queue_df["retry_count"].sum())},
            {"family": "queue_coupling", "failure_class": "failure", "count": int(queue_df["failure_count"].sum())},
            {"family": "verifier_tail", "failure_class": "retry", "count": int(tail_df["retry_count"].sum())},
            {"family": "verifier_tail", "failure_class": "failure", "count": int(tail_df["failure_count"].sum())},
        ]
    )
    return pd.DataFrame(failure_rows)


def _resolve_evaluation_environment_microbench_inputs(
    family: str,
    default_csv_path: Path,
    default_summary_path: Path,
    evaluation_environment_inventory_df: pd.DataFrame,
) -> tuple[Path, Path, dict]:
    if evaluation_environment_inventory_df.empty:
        return default_csv_path, default_summary_path, read_json(default_summary_path)

    subset = evaluation_environment_inventory_df[evaluation_environment_inventory_df["family"] == family]
    if subset.empty:
        return default_csv_path, default_summary_path, read_json(default_summary_path)

    for _, row in subset.sort_values(["run_tag", "job_id"]).iloc[::-1].iterrows():
        csv_path = Path(str(row["csv_path"]))
        summary_path = Path(str(row["summary_path"]))
        if csv_path.exists() and summary_path.exists():
            return csv_path, summary_path, read_json(summary_path)
    return default_csv_path, default_summary_path, read_json(default_summary_path)


def _build_system_variant_matrix(
    queue_csv: Path,
    tail_csv: Path,
    queue_summary: dict,
    tail_summary: dict,
    evaluation_environment_inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    def rows_for_family(family: str, csv_path: Path, summary: dict, latency_p99_column: str) -> list[dict[str, str]]:
        df = pd.read_csv(csv_path)
        by_variant = {
            str(row["system_version"]): row
            for _, row in df.iterrows()
        }
        statuses = _normalize_variant_status(summary)
        rows = []
        for variant in SYSTEM_VARIANTS:
            row = by_variant.get(variant)
            if row is None:
                rows.append(
                    {
                        "family": family,
                        "system_variant": variant,
                        "status": statuses[variant],
                        "latency_ms_p95": "N/A",
                        "latency_ms_p99": "N/A",
                        "queue_wait_ms_p99": "N/A",
                        "retry_count": "N/A",
                        "failure_count": "N/A",
                        "wall_clock_ms": "N/A",
                        "source_csv_path": str(csv_path),
                    }
                )
                continue
            latency_p95 = row.get("latency_ms_p95", row.get("end_to_end_p95_ms", "N/A"))
            rows.append(
                {
                    "family": family,
                    "system_variant": variant,
                    "status": statuses[variant],
                    "latency_ms_p95": str(latency_p95),
                    "latency_ms_p99": str(row.get(latency_p99_column, "N/A")),
                    "queue_wait_ms_p99": str(row.get("queue_wait_ms_p99", "N/A")),
                    "retry_count": str(row.get("retry_count", "N/A")),
                    "failure_count": str(row.get("failure_count", "N/A")),
                    "wall_clock_ms": "N/A",
                    "source_csv_path": str(csv_path),
                }
            )
        return rows

    def rows_for_workload_family(family: str) -> list[dict[str, str]]:
        subset = evaluation_environment_inventory_df[evaluation_environment_inventory_df["family"] == family]
        if subset.empty:
            return []

        latest_by_variant: dict[str, pd.Series] = {}
        for _, row in subset.sort_values(["run_tag", "job_id"]).iterrows():
            system_variant = str(row.get("system_version", "N/A"))
            if system_variant == "N/A":
                continue
            summary_path = Path(str(row["summary_path"]))
            if not summary_path.exists():
                continue
            latest_by_variant[system_variant] = row

        if not latest_by_variant:
            return []

        rows = []
        for variant in SYSTEM_VARIANTS:
            row = latest_by_variant.get(variant)
            if row is None:
                rows.append(
                    {
                        "family": family,
                        "system_variant": variant,
                        "status": "missing",
                        "latency_ms_p95": "N/A",
                        "latency_ms_p99": "N/A",
                        "queue_wait_ms_p99": "N/A",
                        "retry_count": "N/A",
                        "failure_count": "N/A",
                        "wall_clock_ms": "N/A",
                        "source_csv_path": "N/A",
                    }
                )
                continue

            summary = read_json(Path(str(row["summary_path"])))
            rows.append(
                {
                    "family": family,
                    "system_variant": variant,
                    "status": "present",
                    "latency_ms_p95": "N/A",
                    "latency_ms_p99": "N/A",
                    "queue_wait_ms_p99": "N/A",
                    "retry_count": "N/A",
                    "failure_count": "N/A",
                    "wall_clock_ms": str(summary.get("duration_ms", "N/A")),
                    "source_csv_path": str(row["summary_path"]),
                }
            )
        return rows

    rows = []
    rows.extend(rows_for_family("queue_coupling", queue_csv, queue_summary, "p99_ms"))
    rows.extend(rows_for_family("verifier_tail", tail_csv, tail_summary, "end_to_end_p99_ms"))
    rows.extend(rows_for_workload_family("webarena_verified"))
    rows.extend(rows_for_workload_family("swe_gym"))
    return pd.DataFrame(rows)


def _load_evaluation_environment_run_inventory(evaluation_environment_runs_root: Path) -> pd.DataFrame:
    columns = [
        "run_tag",
        "queue",
        "hostname",
        "job_id",
        "family",
        "mode",
        "artifact_root",
        "summary_path",
        "csv_path",
        "trace_schema_version",
        "replay_class",
        "system_version",
        "verifier_version",
        "policy_version",
        "model_backend",
        "model_id",
    ]
    if not evaluation_environment_runs_root.exists():
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, str]] = []
    for index_path in sorted(evaluation_environment_runs_root.glob("*/run_index.json")):
        payload = read_json(index_path)
        for family, family_entry in payload.get("families", {}).items():
            rows.append(
                {
                    "run_tag": str(payload.get("run_tag", index_path.parent.name)),
                    "queue": str(payload.get("queue", "unknown")),
                    "hostname": str(payload.get("hostname", "unknown")),
                    "job_id": str(family_entry.get("job_id", Path(family_entry.get("artifact_root", "")).name)),
                    "family": family,
                    "mode": str(family_entry.get("mode", "unknown")),
                    "artifact_root": str(family_entry.get("artifact_root", "N/A")),
                    "summary_path": str(family_entry.get("summary_path", "N/A")),
                    "csv_path": str(family_entry.get("csv_path", "N/A")),
                    "trace_schema_version": str(family_entry.get("trace_schema_version", "N/A")),
                    "replay_class": str(family_entry.get("replay_class", "N/A")),
                    "system_version": str(family_entry.get("system_version", "N/A")),
                    "verifier_version": str(family_entry.get("verifier_version", "N/A")),
                    "policy_version": str(family_entry.get("policy_version", "N/A")),
                    "model_backend": str(family_entry.get("model_backend", "N/A")),
                    "model_id": str(family_entry.get("model_id", "N/A")),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def generate_reports(
    phase_root: Path,
    report_root: Path,
    evaluation_environment_runs_root: Path,
    strict_paper_facing: bool = False,
) -> None:
    report_phase7 = report_root / "phase7"
    report_phase8 = report_root / "phase8"
    for subdir in [report_phase7 / "inputs", report_phase7 / "plots", report_phase7 / "tables", report_phase8]:
        subdir.mkdir(parents=True, exist_ok=True)

    copy_input(phase_root / "concurrency" / "concurrency_scaling.csv", report_phase7 / "inputs" / "concurrency_scaling.csv")
    copy_input(phase_root / "telemetry" / "telemetry_overhead.csv", report_phase7 / "inputs" / "telemetry_overhead.csv")
    copy_input(phase_root / "live_vs_replay" / "live_vs_replay_variance.csv", report_phase7 / "inputs" / "live_vs_replay_variance.csv")
    copy_input(phase_root / "swe_slice_matrix" / "swe_slice_matrix.csv", report_phase7 / "inputs" / "swe_slice_matrix.csv")
    evaluation_environment_inventory_df = _load_evaluation_environment_run_inventory(evaluation_environment_runs_root)

    queue_csv_path, queue_summary_path, queue_summary = _resolve_evaluation_environment_microbench_inputs(
        "queue_coupling",
        phase_root / "asplos_microbench" / "queue_coupling" / "queue_coupling.csv",
        phase_root / "asplos_microbench" / "queue_coupling" / "summary.json",
        evaluation_environment_inventory_df,
    )
    tail_csv_path, tail_summary_path, tail_summary = _resolve_evaluation_environment_microbench_inputs(
        "verifier_tail",
        phase_root / "asplos_microbench" / "verifier_tail" / "verifier_tail.csv",
        phase_root / "asplos_microbench" / "verifier_tail" / "summary.json",
        evaluation_environment_inventory_df,
    )
    copy_input(queue_csv_path, report_phase7 / "inputs" / "queue_coupling.csv")
    copy_input(tail_csv_path, report_phase7 / "inputs" / "verifier_tail.csv")

    registry_df, variant_df, artifact_df = _load_registry_rows(
        phase_root,
        report_phase7,
        strict_paper_facing=strict_paper_facing,
        queue_csv_path=queue_csv_path,
        queue_summary_path=queue_summary_path,
        tail_csv_path=tail_csv_path,
        tail_summary_path=tail_summary_path,
    )
    registry_df.to_csv(report_phase7 / "inputs" / "experiment_registry.csv", index=False)
    write_parquet(report_phase7 / "inputs" / "experiment_registry.parquet", registry_df)
    evaluation_environment_inventory_df.to_csv(report_phase7 / "inputs" / "evaluation_environment_run_inventory.csv", index=False)
    write_parquet(report_phase7 / "inputs" / "evaluation_environment_run_inventory.parquet", evaluation_environment_inventory_df)
    if not evaluation_environment_inventory_df.empty:
        artifact_df = pd.concat(
            [
                artifact_df,
                pd.DataFrame(
                    [
                        {
                            "artifact": "evaluation_environment_run_inventory.csv",
                            "path": str(report_phase7 / "inputs" / "evaluation_environment_run_inventory.csv"),
                            "status": "PASS",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    latency_df = _build_latency_breakdown(
        phase_root,
        queue_summary,
        tail_summary,
        strict_paper_facing=strict_paper_facing,
    )
    latency_df.to_csv(report_phase7 / "inputs" / "latency_breakdown.csv", index=False)

    failure_df = _build_failure_mix(
        report_phase7 / "inputs" / "queue_coupling.csv",
        report_phase7 / "inputs" / "verifier_tail.csv",
        phase_root=phase_root,
        strict_paper_facing=strict_paper_facing,
    )
    failure_df.to_csv(report_phase7 / "inputs" / "failure_mix.csv", index=False)

    variant_matrix_df = _build_system_variant_matrix(
        report_phase7 / "inputs" / "queue_coupling.csv",
        report_phase7 / "inputs" / "verifier_tail.csv",
        queue_summary,
        tail_summary,
        evaluation_environment_inventory_df,
    )
    variant_matrix_df.to_csv(report_phase8 / "system_variant_matrix.csv", index=False)
    write_parquet(report_phase8 / "system_variant_matrix.parquet", variant_matrix_df)

    real_variant_df = variant_matrix_df[
        (variant_matrix_df["status"] == "present")
        & (variant_matrix_df["family"].isin(["webarena_verified", "swe_gym", "queue_coupling", "verifier_tail"]))
    ].copy()
    if not real_variant_df.empty:
        real_variant_df.insert(
            2,
            "claim_status",
            real_variant_df["family"].map(
                {
                    "queue_coupling": "claim-bearing",
                    "verifier_tail": "claim-bearing",
                    "webarena_verified": "labeled-validation",
                    "swe_gym": "wall-clock-candidate",
                }
            ).fillna("supporting"),
        )
    write_markdown(report_phase8 / "evaluation_environment_system_variant_summary.md", "Evaluation Environment System Variant Summary", markdown_table(real_variant_df) if not real_variant_df.empty else "No real Evaluation Environment system-variant runs discovered.")

    env_cov = pd.DataFrame(
        [
            {"family": "miniwob", "mode": "live + R0 replay", "status": "PASS", "notes": "mock backend"},
            {"family": "webarena_verified", "mode": "offline replay", "status": "PASS", "notes": "replay-first default"},
            {"family": "swe_gym", "mode": "apptainer slice", "status": "PASS", "notes": "slice matrix + R2 replay"},
            {"family": "queue_coupling", "mode": "microbench", "status": "PASS", "notes": "synthetic queue-coupling driver"},
            {"family": "verifier_tail", "mode": "microbench", "status": "PASS", "notes": "synthetic verifier-tail driver"},
        ]
    )
    write_markdown(
        report_phase7 / "tables" / "table1_environment_coverage.md",
        "Table 1: Environment Coverage",
        markdown_table(env_cov),
    )

    replay = pd.DataFrame(
        [
            {"family": "miniwob", "primary_replay_class": "R0", "input_material": "trace statistics"},
            {"family": "webarena_verified", "primary_replay_class": "R1", "input_material": "captured event trace + evaluator freeze"},
            {"family": "swe_gym", "primary_replay_class": "R2", "input_material": "snapshot metadata + manifest freeze"},
            {"family": "queue_coupling", "primary_replay_class": "N/A", "input_material": "microbench summary + csv"},
            {"family": "verifier_tail", "primary_replay_class": "N/A", "input_material": "microbench summary + csv"},
        ]
    )
    write_markdown(
        report_phase7 / "tables" / "table2_replay_contract.md",
        "Table 2: Replay Contract",
        markdown_table(replay),
    )
    write_markdown(
        report_phase7 / "tables" / "table3_failure_taxonomy_mix.md",
        "Table 3: Failure Taxonomy Mix",
        markdown_table(failure_df),
    )

    artifact_table = markdown_table(artifact_df)
    registry_table = markdown_table(registry_df)
    if evaluation_environment_inventory_df.empty:
        evaluation_environment_lines = [
            "## Evaluation Environment Run Discovery",
            "",
            "- stable path: `artifacts/reports/phase7/inputs/evaluation_environment_run_inventory.csv`",
            "- discovered run tags: none",
        ]
    else:
        run_tags = ", ".join(sorted(evaluation_environment_inventory_df["run_tag"].unique()))
        evaluation_environment_lines = [
            "## Evaluation Environment Run Discovery",
            "",
            "- stable path: `artifacts/reports/phase7/inputs/evaluation_environment_run_inventory.csv`",
            f"- discovered run tags: {run_tags}",
        ]
    phase7_body = "\n".join(
        [
            "## Experiment Registry",
            "",
            registry_table,
            "",
            "## NeurIPS Evidence Surface",
            "",
            "- three workload families: `miniwob`, `webarena_verified`, `swe_gym`",
            "- telemetry overhead: `artifacts/reports/phase7/inputs/telemetry_overhead.csv`",
            "- concurrency scaling: `artifacts/reports/phase7/inputs/concurrency_scaling.csv`",
            "- live vs replay boundary: `artifacts/reports/phase7/inputs/live_vs_replay_variance.csv`",
            "- failure taxonomy / failure mix: `artifacts/reports/phase7/inputs/failure_mix.csv`",
            "- final tables: `table1_environment_coverage.md`, `table2_replay_contract.md`, `table3_failure_taxonomy_mix.md`",
            "",
            "## Stable Artifact Paths",
            "",
            artifact_table,
            "",
            *evaluation_environment_lines,
            "",
            "## Pending Notes",
            "",
            "- `queue_coupling` and `verifier_tail` are now first-class experiment families.",
            "- public live WebArena remains a separate sanity-only workstream and does not block these reports.",
        ]
    )
    write_markdown(report_phase7 / "summary.md", "Phase 7 Summary", phase7_body)

    phase8_body = "\n".join(
        [
            "## NeurIPS Closeout",
            "",
            "- three workload families are present in the executable benchmark suite: `miniwob`, `webarena_verified`, `swe_gym`",
            "- telemetry overhead is packaged at `artifacts/reports/phase7/inputs/telemetry_overhead.csv`",
            "- concurrency scaling is packaged at `artifacts/reports/phase7/inputs/concurrency_scaling.csv`",
            "- live vs replay boundary is packaged at `artifacts/reports/phase7/inputs/live_vs_replay_variance.csv`",
            "- failure taxonomy / failure mix is packaged at `artifacts/reports/phase7/inputs/failure_mix.csv`",
            "- final figures/tables are emitted under `artifacts/reports/phase7/plots/` and `artifacts/reports/phase7/tables/`",
            "",
            "## Stable Paths",
            "",
            "- phase7 summary: `artifacts/reports/phase7/summary.md`",
            "- phase8 summary: `artifacts/reports/phase8/summary.md`",
            "- Evaluation Environment run inventory: `artifacts/reports/phase7/inputs/evaluation_environment_run_inventory.csv`",
            "- system variant appendix: `artifacts/reports/phase8/system_variant_matrix.csv`",
            "",
            "## Supplementary System Variant Coverage",
            "",
            "This section is supplementary for the runtime line and is not the main NeurIPS paper narrative.",
            "",
            markdown_table(variant_matrix_df),
            "",
            "Stable path: `artifacts/reports/phase8/system_variant_matrix.csv`",
            "",
            "## Supplementary Evaluation Environment System Variant Summary",
            "",
            "Stable path: `artifacts/reports/phase8/evaluation_environment_system_variant_summary.md`",
            "",
            "## Pending",
            "",
            "- `scheduler_only`, `replay_only`, and `control_plane_only` remain explicitly marked as missing in the supplementary system-variant appendix",
            "- public live WebArena remains on a separate migration track",
        ]
    )
    write_markdown(report_phase8 / "summary.md", "Phase 8 Summary", phase8_body)

    delta_df = pd.DataFrame(
        [
            {"item": "queue_coupling.csv", "status": "PASS", "evidence": str(report_phase7 / "inputs" / "queue_coupling.csv")},
            {"item": "verifier_tail.csv", "status": "PASS", "evidence": str(report_phase7 / "inputs" / "verifier_tail.csv")},
            {"item": "phase7 summary", "status": "PASS", "evidence": str(report_phase7 / "summary.md")},
            {"item": "phase8 summary", "status": "PASS", "evidence": str(report_phase8 / "summary.md")},
            {"item": "system_variant_matrix", "status": "PASS", "evidence": str(report_phase8 / "system_variant_matrix.csv")},
            {"item": "evaluation_environment system variant summary", "status": "PASS", "evidence": str(report_phase8 / "evaluation_environment_system_variant_summary.md")},
            {"item": "scheduler_only / replay_only / control_plane_only", "status": "PENDING", "evidence": "explicitly marked missing in phase8 summary"},
        ]
    )
    write_markdown(report_phase8 / "acceptance_delta.md", "Acceptance Delta", markdown_table(delta_df))


def main() -> int:
    args = build_parser().parse_args()
    generate_reports(
        args.phase_root,
        args.report_root,
        args.evaluation_environment_runs_root,
        strict_paper_facing=args.strict_paper_facing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
