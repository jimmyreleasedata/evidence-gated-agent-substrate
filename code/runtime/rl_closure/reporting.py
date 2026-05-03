"""Reporting helpers for the RL closure appendix case study."""

from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable
import csv
import json
import math
import os
import platform
import subprocess


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def summarize_numeric(values: Iterable[float]) -> dict[str, float]:
    series = [float(value) for value in values]
    if not series:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": mean(series),
        "std": pstdev(series) if len(series) > 1 else 0.0,
    }


def safe_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100.0


def detect_gpu_info() -> dict:
    try:
        output = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
    except OSError:
        output = ""
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    return {"gpu_count": len(rows), "gpus": rows}


def build_compute_resources(block_timings: list[dict]) -> dict:
    mem_total_kb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
                break
    gpu_info = detect_gpu_info()
    total_runtime_s = sum(float(block.get("duration_s", 0.0)) for block in block_timings)
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total_kb": mem_total_kb,
        **gpu_info,
        "backend_blocks": block_timings,
        "total_runtime_s": total_runtime_s,
    }


def render_paper_mapping(
    path: Path,
    report_root: Path,
    summary: dict,
    empty_outputs: list[str],
) -> Path:
    text = f"""# RL Closure Paper Mapping

## Output Paths

- `backend_compatibility.csv`: `{report_root / 'backend_compatibility.csv'}`
- `hook_ablation.csv`: `{report_root / 'hook_ablation.csv'}`
- `controller_trace.csv`: `{report_root / 'controller_trace.csv'}`
- `sample_filter_reasons.csv`: `{report_root / 'sample_filter_reasons.csv'}`
- `training_curve.csv`: `{report_root / 'training_curve.csv'}`
- `run_metadata.json`: `{report_root / 'run_metadata.json'}`
- `compute_resources.json`: `{report_root / 'compute_resources.json'}`

## Appendix Case-Study Summary

- Framework: `veRL`
- Model: `{summary['model_id']}`
- Backends: `{', '.join(summary['backends'])}`
- Main workload: `swe_gym`
- Optional sanity workload: `webarena_verified`
- Hook A best valid-sample rate: `{summary['hook_a_valid_rate']:.3f}`
- Hook B best controller efficiency: `{summary['hook_b_efficiency']:.3f}`
- Hook A+B best utility: `{summary['hook_ab_utility']:.3f}`

## Discussion Paragraph

The appendix closure shows that the benchmark-suite telemetry can be consumed by one external RL-targeted downstream harness without altering the suite’s core benchmark semantics. The same dense Qwen model was exercised under both `vLLM` and `SGLang`, while the downstream controller used suite-native task summaries and verifier signals to implement sample filtering and adaptive scheduling.

## Sparse Or Empty Outputs

{chr(10).join(f"- `{item}`" for item in empty_outputs) if empty_outputs else "- none"}
"""
    path.write_text(text, encoding="utf-8")
    return path
