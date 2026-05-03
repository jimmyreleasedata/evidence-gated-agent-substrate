#!/usr/bin/env bash
set -euo pipefail

ROOT=""
PLANNED_CELLS=""
REUSE_CANONICAL="true"
ALLOW_OPTIONAL_SEED2="false"
DRY_RUN="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --planned-cells) PLANNED_CELLS="$2"; shift 2 ;;
    --reuse-canonical) REUSE_CANONICAL="$2"; shift 2 ;;
    --allow-optional-seed2) ALLOW_OPTIONAL_SEED2="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${ROOT}" ]]; then
  echo "missing --root" >&2
  exit 2
fi
if [[ -z "${PLANNED_CELLS}" ]]; then
  PLANNED_CELLS="${ROOT}/decision_robustness_closeout_v1/manifests/decision_robustness_planned_cells.csv"
fi
if [[ ! -f "${PLANNED_CELLS}" ]]; then
  echo "planned cells not found: ${PLANNED_CELLS}" >&2
  exit 2
fi

CLOSEOUT_ROOT="${ROOT}/decision_robustness_closeout_v1"
LOG_DIR="${CLOSEOUT_ROOT}/logs"
RUNS_DIR="${CLOSEOUT_ROOT}/runs"
mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

PLANNER_PYTHON_BIN="${PLANNER_PYTHON_BIN:-ANON_HOME/.conda/envs/llm-env/bin/python}"
if [[ ! -x "${PLANNER_PYTHON_BIN}" ]]; then
  PLANNER_PYTHON_BIN="python3"
fi
PYTHON_BIN="${PYTHON_BIN:-/var/tmp/bounded_stronger_traffic_source_sanity_playwright_venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PLANNER_PYTHON_BIN}"
fi

export NIPS_CANONICAL_FINAL_ROOT="${NIPS_CANONICAL_FINAL_ROOT:-${ROOT}}"
if [[ -d "/var/tmp/bounded_stronger_traffic_source_sanity_playwright_browsers" && -z "${PLAYWRIGHT_BROWSERS_PATH:-}" ]]; then
  export PLAYWRIGHT_BROWSERS_PATH="/var/tmp/bounded_stronger_traffic_source_sanity_playwright_browsers"
fi

JOBS_CSV="${LOG_DIR}/planned_execution_jobs.csv"
SKIPPED_CSV="${LOG_DIR}/skipped_cells.csv"

"${PLANNER_PYTHON_BIN}" - "${ROOT}" "${PLANNED_CELLS}" "${REUSE_CANONICAL}" "${ALLOW_OPTIONAL_SEED2}" "${JOBS_CSV}" "${SKIPPED_CSV}" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
planned_path = Path(sys.argv[2])
reuse_canonical = sys.argv[3].lower() == "true"
allow_seed2 = sys.argv[4].lower() == "true"
jobs_path = Path(sys.argv[5])
skipped_path = Path(sys.argv[6])
closeout_root = root / "decision_robustness_closeout_v1"
is_smoke = "smoke" in planned_path.name
planned = pd.read_csv(planned_path)
reuse_path = closeout_root / "manifests" / "reuse_decisions.csv"
reuse = pd.read_csv(reuse_path) if reuse_path.exists() else pd.DataFrame()
reuse_ids = set(reuse.loc[reuse.get("reuse_decision", "").eq("reuse"), "planned_cell_id"].astype(str)) if not reuse.empty else set()

jobs: dict[tuple[str, int, int], set[str]] = {}
skipped: list[dict[str, str]] = []
for _, row in planned.iterrows():
    planned_cell_id = str(row["planned_cell_id"])
    seed = int(row["seed"])
    budget = int(row["budget"])
    reason = ""
    if seed == 2 and not allow_seed2:
        reason = "optional_seed2_not_enabled"
    elif str(row.get("executable", "true")).lower() != "true":
        reason = str(row.get("notes") or "not_executable")
    elif "budget_exceeds_discovered_task_count" in str(row.get("notes") or ""):
        reason = "budget_exceeds_discovered_task_count"
    elif reuse_canonical and planned_cell_id in reuse_ids:
        reason = "reused_canonical_seed0_budget7"

    if reason:
        skipped.append(
            {
                "planned_cell_id": planned_cell_id,
                "task_id": str(row["task_id"]),
                "backend_engine": str(row["backend_engine"]),
                "regime": str(row["regime"]),
                "controller_id": str(row["controller_id"]),
                "seed": str(seed),
                "budget": str(budget),
                "skip_reason": reason,
            }
        )
        continue
    jobs.setdefault((str(row["backend_engine"]), seed, budget), set()).add(str(row["task_id"]))

job_rows: list[dict[str, str]] = []
for index, ((backend, seed, budget), task_ids) in enumerate(sorted(jobs.items()), start=1):
    task_file = closeout_root / "manifests" / f"task_ids_runtime_{backend}_seed{seed}_budget{budget}.txt"
    selected = sorted(task_ids, key=lambda value: (int(value), value) if value.isdigit() else (10**9, value))
    task_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
    base_output_dir = closeout_root / ("preflight_only" if is_smoke else "runs")
    output_root = base_output_dir / f"backend_{backend}" / f"seed_{seed}" / f"budget_{budget}"
    log_prefix = "smoke" if is_smoke else "run"
    log_path = closeout_root / "logs" / f"{log_prefix}_backend_{backend}_seed{seed}_budget{budget}.log"
    job_rows.append(
        {
            "job_index": str(index),
            "backend_engine": backend,
            "seed": str(seed),
            "budget": str(budget),
            "task_count": str(len(selected)),
            "task_file": str(task_file),
            "output_root": str(output_root),
            "log_path": str(log_path),
            "command": (
                f'NIPS_WEBARENA_CONTROLLER_FIXED_BUDGET={budget} '
                f'bash scripts/run_webarena_real_controller_pilot.sh --tasks "{task_file}" '
                f'--limit "{len(selected)}" --output-root "{output_root}" '
                f'--controllers "hook_a_only,hook_b_only" --regimes "clean,medium" '
                f'--backends "{backend}" --seeds "{seed}"'
            ),
        }
    )

jobs_path.parent.mkdir(parents=True, exist_ok=True)
with jobs_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = ["job_index", "backend_engine", "seed", "budget", "task_count", "task_file", "output_root", "log_path", "command"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in job_rows:
        writer.writerow(row)
with skipped_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = ["planned_cell_id", "task_id", "backend_engine", "regime", "controller_id", "seed", "budget", "skip_reason"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in skipped:
        writer.writerow(row)
print(f"jobs={len(job_rows)} skipped={len(skipped)}")
PY

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "dry_run=true"
  echo "jobs_csv=${JOBS_CSV}"
  echo "skipped_csv=${SKIPPED_CSV}"
  exit 0
fi

missing_env=()
for key in WA_SHOPPING WA_REDDIT WA_GITLAB WA_HOMEPAGE REDACTED_REQUEST_METADATA_ENV NIPS_BENCH_WEBARENA_STORAGE_STATE NIPS_WEBARENA_VERIFIED_AVAILABLE_TASKS NIPS_WEBARENA_VERIFIED_DATASET NIPS_WEBARENA_VERIFIED_ROOT; do
  if [[ -z "${!key:-}" ]]; then
    missing_env+=("${key}")
  fi
done
if [[ ${#missing_env[@]} -gt 0 ]]; then
  {
    echo "# Smoke/Run Failure Report"
    echo
    echo "failure_type: missing_dependency"
    echo "missing_env_vars: ${missing_env[*]}"
    echo "No WebArena traffic was run."
  } > "${CLOSEOUT_ROOT}/reports/smoke_failure_report.md"
  echo "missing required env vars: ${missing_env[*]}" >&2
  exit 3
fi

tail -n +2 "${JOBS_CSV}" | while IFS=, read -r job_index backend seed budget task_count task_file output_root log_path command; do
  mkdir -p "${output_root}" "$(dirname "${log_path}")"
  echo "running job ${job_index}: backend=${backend} seed=${seed} budget=${budget} tasks=${task_count}" | tee "${log_path}"
  NIPS_WEBARENA_CONTROLLER_FIXED_BUDGET="${budget}" \
    bash scripts/run_webarena_real_controller_pilot.sh \
      --tasks "${task_file}" \
      --limit "${task_count}" \
      --output-root "${output_root}" \
      --controllers "hook_a_only,hook_b_only" \
      --regimes "clean,medium" \
      --backends "${backend}" \
      --seeds "${seed}" >> "${log_path}" 2>&1
done
