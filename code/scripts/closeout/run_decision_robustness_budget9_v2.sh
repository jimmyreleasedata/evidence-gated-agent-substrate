#!/usr/bin/env bash
set -euo pipefail

ROOT=""
DRY_RUN="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${ROOT}" ]]; then
  echo "missing --root" >&2
  exit 2
fi

CLOSEOUT_ROOT="${ROOT}/decision_robustness_closeout_v2_budget9"
JOBS="${CLOSEOUT_ROOT}/manifests/execution_jobs.csv"
if [[ ! -f "${JOBS}" ]]; then
  echo "missing execution jobs: ${JOBS}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/var/tmp/bounded_stronger_traffic_source_sanity_playwright_venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

export NIPS_CANONICAL_FINAL_ROOT="${NIPS_CANONICAL_FINAL_ROOT:-${ROOT}}"
if [[ -d "/var/tmp/bounded_stronger_traffic_source_sanity_playwright_browsers" && -z "${PLAYWRIGHT_BROWSERS_PATH:-}" ]]; then
  export PLAYWRIGHT_BROWSERS_PATH="/var/tmp/bounded_stronger_traffic_source_sanity_playwright_browsers"
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "dry_run=true"
  echo "jobs=${JOBS}"
  tail -n +2 "${JOBS}"
  exit 0
fi

missing_env=()
for key in WA_SHOPPING WA_REDDIT WA_GITLAB WA_HOMEPAGE REDACTED_REQUEST_METADATA_ENV NIPS_BENCH_WEBARENA_STORAGE_STATE NIPS_WEBARENA_VERIFIED_AVAILABLE_TASKS NIPS_WEBARENA_VERIFIED_DATASET NIPS_WEBARENA_VERIFIED_ROOT; do
  if [[ -z "${!key:-}" ]]; then
    missing_env+=("${key}")
  fi
done
if [[ ${#missing_env[@]} -gt 0 ]]; then
  echo "missing required env vars: ${missing_env[*]}" >&2
  exit 3
fi

tail -n +2 "${JOBS}" | while IFS=, read -r job_index backend seed budget task_count task_file output_root log_path; do
  mkdir -p "${output_root}" "$(dirname "${log_path}")"
  echo "running job ${job_index}: backend=${backend} seed=${seed} budget=${budget} tasks=${task_count}" | tee "${log_path}"
  NIPS_WEBARENA_CONTROLLER_FIXED_BUDGET="${budget}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/run_webarena_real_controller_pilot.sh \
      --tasks "${task_file}" \
      --limit "${task_count}" \
      --output-root "${output_root}" \
      --controllers "hook_a_only,hook_b_only" \
      --regimes "clean,medium" \
      --backends "${backend}" \
      --seeds "${seed}" >> "${log_path}" 2>&1
done
