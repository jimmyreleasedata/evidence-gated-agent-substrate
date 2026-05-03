#!/usr/bin/env bash
set -euo pipefail

ROOT="${NIPS_CANONICAL_FINAL_ROOT:-artifact_release_root}"
PYTHON_BIN="${PYTHON_BIN:-ANON_HOME/.conda/envs/llm-env/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec "$PYTHON_BIN" "$REPO_ROOT/scripts/final/audit_decision_sensitive_slice_v1.py" \
  --canonical-root "$ROOT" \
  --out-dir "$ROOT/decision_sensitive_admission" \
  --discovery-mode "${DECISION_SLICE_DISCOVERY_MODE:-known_only}"
