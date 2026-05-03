"""Validation helpers for real benchmark backends."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import importlib
import json
import os
import sys

from runtime.external_paths import DEFAULT_LOCAL_PATHS_MANIFEST, load_local_paths
from runtime.repo_probe import RepoProbe, probe_repo


def _credential_status() -> str:
    REDACTED_BROWSER_STATE_LABEL = os.environ.get("NIPS_BENCH_WEBARENA_STORAGE_STATE")
    base_url = os.environ.get("NIPS_BENCH_WEBARENA_BASE_URL")
    if REDACTED_BROWSER_STATE_LABEL and base_url and Path(REDACTED_BROWSER_STATE_LABEL).exists():
        return "configured"
    return "not_configured"


def _import_status(import_name: str, repo_root: Path | None, *, check_imports: bool) -> tuple[str, str | None]:
    if not check_imports:
        return "not_checked", None
    if repo_root is None or not repo_root.exists():
        return "skipped", "repo_missing"

    sys.path.insert(0, str(repo_root))
    try:
        importlib.import_module(import_name)
    except Exception as exc:  # pragma: no cover - exercised in live validation
        return "error", str(exc)
    finally:
        try:
            sys.path.remove(str(repo_root))
        except ValueError:
            pass
    return "ok", None


def _probe_payload(
    probe: RepoProbe,
    *,
    import_name: str,
    check_imports: bool,
    credential_status: str | None = None,
) -> dict[str, object]:
    import_status, import_error = _import_status(import_name, probe.root, check_imports=check_imports)
    payload: dict[str, object] = {
        "root": None if probe.root is None else str(probe.root),
        "repo_present": probe.present,
        "git_head": probe.git_head,
        "dirty": probe.dirty,
        "head_ref": probe.head_ref,
        "markers": list(probe.markers),
        "error": probe.error,
        "import_status": import_status,
        "import_error": import_error,
    }
    if credential_status is not None:
        payload["credential_status"] = credential_status
    return payload


def validate_real_backends(
    *,
    browsergym_repo: Path | None,
    swebench_repo: Path | None,
    require_credentials: bool,
    check_imports: bool = False,
) -> dict[str, object]:
    browsergym_probe = probe_repo("browsergym", browsergym_repo)
    swebench_probe = probe_repo("swebench", swebench_repo)
    credential_status = _credential_status()
    required_repos_present = browsergym_probe.present and swebench_probe.present
    browsergym_payload = _probe_payload(
        browsergym_probe,
        import_name="browsergym",
        check_imports=check_imports,
        credential_status=credential_status,
    )
    swebench_payload = _probe_payload(
        swebench_probe,
        import_name="swebench",
        check_imports=check_imports,
    )
    failure_reasons: list[str] = []
    if not browsergym_probe.present:
        failure_reasons.append("browsergym_probe_failed")
    if not swebench_probe.present:
        failure_reasons.append("swebench_probe_failed")
    if browsergym_payload["import_status"] == "error":
        failure_reasons.append("browsergym_import_error")
    if swebench_payload["import_status"] == "error":
        failure_reasons.append("swebench_import_error")
    validation_pass = not failure_reasons

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_required_repos_present": required_repos_present,
        "credentials_required": require_credentials,
        "repo_bootstrap_ready": required_repos_present,
        "live_webarena_ready": required_repos_present and credential_status == "configured",
        "validation_pass": validation_pass,
        "failure_reasons": failure_reasons,
        "browsergym": browsergym_payload,
        "swebench": swebench_payload,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browsergym-repo", type=Path)
    parser.add_argument("--swebench-repo", type=Path)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_LOCAL_PATHS_MANIFEST)
    parser.add_argument("--require-credentials", action="store_true")
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    local_paths = load_local_paths(args.manifest_path)
    summary = validate_real_backends(
        browsergym_repo=args.browsergym_repo or local_paths.browsergym_repo,
        swebench_repo=args.swebench_repo or local_paths.swebench_repo,
        require_credentials=args.require_credentials,
        check_imports=args.check_imports,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.strict:
        if not summary["validation_pass"]:
            raise SystemExit(1)
        if args.require_credentials and not summary["live_webarena_ready"]:
            raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
