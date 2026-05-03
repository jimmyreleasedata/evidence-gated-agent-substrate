"""Build stable run indices for Evaluation Environment collection roots."""

from __future__ import annotations

from pathlib import Path
import argparse
import json


DEFAULT_MODES = {
    "webarena_verified": "replay_small",
    "swe_gym": "slice_small",
    "queue_coupling": "microbench",
    "verifier_tail": "microbench",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--queue", type=str, default="unknown")
    parser.add_argument("--hostname", type=str, default="unknown")
    parser.add_argument("--family-root", action="append", default=[], help="family=/abs/path/to/artifact_root")
    parser.add_argument("--family-mode", action="append", default=[], help="family=mode override for explicit family roots")
    parser.add_argument("--job-id", action="append", default=[], help="family=pbs_job_id override")
    return parser


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_mapping(items: list[str], label: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise SystemExit(f"invalid {label}: {item!r}; expected family=value")
        mapping[key] = value
    return mapping


def _locate_file(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def _locate_csv_path(family: str, artifact_root: Path, summary: dict) -> str:
    summary_csv = str(summary.get("csv_path", "N/A"))
    if summary_csv and summary_csv != "N/A":
        return summary_csv

    preferred = {
        "queue_coupling": "queue_coupling.csv",
        "verifier_tail": "verifier_tail.csv",
    }.get(family)
    if preferred:
        path = artifact_root / preferred
        if path.exists():
            return str(path)

    aggregate_matches = sorted(artifact_root.glob("aggregate*.csv"))
    if aggregate_matches:
        return str(aggregate_matches[-1])

    csv_matches = sorted(artifact_root.rglob("*.csv"))
    return str(csv_matches[0]) if csv_matches else "N/A"


def _model_backend(summary: dict, manifest: dict) -> str:
    return str(
        summary.get("backend")
        or summary.get("runtime_backend")
        or manifest.get("backend")
        or manifest.get("runtime_backend")
        or "N/A"
    )


def _collect_family_entry(family: str, mode: str, artifact_root: Path, job_id_override: str | None = None) -> dict[str, str]:
    artifact_root = artifact_root.resolve()
    summary_path = _locate_file(artifact_root, "summary.json")
    if summary_path is None:
        raise FileNotFoundError(f"missing summary.json under {artifact_root}")

    manifest_path = _locate_file(artifact_root, "run_manifest.json")
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path) if manifest_path is not None else {}

    return {
        "family": family,
        "mode": mode,
        "job_id": job_id_override or artifact_root.name,
        "artifact_root": str(artifact_root),
        "summary_path": str(summary_path),
        "run_manifest_path": str(manifest_path) if manifest_path is not None else "N/A",
        "csv_path": _locate_csv_path(family, artifact_root, summary),
        "trace_schema_version": str(summary.get("trace_schema_version", manifest.get("trace_schema_version", "N/A"))),
        "replay_class": str(summary.get("replay_class", manifest.get("replay_class", "N/A"))),
        "system_version": str(summary.get("system_version", manifest.get("system_version", "N/A"))),
        "verifier_version": str(
            summary.get("verifier_version")
            or summary.get("evaluator_version")
            or manifest.get("verifier_version")
            or "N/A"
        ),
        "policy_version": str(summary.get("policy_version", manifest.get("policy_version", "N/A"))),
        "model_backend": _model_backend(summary, manifest),
        "model_id": str(summary.get("model_id", manifest.get("model_id", "N/A"))),
    }


def _discover_family_entries(
    run_root: Path,
    explicit_roots: dict[str, str],
    explicit_modes: dict[str, str],
    explicit_job_ids: dict[str, str],
) -> dict[str, dict[str, str]]:
    if explicit_roots:
        entries = {}
        for family, root_str in explicit_roots.items():
            mode = explicit_modes.get(family, DEFAULT_MODES.get(family, "unknown"))
            entries[family] = _collect_family_entry(family, mode, Path(root_str).resolve(), explicit_job_ids.get(family))
        return entries

    entries = {}
    for family, default_mode in DEFAULT_MODES.items():
        family_root = run_root / family
        if not family_root.exists():
            continue
        mode_dirs = sorted(path for path in family_root.iterdir() if path.is_dir())
        if not mode_dirs:
            continue
        chosen_mode_dir = None
        for path in mode_dirs:
            if path.name == default_mode:
                chosen_mode_dir = path
                break
        if chosen_mode_dir is None:
            chosen_mode_dir = mode_dirs[-1]

        job_dirs = sorted(path for path in chosen_mode_dir.iterdir() if path.is_dir())
        if not job_dirs:
            continue
        entries[family] = _collect_family_entry(family, chosen_mode_dir.name, job_dirs[-1], explicit_job_ids.get(family))
    return entries


def _merge_labels(existing: dict, queue: str, hostname: str) -> tuple[str, list[str], str, list[str]]:
    queues = set(existing.get("queues", []))
    if existing.get("queue") and existing.get("queue") != "mixed":
        queues.add(existing["queue"])
    if queue:
        queues.add(queue)

    hosts = set(existing.get("hosts", []))
    if existing.get("hostname") and existing.get("hostname") != "mixed":
        hosts.add(existing["hostname"])
    if hostname:
        hosts.add(hostname)

    queue_value = sorted(queues)[0] if len(queues) == 1 else "mixed"
    host_value = sorted(hosts)[0] if len(hosts) == 1 else "mixed"
    return queue_value, sorted(queues), host_value, sorted(hosts)


def build_run_index(
    run_root: Path,
    run_tag: str,
    queue: str,
    hostname: str,
    explicit_roots: dict[str, str] | None = None,
    explicit_modes: dict[str, str] | None = None,
    explicit_job_ids: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    explicit_roots = explicit_roots or {}
    explicit_modes = explicit_modes or {}
    explicit_job_ids = explicit_job_ids or {}
    run_root.mkdir(parents=True, exist_ok=True)

    entries = _discover_family_entries(run_root, explicit_roots, explicit_modes, explicit_job_ids)
    if not entries:
        raise FileNotFoundError(f"no Evaluation Environment family artifacts discovered under {run_root}")

    existing_index_path = run_root / "run_index.json"
    existing = _read_json(existing_index_path) if existing_index_path.exists() else {}
    queue_value, queues, host_value, hosts = _merge_labels(existing, queue, hostname)

    pbs_job_ids = sorted({entry["job_id"] for entry in entries.values()})
    run_index = {
        "run_tag": run_tag,
        "run_root": str(run_root),
        "queue": queue_value,
        "queues": queues,
        "hostname": host_value,
        "hosts": hosts,
        "pbs_job_ids": pbs_job_ids,
        "families": entries,
    }
    phase_manifest = {
        "run_tag": run_tag,
        "run_root": str(run_root),
        "queue": queue_value,
        "hostname": host_value,
        "pbs_job_ids": pbs_job_ids,
        "family_count": len(entries),
        "family_paths": {family: entry["artifact_root"] for family, entry in entries.items()},
        "run_index_path": str(run_root / "run_index.json"),
        "phase_manifest_path": str(run_root / "phase_manifest.json"),
    }
    return run_index, phase_manifest


def main() -> None:
    args = build_parser().parse_args()
    run_root = args.run_root.resolve()
    run_tag = args.run_tag or run_root.name
    explicit_roots = _parse_mapping(args.family_root, "--family-root")
    explicit_modes = _parse_mapping(args.family_mode, "--family-mode")
    explicit_job_ids = _parse_mapping(args.job_id, "--job-id")
    run_index, phase_manifest = build_run_index(
        run_root=run_root,
        run_tag=run_tag,
        queue=args.queue,
        hostname=args.hostname,
        explicit_roots=explicit_roots,
        explicit_modes=explicit_modes,
        explicit_job_ids=explicit_job_ids,
    )
    _write_json(run_root / "run_index.json", run_index)
    _write_json(run_root / "phase_manifest.json", phase_manifest)
    print(run_root / "run_index.json")


if __name__ == "__main__":
    main()
