#!/usr/bin/env python3
"""Collect the full NeurIPS evidence v2 package from historical and current roots."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.build_expected_roots_manifest import build_expected_roots_manifest
from scripts.final.build_full_claim_support_matrix import build_full_claim_support_matrix
from scripts.final.build_full_figure_input_inventory import build_full_figure_input_inventory
from scripts.final.build_full_paper_facing_surface import build_full_paper_facing_surface
from scripts.final.deduplicate_and_reconcile_evidence import deduplicate_and_reconcile_evidence
from scripts.final.discover_all_experiment_roots import discover_experiment_roots
from scripts.final.audit_reproducibility_canonical_paths import audit_reproducibility_canonical_paths
from scripts.final.full_evidence_common import (
    CATEGORY_DIRS,
    MAX_COPY_BYTES,
    MAX_HASH_BYTES,
    PACKAGE_NAME,
    copy_or_redact_artifact,
    ensure_output_dirs,
    expected_roots,
    git_commit,
    infer_category,
    is_experiment_filename,
    is_sensitive_artifact,
    iter_files_limited,
    read_csv,
    repo_path,
    resolve_final_root,
    safe_relative,
    sha256_file,
    write_csv,
    write_json,
    write_parquet_if_possible,
    write_yaml,
)
from scripts.final.normalize_all_evidence_rows import normalize_all_evidence_rows
from scripts.final.run_full_evidence_gate import run_full_evidence_gate


KNOWN_FINAL_PATCH = {
    "admitted_declared_traffic_rows": 1190,
    "rejected_rows": 0,
    "smoke_only_rows": 33,
    "MiniWoB++": 1008,
    "SWE": 105,
    "WebArena Verified": 77,
}


ARTIFACT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".yaml", ".yml", ".txt", ".log", ".sha256"}
BROAD_CONTAINER_ROOTS = {
    "artifacts/evaluation_environment_runs",
    "artifacts/reports/model_backend_matrix",
    "artifacts/reports/readiness",
    "artifacts/reports/phase5_webarena",
    "artifacts/reports/phase5_swe",
    "artifacts/real_cutover",
    "nips_upstream_assets/webarena_verified",
    "nips_upstream_assets/swebench",
    "nips_upstream_assets/miniwob",
}


def _is_evidence_artifact_file(path: Path) -> bool:
    name = path.name
    if is_sensitive_artifact(path):
        return True
    if is_experiment_filename(name):
        return True
    if name in {
        "generated_patch.diff",
        "raw_model_response.txt",
        "artifact_hashes.sha256",
        "release_manifest.yaml",
        "croissant.json",
        "README.md",
    }:
        return True
    lowered = name.lower()
    if any(token in lowered for token in ("manifest", "summary", "inventory", "report", "matrix", "gate", "claim", "figure", "hash")):
        return path.suffix.lower() in ARTIFACT_SUFFIXES
    return False


def _is_broad_container(path: Path, repo_root: Path) -> bool:
    try:
        rel = str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return False
    return rel in BROAD_CONTAINER_ROOTS


def _manifest_roots(final_root: Path) -> list[dict[str, object]]:
    path = final_root / "inventories" / "expected_roots_manifest.yaml"
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(payload.get("roots", []))


def _collect_artifacts(
    *,
    repo_root: Path,
    final_root: Path,
    max_copy_bytes: int,
    max_hash_bytes: int,
    max_files_per_root: int | None,
) -> dict[str, object]:
    expected = _manifest_roots(final_root)
    discovered = read_csv(final_root / "inventories" / "discovered_roots.csv")
    root_specs: list[tuple[Path, str, str]] = []
    seen_roots: set[Path] = set()

    for entry in expected:
        path = repo_path(repo_root, str(entry["expected_path"]))
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved == final_root.resolve():
            continue
        if path.is_dir() and _is_broad_container(path, repo_root):
            # These roots are represented by auto-discovered child run roots.
            # Recursing the parent duplicates thousands of files and can turn
            # the evidence package into a runtime workspace mirror.
            continue
        if resolved not in seen_roots:
            seen_roots.add(resolved)
            root_specs.append((path, str(entry["category"]), str(entry["root_id"])))
    for row in discovered:
        path = repo_path(repo_root, row["path"])
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved == final_root.resolve():
            continue
        if resolved not in seen_roots:
            seen_roots.add(resolved)
            root_specs.append((path, row.get("inferred_category") or infer_category(path), row.get("discovered_root_id") or path.name))

    artifact_rows: list[dict[str, object]] = []
    redactions: list[dict[str, object]] = []
    hash_lines: list[str] = []
    run_root_rows: list[dict[str, object]] = []
    seen_files: set[Path] = set()
    for root, category, root_id in root_specs:
        file_count = 0
        files = [root] if root.is_file() else iter_files_limited(root, skip_root=final_root, max_files=max_files_per_root)
        for file_path in files:
            if not _is_evidence_artifact_file(file_path):
                continue
            resolved_file = file_path.resolve()
            if resolved_file in seen_files:
                continue
            seen_files.add(resolved_file)
            file_count += 1
            digest, hash_status = sha256_file(file_path, max_bytes=max_hash_bytes)
            copy_status, redaction = copy_or_redact_artifact(
                file_path=file_path,
                repo_root=repo_root,
                final_root=final_root,
                category=category,
                digest=digest,
                max_copy_bytes=max_copy_bytes,
            )
            if redaction:
                redactions.append(redaction)
            rel = safe_relative(file_path, repo_root)
            artifact_rows.append(
                {
                    "root_id": root_id,
                    "source_path": str(file_path),
                    "relative_path": rel,
                    "category": category,
                    "sha256": digest,
                    "hash_status": hash_status,
                    "size_bytes": file_path.stat().st_size,
                    "copy_status": copy_status,
                    "redaction_status": redaction["redaction_status"] if redaction else "",
                }
            )
            if hash_status == "hashed":
                hash_lines.append(f"{digest}  {rel}")
        run_root_rows.append(
            {
                "root_id": root_id,
                "run_root": safe_relative(root, repo_root),
                "category": category,
                "file_count_indexed": file_count,
                "index_truncated": str(max_files_per_root is not None and file_count >= max_files_per_root).lower(),
            }
        )

    write_csv(final_root / "inventories" / "all_artifacts_inventory.csv", artifact_rows)
    write_parquet_if_possible(final_root / "inventories" / "all_artifacts_inventory.csv", artifact_rows)
    write_csv(final_root / "inventories" / "all_run_roots_inventory.csv", run_root_rows)
    write_parquet_if_possible(final_root / "inventories" / "all_run_roots_inventory.csv", run_root_rows)
    write_csv(final_root / "sensitive_artifact_redaction_manifest.csv", redactions)
    (final_root / "artifact_hashes.sha256").write_text("\n".join(sorted(hash_lines)) + ("\n" if hash_lines else ""), encoding="utf-8")
    return {"artifact_count": len(artifact_rows), "redaction_count": len(redactions), "run_root_count": len(run_root_rows)}


def _write_final_numbers_comparison(final_root: Path, paper_rows: list[dict[str, str]], gate: dict[str, object]) -> None:
    family_counts = Counter(row.get("family", "") for row in paper_rows)
    reconstructed = {
        "admitted_declared_traffic_rows": len(paper_rows),
        "rejected_rows": int(gate.get("rejected_rows", 0)),
        "smoke_only_rows": len(read_csv(final_root / "inventories" / "excluded_rows.csv")),
        "MiniWoB++": family_counts.get("miniwob", 0),
        "SWE": family_counts.get("swe_gym", 0),
        "WebArena Verified": family_counts.get("webarena_verified", 0),
    }
    lines = [
        "# Final Numbers Comparison",
        "",
        "Known latest patch values are sanity checks, not hardcoded truth. Mismatches mean the collector should identify missing roots or explain why the reconstructed ledger differs.",
        "",
        "| Metric | Known patch | Reconstructed | Status |",
        "|---|---:|---:|---|",
    ]
    mismatches = []
    for key, expected in KNOWN_FINAL_PATCH.items():
        actual = reconstructed.get(key, 0)
        status = "match" if actual == expected else "mismatch"
        if status == "mismatch":
            mismatches.append(key)
        lines.append(f"| {key} | {expected} | {actual} | {status} |")
    lines.extend(
        [
            "",
            "## Required Handling",
            "",
            "- If mismatches remain, do not force the known values.",
            "- Use `inventories/missing_expected_roots.csv` and `inventories/discovered_roots.csv` to locate missing roots.",
            "- Use `reports/missing_data_report.md` for the explicit missing-root explanation.",
            "",
            f"Mismatched metrics: {', '.join(mismatches) if mismatches else 'none'}",
        ]
    )
    (final_root / "reports" / "final_numbers_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reports(final_root: Path, paper_rows: list[dict[str, str]], gate: dict[str, object]) -> None:
    missing = read_csv(final_root / "inventories" / "missing_expected_roots.csv")
    discovered = read_csv(final_root / "inventories" / "discovered_roots.csv")
    excluded = read_csv(final_root / "inventories" / "excluded_rows.csv")
    artifacts = read_csv(final_root / "inventories" / "all_artifacts_inventory.csv")
    family_counts = Counter(row.get("family", "") for row in paper_rows)
    category_counts = Counter(row.get("evidence_class", "") for row in paper_rows)
    driver_counts = Counter(row.get("driver_type", "") for row in paper_rows)
    model_counts = Counter(row.get("model_family", "") for row in paper_rows)
    backend_counts = Counter(row.get("backend_engine", "") for row in paper_rows)

    def counter_md(counter: Counter[str]) -> str:
        return "\n".join(f"- `{key or 'unknown'}`: {value}" for key, value in sorted(counter.items())) or "- none"

    full_summary = [
        "# Full Evidence Summary",
        "",
        f"- Package: `{PACKAGE_NAME}`",
        f"- Artifacts indexed: {len(artifacts)}",
        f"- Discovered roots: {len(discovered)}",
        f"- Missing expected roots: {len(missing)}",
        f"- Paper-facing rows: {len(paper_rows)}",
        f"- Excluded rows: {len(excluded)}",
        f"- Rejected rows: {gate.get('rejected_rows', 0)}",
        "",
        "## Rows By Evidence Category",
        counter_md(category_counts),
        "",
        "## Rows By Family",
        counter_md(family_counts),
        "",
        "## Rows By Driver Type",
        counter_md(driver_counts),
        "",
        "## Rows By Model Family",
        counter_md(model_counts),
        "",
        "## Rows By Backend Engine",
        counter_md(backend_counts),
    ]
    (final_root / "reports" / "full_evidence_summary.md").write_text("\n".join(full_summary) + "\n", encoding="utf-8")

    coverage = [
        "# Collection Coverage Report",
        "",
        f"- Expected roots missing: {len(missing)}",
        f"- Auto-discovered roots: {len(discovered)}",
        "",
        "## Missing Expected Roots",
        "",
        *[f"- `{row['expected_path']}` ({row['category']}): {row['missing_reason']}" for row in missing[:300]],
        "",
        "## Auto-Discovered Roots Not In Expected Manifest",
        "",
        "See `inventories/discovered_roots.csv`; this report intentionally does not collapse dynamic roots into the expected manifest.",
    ]
    (final_root / "reports" / "collection_coverage_report.md").write_text("\n".join(coverage) + "\n", encoding="utf-8")
    (final_root / "reports" / "missing_data_report.md").write_text(
        "\n".join(
            [
                "# Missing Data Report",
                "",
                "Missing expected roots are recorded rather than ignored. Optional roots may be absent without failing collection, but required real-cutover/model-backend roots should be investigated if missing.",
                "",
                *[f"- `{row['expected_path']}`: {row['missing_reason']}" for row in missing],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (final_root / "reports" / "conflict_resolution_report.md").write_text(
        "\n".join(
            [
                "# Conflict Resolution Report",
                "",
                "Rows are not dropped blindly. Exact duplicate hashes and conflicting logical cells are recorded in inventories.",
                "",
                "- Duplicate report: `inventories/duplicate_report.csv`",
                "- Conflict report: `inventories/conflict_report.csv`",
                "- Rows demoted from paper-facing: `inventories/excluded_rows.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (final_root / "reports" / "final_limitations_and_blockers.md").write_text(
        "\n".join(
            [
                "# Final Limitations And Blockers",
                "",
                "- External model access and memory blockers remain explicit in readiness/model-backend inventories where present.",
                "- MiniMax is optional and not part of the current main external-model surface.",
                "- WebArena Qwen-30B rows must remain smoke-only unless admitted by the full gate.",
                "- Do not claim WebArena p99/component shifts when decision labels or component fields are absent.",
                "- Do not claim SWE ecological failure diversity when one category has share 1.0.",
                "- Negative blocker evidence documents attempted routes but is not quantitative paper-facing evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def collect_full_evidence(
    *,
    repo_root: Path,
    final_root: Path | None = None,
    max_copy_bytes: int = MAX_COPY_BYTES,
    max_hash_bytes: int = MAX_HASH_BYTES,
    max_files_per_root: int | None = None,
    discovery_max_dirs: int | None = None,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    final_root = resolve_final_root(repo_root, final_root)
    ensure_output_dirs(final_root)

    expected_result = build_expected_roots_manifest(repo_root=repo_root, final_root=final_root)
    discovered = discover_experiment_roots(repo_root=repo_root, final_root=final_root, max_dirs=discovery_max_dirs)
    canonical_audit = audit_reproducibility_canonical_paths(repo_root=repo_root, final_root=final_root)
    evidence_rows = normalize_all_evidence_rows(repo_root=repo_root, final_root=final_root)
    reconciliation = deduplicate_and_reconcile_evidence(final_root=final_root)
    gate = run_full_evidence_gate(final_root=final_root)
    paper_rows = build_full_paper_facing_surface(final_root=final_root)
    figure_rows = build_full_figure_input_inventory(final_root=final_root)
    claim_rows = build_full_claim_support_matrix(final_root=final_root)
    artifact_result = _collect_artifacts(
        repo_root=repo_root,
        final_root=final_root,
        max_copy_bytes=max_copy_bytes,
        max_hash_bytes=max_hash_bytes,
        max_files_per_root=max_files_per_root,
    )

    _write_final_numbers_comparison(final_root, paper_rows, gate)
    _write_reports(final_root, paper_rows, gate)

    manifest = {
        "package_name": PACKAGE_NAME,
        "repo_root": str(repo_root),
        "final_root": str(final_root),
        "repo_commit": git_commit(repo_root),
        "expected_root_count": expected_result["expected_root_count"],
        "missing_root_count": expected_result["missing_root_count"],
        "discovered_root_count": len(discovered),
        "evidence_row_count": len(evidence_rows),
        "paper_facing_row_count": len(paper_rows),
        "excluded_row_count": int(gate.get("rejected_rows", 0)),
        "artifact_count": artifact_result["artifact_count"],
        "sensitive_redaction_count": artifact_result["redaction_count"],
        "claim_count": len(claim_rows),
        "figure_input_count": len(figure_rows),
        "duplicate_count": reconciliation["duplicate_count"],
        "conflict_count": reconciliation["conflict_count"],
        "canonical_found_exact_count": canonical_audit["found_exact_count"],
        "canonical_alias_resolved_count": canonical_audit["alias_resolved_count"],
        "canonical_missing_optional_count": canonical_audit["missing_optional_count"],
        "canonical_missing_required_count": canonical_audit["missing_required_count"],
    }
    write_json(final_root / "collection_manifest.json", manifest)
    write_yaml(final_root / "collection_manifest.yaml", manifest)
    (final_root / "README.md").write_text(
        "\n".join(
            [
                f"# {PACKAGE_NAME}",
                "",
                "Full NeurIPS 2026 E&D evidence re-index and collection package.",
                "",
                "This package includes historical roots, real cutover roots, model/backend expansion, closeout data, framework adapter demos, final patch closeout inputs, release metadata, and negative blocker evidence where present.",
                "",
                "Sensitive runtime artifacts are hash-only and listed in `sensitive_artifact_redaction_manifest.csv`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--final-root", type=Path)
    parser.add_argument("--max-copy-bytes", type=int, default=MAX_COPY_BYTES)
    parser.add_argument("--max-hash-bytes", type=int, default=MAX_HASH_BYTES)
    parser.add_argument("--max-files-per-root", type=int)
    parser.add_argument("--discovery-max-dirs", type=int)
    args = parser.parse_args()
    result = collect_full_evidence(
        repo_root=args.repo_root,
        final_root=args.final_root,
        max_copy_bytes=args.max_copy_bytes,
        max_hash_bytes=args.max_hash_bytes,
        max_files_per_root=args.max_files_per_root,
        discovery_max_dirs=args.discovery_max_dirs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
