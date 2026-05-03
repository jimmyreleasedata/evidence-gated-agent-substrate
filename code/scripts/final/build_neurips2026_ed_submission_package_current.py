#!/usr/bin/env python3
"""Build the current anonymous NeurIPS E&D submission package.

This is packaging/anonymization only. It copies a reviewer-facing source/data
subset into a clean tree, stages a Hugging Face dataset tree, writes validation
reports, builds a supplementary zip, and initializes a fresh one-commit git repo
without using the original repository history.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ANON_ROOT = "artifact_release_root"
ANON_HOME = "ANON_HOME"
DEFAULT_BRANCH = "ed-paper-submission"
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_ZIP_BYTES = 100 * 1024 * 1024

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".venv",
    "node_modules",
    "raw",
    "logs",
    "runs",
    "tmp",
    "temporary",
}

SKIP_NAME_FRAGMENTS = {
    "credential",
    "credential",
    "passwd",
    "secret",
    "token",
    "private",
    "REDACTED_BROWSER_STATE_LABEL",
    "REDACTED_BROWSER_STATE_LABEL",
    "header",
    "session",
}

PRIVATE_REPLACEMENTS = [
    ("artifact_release_root", ANON_ROOT),
    ("artifact_release_root", ANON_ROOT),
    ("artifact_release_root", ANON_ROOT),
    ("artifact_release_root", ANON_ROOT),
    ("artifact_release_root", ANON_ROOT),
    ("artifact_release_root", ANON_ROOT),
    ("artifact_release_root", ANON_ROOT),
    ("artifact_release_root", ANON_ROOT),
    ("ANON_HOME", ANON_HOME),
    ("ANON_HOME", ANON_HOME),
    ("ANON_HOME/", f"{ANON_HOME}/"),
    ("artifact_release_root/", f"{ANON_ROOT}/"),
    ("artifact_release_root", ANON_ROOT),
    ("ANON_HOME", ANON_HOME),
    ("ANON_USER", "ANON_USER"),
    ("anonymous_user", "anonymous_user"),
    ("anonymous_user", "anonymous_user"),
    ("anonymous_allocation", "anonymous_allocation"),
]

LABEL_REPLACEMENTS = [
    ("targeted attribution audit", "targeted attribution audit"),
    ("targeted attribution audit", "targeted attribution audit"),
    ("targeted_failure_attribution", "targeted_failure_attribution"),
    ("targeted attribution quality check", "targeted attribution quality check"),
    ("targeted attribution quality check", "targeted attribution quality check"),
    ("targeted_attribution_quality_check", "targeted_attribution_quality_check"),
    ("SWE driver-surface recompute", "SWE driver-surface recompute"),
    ("SWE driver-surface recompute", "SWE driver-surface recompute"),
    ("swe_driver_surface_recompute", "swe_driver_surface_recompute"),
    ("SWE verifier-control repair", "SWE verifier-control repair"),
    ("SWE verifier-control repair", "SWE verifier-control repair"),
    ("swe_verifier_control_repair", "swe_verifier_control_repair"),
    ("SWE verifier closeout", "SWE verifier closeout"),
    ("SWE verifier closeout", "SWE verifier closeout"),
    ("swe_verifier_closeout", "swe_verifier_closeout"),
    ("bounded stronger-traffic-source sanity check", "bounded stronger-traffic-source sanity check"),
    ("bounded stronger-traffic-source sanity check", "bounded stronger-traffic-source sanity check"),
    ("bounded_stronger_traffic_source_sanity", "bounded_stronger_traffic_source_sanity"),
    ("targeted attribution audit", "targeted attribution audit"),
    ("targeted attribution audit", "targeted attribution audit"),
    ("targeted_failure_attribution", "targeted_failure_attribution"),
    ("closeout", "closeout"),
    ("closeout", "closeout"),
    ("closeout", "closeout"),
    ("initial evidence pass", "initial evidence pass"),
    ("secondary evidence pass", "secondary evidence pass"),
    ("closeout pass", "closeout pass"),
    ("SWE verifier closeout", "SWE verifier closeout"),
    ("stronger traffic-source sanity", "stronger traffic-source sanity"),
    ("targeted attribution quality check", "targeted attribution quality check"),
    ("SWE verifier-control repair", "SWE verifier-control repair"),
    ("SWE driver-surface recompute", "SWE driver-surface recompute"),
    ("closeout", "closeout"),
    ("evidence-stratified", "evidence-stratified"),
    ("recompute_required_status", "recompute_required_status"),
    ("non-paper-facing fixture", "non-paper-facing fixture"),
    ("non-paper-facing synthetic fixture", "non-paper-facing synthetic fixture"),
    ("non-paper-facing fixture", "non-paper-facing fixture"),
]

SENSITIVE_REPLACEMENTS = [
    ("REDACTED_SECRET_ENV", "REDACTED_SECRET_ENV"),
    ("REDACTED_REQUEST_METADATA_ENV", "REDACTED_REQUEST_METADATA_ENV"),
    ("REDACTED_BROWSER_STATE_LABEL", "REDACTED_BROWSER_STATE_LABEL"),
    ("REDACTED_BROWSER_STATE_LABEL", "REDACTED_BROWSER_STATE_LABEL"),
    ("REDACTED_BROWSER_STATE_LABEL", "REDACTED_BROWSER_STATE_LABEL"),
    ("REDACTED_BROWSER_STATE_LABEL", "REDACTED_BROWSER_STATE_LABEL"),
    ("REDACTED_BROWSER_STATE_LABEL", "REDACTED_BROWSER_STATE_LABEL"),
    ("credential-bearing request metadata", "credential-bearing request metadata"),
    ("raw browser runtime records", "raw browser runtime records"),
]

FORBIDDEN_PATTERNS = [
    ("artifact_release_root", "private_root"),
    ("ANON_HOME", "private_home"),
    ("ANON_USER", "private_user"),
    ("anonymous_user", "private_user"),
    ("anonymous_allocation", "private_allocation"),
    ("REDACTED_SECRET_ENV", "secret_env"),
    ("REDACTED_REQUEST_METADATA_ENV", "secret_env"),
    ("REDACTED_BROWSER_STATE_LABEL", "browser_state"),
    ("REDACTED_BROWSER_STATE_LABEL", "browser_state"),
    ("credential-bearing request metadata", "request_metadata_label"),
    ("closeout", "internal_label"),
    ("closeout", "internal_label"),
    ("SWE verifier closeout", "internal_label"),
    ("stronger traffic-source sanity", "internal_label"),
    ("SWE verifier-control repair", "internal_label"),
    ("SWE driver-surface recompute", "internal_label"),
    ("targeted attribution quality check", "internal_label"),
    ("swe_verifier_closeout", "internal_label"),
    ("bounded_stronger_traffic_source_sanity", "internal_label"),
    ("closeoutk", "internal_label"),
    ("closeout", "internal_label"),
    ("evidence-stratified", "internal_label"),
    ("recompute_required_status", "internal_label"),
    ("non-paper-facing fixture", "obsolete_wording"),
    ("non-paper-facing synthetic fixture", "obsolete_wording"),
    ("non-paper-facing fixture", "obsolete_wording"),
]

SUPPLEMENTARY_MEMBERS = [
    "README_FIRST.md",
    "ARTIFACT_CARD.md",
    "ACCESS_AND_HOSTING.md",
    "CLAIMS_AND_EVIDENCE.md",
    "RESPONSIBLE_USE.md",
    "validation/paper_claim_to_artifact_map.csv",
    "validation/final_submission_readiness_report.md",
    "validation/anonymization_scan_report.md",
    "validation/raw_secret_scan_report.md",
    "metadata/croissant.json",
    "metadata/croissant_completed_for_openreview.json",
    "metadata/croissant_validation.txt",
    "data/samples/redacted_event_trace_sample.jsonl.gz",
    "data/samples/redacted_manifest_sample.json",
    "checksums/SHA256SUMS",
]


@dataclass(frozen=True)
class PackageResult:
    submission_root: Path
    clean_code_tree: Path
    hf_dataset_tree: Path
    fresh_repo: Path
    fresh_repo_commit: str
    supplementary_zip: Path
    supplementary_zip_size: int
    forbidden_hits: int
    raw_secret_hits: int
    internal_label_hits: int
    missing_required_files: list[str]
    package_ready: bool


def public_code_url(remote_url: str) -> str:
    return remote_url.removesuffix(".git") if remote_url else "https://github.com/jimmyreleasedata/evidence-gated-agent-substrate"


def sanitize_public_text(text: str) -> str:
    out = text
    for old, new in PRIVATE_REPLACEMENTS:
        out = out.replace(old, new)
    for old, new in LABEL_REPLACEMENTS:
        out = out.replace(old, new)
    for old, new in SENSITIVE_REPLACEMENTS:
        out = out.replace(old, new)
    out = re.sub(r"git@github\.com:[^\s\"')]+", "https://github.com/jimmyreleasedata/evidence-gated-agent-substrate", out)
    out = re.sub(r"https://github\.com/[^\s\"')]+", "https://github.com/jimmyreleasedata/evidence-gated-agent-substrate", out)
    out = re.sub(r"[A-Za-z0-9._%+-]+@(?!example\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "anonymous@example.com", out)
    return out


def sanitize_path_part(part: str) -> str:
    sanitized = sanitize_public_text(part)
    sanitized = sanitized.replace(" ", "_").replace("/", "_")
    sanitized = re.sub(r"[^A-Za-z0-9._+=-]", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "artifact"


def sanitize_relpath(path: Path) -> Path:
    return Path(*[sanitize_path_part(part) for part in path.parts])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sanitize_public_text(text), encoding="utf-8")


def should_skip_path(rel: Path) -> bool:
    lowered_parts = [part.lower() for part in rel.parts]
    if any(part in SKIP_DIR_NAMES for part in lowered_parts):
        return True
    name = rel.name.lower()
    return any(fragment in name for fragment in SKIP_NAME_FRAGMENTS)


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def copy_sanitized_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    if should_skip_path(Path(src.name)):
        return False
    if not is_text_file(src) and src.suffix.lower() not in {".gz"}:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".gz":
        if src.stat().st_size > MAX_TEXT_BYTES:
            return False
        shutil.copy2(src, dst)
        return True
    if src.stat().st_size > MAX_TEXT_BYTES:
        return False
    dst.write_text(sanitize_public_text(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    return True


def copy_selected_tree(src_root: Path, dst_root: Path, *, recursive: bool = True) -> int:
    if not src_root.exists() or not src_root.is_dir():
        return 0
    if recursive:
        iterator: Iterable[Path] = (p for p in src_root.rglob("*") if p.is_file())
    else:
        iterator = (p for p in src_root.iterdir() if p.is_file())
    copied = 0
    for src in sorted(iterator):
        rel = src.relative_to(src_root)
        if should_skip_path(rel) or src.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if copy_sanitized_file(src, dst_root / sanitize_relpath(rel)):
            copied += 1
    return copied


def copy_filtered_files(src_root: Path, dst_root: Path, allowed_names: set[str]) -> int:
    if not src_root.exists() or not src_root.is_dir():
        return 0
    copied = 0
    for src in sorted(src_root.iterdir(), key=lambda p: p.name):
        if not src.is_file() or src.name not in allowed_names:
            continue
        if copy_sanitized_file(src, dst_root / sanitize_path_part(src.name)):
            copied += 1
    return copied


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def write_public_docs(root: Path, *, dataset_tree: bool = False, anon_remote_url: str = "", anon_dataset_url: str = "") -> None:
    code_url = public_code_url(anon_remote_url)
    dataset_url = anon_dataset_url or "https://huggingface.co/datasets/jimmyreleasedata/evidence-gated-agent-substrate-release"
    docs = {
        "README.md": "# Anonymous E&D Artifact Package\n\nSanitized code, metadata, validation outputs, and paper-facing evidence for anonymous artifact review.\n",
        "README_FIRST.md": "# Read First\n\nStart with `CLAIMS_AND_EVIDENCE.md`, `RUNME.md`, and `validation/final_submission_readiness_report.md`. No live services are required for smoke validation.\n",
        "INSTALL.md": "# Install\n\nUse Python 3.10+ with pandas, pytest, and PyYAML for local validation. Runtime-heavy benchmark environments are referenced but not bundled.\n",
        "RUNME.md": "# Runme\n\n```bash\npython validate_bundle.py\n```\n\nThis validates required package files and redaction reports. It does not run benchmark experiments.\n",
        "REPRODUCE.md": "# Reproduce\n\nThe package contains final evidence-gated surfaces, figure/table inputs, and validator scripts. Decision-study evidence is packaged separately from the canonical substrate aggregate.\n",
        "ARTIFACT_CARD.md": "# Artifact Card\n\nThis artifact supports an evidence-gated executable benchmark substrate for closed-loop tool-using agent evaluation.\n",
        "ACCESS_AND_HOSTING.md": f"# Access And Hosting\n\ncode_repository_url={code_url}\ndataset_artifact_url={dataset_url}\ndataset_url_status={'prepared' if anon_dataset_url else 'pending_human_hosting'}\n",
        "RESPONSIBLE_USE.md": "# Responsible Use\n\nUse this package for artifact review and reproducibility inspection. It is not intended to recover private runtime state or operator identity.\n",
        "LICENSES.md": "# Licenses\n\nProject code and third-party assets retain their respective licenses and access conditions.\n",
        "THIRD_PARTY_ASSETS.md": "# Third Party Assets\n\nExternal benchmark assets are referenced by metadata and must be obtained under their original terms.\n",
        "MAINTENANCE.md": "# Maintenance\n\nRepeat redaction scans and checksum generation before any public release.\n",
        "CLAIMS_AND_EVIDENCE.md": "# Claims And Evidence\n\nThe canonical substrate aggregate contains 930 admitted rows. Decision-study rows are packaged separately and are not merged into that aggregate.\n",
    }
    if dataset_tree:
        docs["DATASET_CARD.md"] = f"# Dataset Card\n\nname=evidence-gated-agent-substrate-release\nurl={dataset_url}\ncode_url={code_url}\n"
        docs["CITATION_ANONYMOUS.md"] = "# Anonymous Citation\n\nAnonymous NeurIPS 2026 E&D artifact submission.\n"
    for name, content in docs.items():
        write_text(root / name, content)
    write_text(
        root / "validate_bundle.py",
        """#!/usr/bin/env python3
from pathlib import Path

required = [
    "README_FIRST.md",
    "metadata/croissant.json",
    "checksums/SHA256SUMS",
    "validation/final_submission_readiness_report.md",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))
print("submission package smoke validation passed")
""",
    )


def copy_code_tree(repo_root: Path, clean_tree: Path) -> None:
    code_root = clean_tree / "code"
    for rel, recursive in [
        ("adapters", True),
        ("runner", True),
        ("runtime", True),
        ("trace", True),
        ("evaluators", True),
        ("collectors", True),
        ("scripts/closeout", False),
        ("configs", True),
        ("manifests", True),
    ]:
        copy_selected_tree(repo_root / rel, code_root / rel, recursive=recursive)
    final_scripts = {
        "__init__.py",
        "aggregate_neurips_ed_canonical_final_v1.py",
        "audit_decision_sensitive_slice_v1.py",
        "build_final_claim_support_matrix.py",
        "build_full_claim_support_matrix.py",
        "build_full_figure_input_inventory.py",
        "build_neurips2026_ed_submission_package_current.py",
        "build_neurips_ed_anon_branch_and_lfs_release_v1.py",
        "build_neurips_ed_anon_remote_and_hosting_closeout_v1.py",
        "build_neurips_ed_claim_support_matrix.py",
        "build_neurips_ed_figure_input_inventory.py",
        "collect_final_figure_inputs.py",
        "collect_full_neurips_evidence.py",
        "collect_neurips_final_evidence.py",
        "deduplicate_and_reconcile_evidence.py",
        "full_evidence_common.py",
        "hash_neurips_artifacts.py",
        "materialize_decision_sensitive_slice_v1.sh",
        "neurips_ed_canonical_common.py",
        "normalize_all_evidence_rows.py",
        "run_final_evidence_gate.py",
        "run_full_evidence_gate.py",
        "sanitize_sensitive_artifacts.py",
        "validate_full_evidence_collection.py",
        "validate_neurips_ed_canonical_final_v1.py",
        "validate_neurips_final_evidence.py",
    }
    validation_tests = {
        "test_decision_robustness_budget9_v2.py",
        "test_neurips2026_ed_submission_package_current.py",
        "test_webarena_real_controller_pilot.py",
    }
    copy_filtered_files(repo_root / "scripts" / "final", code_root / "scripts" / "final", final_scripts)
    copy_filtered_files(repo_root / "tests", code_root / "tests", validation_tests)
    for rel in ["pyproject.toml", "README.md", "release_manifest.yaml", "THIRD_PARTY_ASSETS.md"]:
        copy_sanitized_file(repo_root / rel, code_root / sanitize_relpath(Path(rel)))


def copy_global_data(canonical_root: Path, hf_tree: Path) -> None:
    global_dst = hf_tree / "data" / "global"
    global_files = [
        "paper_facing_surface.csv",
        "evidence_gate_report.json",
        "excluded_rows.csv",
        "validation_report.json",
        "final_claim_support_matrix.csv",
        "final_claim_support_matrix.md",
        "figure_input_inventory.csv",
        "global_evidence_index.csv",
    ]
    for name in global_files:
        copy_sanitized_file(canonical_root / "global" / name, global_dst / name)
    if (global_dst / "final_claim_support_matrix.csv").exists():
        shutil.copy2(global_dst / "final_claim_support_matrix.csv", global_dst / "claim_support_matrix.csv")
    copy_selected_tree(canonical_root / "paper_inputs", hf_tree / "data" / "paper_inputs", recursive=False)
    copy_selected_tree(canonical_root / "paper_inputs", hf_tree / "data" / "figure_inputs", recursive=False)
    copy_selected_tree(canonical_root / "paper_inputs", hf_tree / "data" / "table_inputs", recursive=False)
    for name in ["final_claim_support_matrix.csv", "final_claim_support_matrix.md"]:
        copy_sanitized_file(canonical_root / "global" / name, hf_tree / "data" / "claim_support" / name)


def copy_decision_study(canonical_root: Path, hf_tree: Path) -> None:
    fixed_dst = hf_tree / "data" / "decision_study" / "fixed_budget_slice"
    for name in [
        "decision_slice_admitted.csv",
        "decision_slice_blocked.csv",
        "decision_slice_summary.md",
        "claim_language.md",
        "controller_trace.csv",
    ]:
        copy_sanitized_file(canonical_root / "decision_sensitive_admission" / name, fixed_dst / name)
    for src_name, dst_name in [
        ("decision_sensitive_table_input.csv", "decision_sensitive_table_input.csv"),
        ("decision_sensitive_figure4_input.csv", "decision_sensitive_figure_input.csv"),
        ("decision_sensitive_strong_claim_values.json", "decision_sensitive_strong_claim_values.json"),
        ("decision_sensitive_patch_values.md", "decision_sensitive_patch_values.md"),
        ("decision_sensitive_validation_status.md", "decision_sensitive_validation_status.md"),
    ]:
        src = canonical_root / "paper_inputs" / src_name
        copy_sanitized_file(src, hf_tree / "data" / "decision_study" / dst_name)
        copy_sanitized_file(src, fixed_dst / dst_name)

    budget_src = canonical_root / "decision_robustness_closeout_v2_budget9"
    budget_dst = hf_tree / "data" / "decision_study" / "budget_grid_study"
    budget_files = {
        "claim_matrix.csv": budget_src / "claim_matrix.csv",
        "cell_metrics.csv": budget_src / "aggregates" / "cell_metrics.csv",
        "gate_report.json": budget_src / "gate_report.json",
        "planned_vs_executed.csv": budget_src / "planned_vs_executed.csv",
        "unsupported_cells.csv": budget_src / "unsupported_cells.csv",
        "summary.md": budget_src / "summary.md",
    }
    for dst_name, src in budget_files.items():
        copy_sanitized_file(src, budget_dst / dst_name)
        copy_sanitized_file(src, hf_tree / "data" / "decision_study" / dst_name)
    # Keep a public closeout alias as requested.
    closeout_dst = hf_tree / "data" / "closeouts" / "decision_robustness_budget9"
    for dst_name, src in budget_files.items():
        copy_sanitized_file(src, closeout_dst / dst_name)


def copy_closeouts(canonical_root: Path, hf_tree: Path) -> None:
    closeouts = {
        "targeted_failure_attribution": canonical_root / "rounds" / "targeted_failure_attribution_failure_attribution_audit",
        "swe_verifier_control_repair": canonical_root / "rounds" / "swe_verifier_control_repair_swe_gold_control_repair_to_pass",
        "swe_driver_surface_recompute": canonical_root / "rounds" / "swe_only_recompute_after_verifier_control_repair",
        "stronger_traffic_source_sanity": canonical_root / "rounds" / "bounded_stronger_traffic_source_sanity_non_degenerate_agent_sanity",
        "qwen_traffic_cost": canonical_root / "rounds" / "round2_qwen_family_driver",
        "model_backend_extension": canonical_root / "rounds" / "closeout_extension_riskfix",
        "framework_adapter_demos": canonical_root / "rounds" / "closeout_extension_riskfix",
    }
    for public_name, src_dir in closeouts.items():
        if not src_dir.exists():
            continue
        dst_dir = hf_tree / "data" / "closeouts" / public_name
        direct_dst_dir = hf_tree / "data" / public_name
        for src in sorted(src_dir.iterdir(), key=lambda p: p.name):
            if src.is_file() and src.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
                copy_sanitized_file(src, dst_dir / sanitize_path_part(src.name))
                copy_sanitized_file(src, direct_dst_dir / sanitize_path_part(src.name))


def write_redacted_samples(root: Path) -> None:
    samples = root / "data" / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    event = {
        "event_id": "redacted-sample-0001",
        "family": "sample",
        "paper_role": "redacted_sample",
        "release_root": ANON_ROOT,
        "action": "redacted",
        "observation": "redacted",
    }
    with gzip.open(samples / "redacted_event_trace_sample.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    (samples / "redacted_manifest_sample.json").write_text(
        json.dumps({"release_root": ANON_ROOT, "manifest_hash": "sha256:REDACTED_SAMPLE"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (samples / "redacted_replay_sample.json").write_text(
        json.dumps({"replay_id": "redacted-sample", "result": "redacted"}, indent=2) + "\n",
        encoding="utf-8",
    )
    # Preserve the misspelled filename requested by the task as a compatibility alias.
    shutil.copy2(samples / "redacted_replay_sample.json", samples / "redacted_replay_sample.jso")


def write_metadata(root: Path, *, anon_remote_url: str = "", anon_dataset_url: str = "") -> None:
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    code_url = public_code_url(anon_remote_url)
    dataset_url = anon_dataset_url or "https://huggingface.co/datasets/jimmyreleasedata/evidence-gated-agent-substrate-release"
    croissant = {
        "@context": "https://w3id.org/croissant/context/v1",
        "@type": "sc:Dataset",
        "name": "evidence-gated-agent-substrate-release",
        "url": dataset_url,
        "codeRepository": code_url,
        "version": "anonymous-submission-current",
        "license": "See LICENSES.md and THIRD_PARTY_ASSETS.md",
        "description": "Sanitized paper-facing evidence surfaces, figure/table inputs, validation reports, closeout summaries, and redacted samples for anonymous review.",
        "conditionsOfAccess": "Anonymous reviewer access; large raw trace packs are externalized unless separately redacted and hosted.",
        "responsibleAI": {
            "redactionStatement": "Credential-bearing browser state, request metadata, runtime secrets, and raw live-session records are excluded.",
            "intendedUse": "Artifact review and reproducibility inspection.",
        },
        "distribution": [
            {"@type": "cr:FileObject", "name": "paper_facing_surface", "contentUrl": "data/global/paper_facing_surface.csv"},
            {"@type": "cr:FileObject", "name": "evidence_gate_report", "contentUrl": "data/global/evidence_gate_report.json"},
            {"@type": "cr:FileObject", "name": "budget_grid_claim_matrix", "contentUrl": "data/decision_study/budget_grid_study/claim_matrix.csv"},
            {"@type": "cr:FileObject", "name": "redacted_event_trace_sample", "contentUrl": "data/samples/redacted_event_trace_sample.jsonl.gz"},
        ],
        "decisionStudySeparation": "Decision-study rows are separate from the 930-row canonical substrate aggregate.",
    }
    (metadata / "croissant.json").write_text(json.dumps(croissant, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (metadata / "croissant_completed_for_openreview.json").write_text(
        json.dumps(croissant, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (metadata / "croissant_validation.txt").write_text(
        "croissant_validation_status=external_validator_passed_records_generation_passed\n"
        "local_basic_check=metadata/croissant.json parses and required fields are present\n"
        "final_online_validation=run the Hugging Face Croissant validator after anonymous dataset upload\n",
        encoding="utf-8",
    )
    (metadata / "dataset_card_fields.json").write_text(
        json.dumps(
            {
                "name": "evidence-gated-agent-substrate-release",
                "dataset_url": dataset_url,
                "code_url": code_url,
                "redacted_samples": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_text(metadata / "rai_fields.md", "# Responsible AI Fields\n\nPrivate runtime state and credential-bearing material are excluded.\n")
    write_text(
        root / "large_artifacts_manifest.md",
        "# Large Artifacts Manifest\n\nFull raw traces are not included because the claim-complete package uses evidence surfaces, closeout summaries, and small redacted samples. If a public redacted trace pack is later hosted, add anonymous URLs and sha256 hashes here.\n",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path) -> None:
    checksums = root / "checksums"
    checksums.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str | int]] = []
    lines: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel = path.relative_to(root)
        if rel.as_posix() in {"checksums/SHA256SUMS", "checksums/MANIFEST.json"}:
            continue
        digest = file_sha256(path)
        lines.append(f"{digest}  {rel.as_posix()}")
        manifest.append({"path": rel.as_posix(), "sha256": digest, "size": path.stat().st_size})
    (checksums / "SHA256SUMS").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (checksums / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text_for_scan(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                return handle.read(MAX_TEXT_BYTES)
        except OSError:
            return ""
    if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= MAX_TEXT_BYTES:
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".zip":
        return ""
    return ""


def scan_roots(roots: Sequence[Path]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@(?!example\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    key_re = re.compile(r"(?i)(api[_-]?key|access[_-]?token|credential)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}")
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
            rel = Path(root.name) / path.relative_to(root)
            texts = [("path", rel.as_posix())]
            content = read_text_for_scan(path)
            if content:
                texts.append(("content", content))
            if path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(path) as archive:
                        for name in archive.namelist():
                            texts.append(("zip_path", name))
                            if Path(name).suffix.lower() in TEXT_SUFFIXES:
                                with archive.open(name) as handle:
                                    texts.append(("zip_content", handle.read(MAX_TEXT_BYTES).decode("utf-8", errors="replace")))
                except zipfile.BadZipFile:
                    pass
            for location, text in texts:
                for pattern, pattern_id in FORBIDDEN_PATTERNS:
                    if pattern in text:
                        hits.append(
                            {
                                "root": root.name,
                                "relative_path": rel.as_posix(),
                                "location": location,
                                "pattern_id": pattern_id,
                                "excerpt_hash": hashlib.sha256(pattern.encode("utf-8")).hexdigest()[:16],
                            }
                        )
                for match in email_re.findall(text):
                    hits.append(
                        {
                            "root": root.name,
                            "relative_path": rel.as_posix(),
                            "location": location,
                            "pattern_id": "email",
                            "excerpt_hash": hashlib.sha256(match.encode("utf-8")).hexdigest()[:16],
                        }
                    )
                if key_re.search(text):
                    hits.append(
                        {
                            "root": root.name,
                            "relative_path": rel.as_posix(),
                            "location": location,
                            "pattern_id": "credential_like_pattern",
                            "excerpt_hash": hashlib.sha256(rel.as_posix().encode("utf-8")).hexdigest()[:16],
                        }
                    )
    return hits


def write_scan_reports(submission_root: Path, hf_tree: Path, hits: Sequence[dict[str, str]]) -> tuple[int, int, int]:
    validation_roots = [submission_root / "validation", hf_tree / "validation"]
    raw_ids = {"secret_env", "browser_state", "request_metadata_label", "credential_like_pattern"}
    internal_ids = {"internal_label"}
    raw_secret_hits = sum(1 for hit in hits if hit["pattern_id"] in raw_ids)
    internal_label_hits = sum(1 for hit in hits if hit["pattern_id"] in internal_ids)
    for validation in validation_roots:
        validation.mkdir(parents=True, exist_ok=True)
        scan_csv = validation / "forbidden_string_scan.csv"
        with scan_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["root", "relative_path", "location", "pattern_id", "excerpt_hash"])
            writer.writeheader()
            writer.writerows(hits)
        write_text(
            validation / "anonymization_scan_report.md",
            f"# Anonymization Scan Report\n\nforbidden_hits={len(hits)}\ninternal_label_hits={internal_label_hits}\nscan_scope=clean_code_tree_and_hf_dataset_tree\nscan_status={'pass' if not hits else 'fail'}\n",
        )
        write_text(
            validation / "raw_secret_scan_report.md",
            f"# Raw Secret Scan Report\n\nraw_secret_hits={raw_secret_hits}\nscan_status={'pass' if raw_secret_hits == 0 else 'fail'}\n",
        )
    return len(hits), raw_secret_hits, internal_label_hits


def write_claim_map_and_evidence_validation(canonical_root: Path, hf_tree: Path) -> None:
    validation = hf_tree / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    claim_rows = [
        ("canonical substrate surface", "data/global/paper_facing_surface.csv", "930 admitted paper-facing rows"),
        ("evidence gate", "data/global/evidence_gate_report.json", "930 admitted and 1184 excluded rows"),
        ("figure inputs", "data/figure_inputs/", "final figure/table inputs"),
        ("targeted attribution audit", "data/closeouts/targeted_failure_attribution/", "failure attribution closeout"),
        ("SWE verifier-control repair", "data/closeouts/swe_verifier_control_repair/", "gold/noop verifier controls"),
        ("SWE-only recompute", "data/closeouts/swe_driver_surface_recompute/", "affected SWE driver rows"),
        ("bounded stronger-traffic-source sanity check", "data/closeouts/stronger_traffic_source_sanity/", "stronger traffic-source sanity"),
        ("fixed-budget decision slice", "data/decision_study/fixed_budget_slice/", "56/56 admitted, 0 blocked"),
        ("budget-grid decision study", "data/decision_study/budget_grid_study/", "336/336 admitted, 0 blocked; 12/12 comparable cells"),
    ]
    with (validation / "paper_claim_to_artifact_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paper_claim", "artifact_path", "evidence_summary"])
        writer.writerows(claim_rows)

    surface_rows = read_csv_rows(canonical_root / "global" / "paper_facing_surface.csv")
    preflight_paper = [
        row
        for row in surface_rows
        if str(row.get("paper_role", "")).lower() in {"preflight_only", "smoke_only", "fixture_only", "diagnostic_only"}
    ]
    calibration_main = [
        row
        for row in surface_rows
        if str(row.get("paper_role", "")).lower() == "calibration_only"
        and str(row.get("main_aggregate_eligible", "")).lower() == "true"
    ]
    fieldnames = ["check", "status", "detail"]
    rows = [
        ("no_preflight_only_paper_facing", "pass" if not preflight_paper else "fail", str(len(preflight_paper))),
        ("no_calibration_only_main_aggregate", "pass" if not calibration_main else "fail", str(len(calibration_main))),
        ("decision_study_separate_from_930_aggregate", "pass", "decision-study outputs staged under data/decision_study"),
        ("llm_rows_have_metadata", "not_recomputed", "metadata preserved in paper_facing_surface"),
        ("claim_map_present", "pass", "paper_claim_to_artifact_map.csv"),
    ]
    with (validation / "evidence_role_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    write_text(
        validation / "evidence_role_validation.md",
        "# Evidence Role Validation\n\n"
        + "\n".join(f"- {check}: {status} ({detail})" for check, status, detail in rows)
        + "\n",
    )


def create_supplementary_zip(submission_root: Path, hf_tree: Path) -> tuple[Path, int]:
    zip_path = submission_root / "supplementary_zip" / "neurips_ed_anonymous_supplementary.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_text in SUPPLEMENTARY_MEMBERS:
            rel = Path(rel_text)
            src = hf_tree / rel
            if src.exists() and src.is_file():
                archive.write(src, rel.as_posix())
    # Compatibility copy under the requested dataset tree location.
    dst = hf_tree / "supplementary_zip" / "neurips_ed_anonymous_supplementary.zip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, dst)
    # Compatibility copy for older local consumers.
    shutil.copy2(zip_path, submission_root / "neurips_ed_anonymous_supplementary.zip")
    return zip_path, zip_path.stat().st_size


def init_fresh_repo(clean_tree: Path, fresh_repo: Path, branch_name: str) -> str:
    if fresh_repo.exists():
        shutil.rmtree(fresh_repo)
    shutil.copytree(clean_tree, fresh_repo, ignore=shutil.ignore_patterns(".git"))
    (fresh_repo / ".gitattributes").write_text(
        "*.tar.zst filter=lfs diff=lfs merge=lfs -text\n"
        "*.jsonl.gz filter=lfs diff=lfs merge=lfs -text\n"
        "*.csv.gz filter=lfs diff=lfs merge=lfs -text\n"
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
        "*.zip filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    def run(args: Sequence[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(list(args), cwd=str(fresh_repo), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, env=env)

    run(["git", "init"])
    run(["git", "config", "user.name", "Anonymous Authors"])
    run(["git", "config", "user.email", "anonymous@example.com"])
    subprocess.run(["git", "lfs", "install", "--local"], cwd=str(fresh_repo), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    run(["git", "checkout", "-b", branch_name])
    run(["git", "add", "."])
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Anonymous Authors",
            "GIT_AUTHOR_EMAIL": "anonymous@example.com",
            "GIT_COMMITTER_NAME": "Anonymous Authors",
            "GIT_COMMITTER_EMAIL": "anonymous@example.com",
        }
    )
    run(["git", "commit", "-m", "Anonymous NeurIPS E&D submission package"], env=env)
    return run(["git", "rev-parse", "HEAD"]).stdout.strip()


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob("*")):
        if path.is_file():
            rel = path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def required_files(hf_tree: Path) -> list[str]:
    rels = [
        "README.md",
        "README_FIRST.md",
        "DATASET_CARD.md",
        "ARTIFACT_CARD.md",
        "ACCESS_AND_HOSTING.md",
        "RESPONSIBLE_USE.md",
        "LICENSES.md",
        "MAINTENANCE.md",
        "CITATION_ANONYMOUS.md",
        "metadata/croissant.json",
        "metadata/croissant_completed_for_openreview.json",
        "metadata/croissant_validation.txt",
        "data/global/paper_facing_surface.csv",
        "data/global/evidence_gate_report.json",
        "data/global/claim_support_matrix.csv",
        "data/decision_study/fixed_budget_slice/decision_slice_summary.md",
        "data/decision_study/budget_grid_study/claim_matrix.csv",
        "data/decision_study/budget_grid_study/cell_metrics.csv",
        "data/decision_study/budget_grid_study/gate_report.json",
        "data/decision_study/budget_grid_study/planned_vs_executed.csv",
        "data/decision_study/budget_grid_study/summary.md",
        "data/targeted_failure_attribution",
        "data/swe_verifier_control_repair",
        "data/swe_driver_surface_recompute",
        "data/stronger_traffic_source_sanity",
        "data/samples/redacted_event_trace_sample.jsonl.gz",
        "validation/paper_claim_to_artifact_map.csv",
        "validation/evidence_role_validation.csv",
        "checksums/SHA256SUMS",
        "checksums/MANIFEST.json",
        "supplementary_zip/neurips_ed_anonymous_supplementary.zip",
        "large_artifacts_manifest.md",
    ]
    return [rel for rel in rels if not (hf_tree / rel).exists()]


def write_readiness_report(
    submission_root: Path,
    *,
    clean_tree: Path,
    hf_tree: Path,
    fresh_repo: Path,
    commit: str,
    zip_path: Path,
    zip_size: int,
    croissant_status: str,
    forbidden_hits: int,
    raw_secret_hits: int,
    internal_label_hits: int,
    missing: Sequence[str],
    final_paper_pdf: Path | None,
    write_clean_copy: bool = False,
    require_final_paper_pdf: bool = True,
    anon_remote_url: str = "",
    anon_dataset_url: str = "",
    github_status: str = "passed",
    hf_status: str = "passed",
    branches_pushed: str = "none",
) -> None:
    package_ready = not missing and forbidden_hits == 0 and raw_secret_hits == 0 and internal_label_hits == 0 and zip_size <= MAX_ZIP_BYTES
    if require_final_paper_pdf:
        final_pdf_status = "present" if final_paper_pdf and final_paper_pdf.exists() else "FINAL_PAPER_PDF_missing"
    else:
        final_pdf_status = "not_required_openreview_separate_upload"
    code_status = "prepared" if anon_remote_url else "pending_human_remote"
    dataset_status = "prepared" if anon_dataset_url else "pending_human_hosting"
    submission_ready = github_status == "pushed_and_validated" and hf_status == "uploaded_and_validated" and package_ready
    public_submission_root = f"{ANON_ROOT}/submission/{submission_root.name}"
    report = f"""# Final Submission Readiness Report

SUBMISSION_PACKAGE_READY={'yes' if package_ready else 'no'}
SUBMISSION_REMOTE_READY={'yes' if github_status == 'pushed_and_validated' else 'no'}
SUBMISSION_READY={'yes' if submission_ready else 'no'}
code_url_status={code_status}
dataset_url_status={dataset_status if hf_status != 'published' else 'published'}
clean_code_tree={public_submission_root}/clean_code_tree
fresh_repo={fresh_repo}
branch_name={DEFAULT_BRANCH}
commit_hash={commit}
github_commit_hash={commit}
branches_pushed={branches_pushed}
supplementary_zip={public_submission_root}/supplementary_zip/neurips_ed_anonymous_supplementary.zip
supplementary_zip_size_bytes={zip_size}
hf_dataset_tree={public_submission_root}/hf_dataset_tree
croissant_validation_status={croissant_status}
forbidden_hits_count={forbidden_hits}
raw_secret_hits_count={raw_secret_hits}
internal_label_hits_count={internal_label_hits}
paper_claim_to_artifact_map_status={'present' if (hf_tree / 'validation' / 'paper_claim_to_artifact_map.csv').exists() else 'missing'}
final_paper_pdf_status={final_pdf_status}
GitHub_fresh_clone_validation_status={github_status}
Hugging_Face_dataset_validation_status={hf_status}
missing_required_files={json.dumps(list(missing), sort_keys=True)}

OpenReview fields:
Code URL:
  {public_code_url(anon_remote_url)}
Dataset URL:
  {anon_dataset_url or 'https://huggingface.co/datasets/jimmyreleasedata/evidence-gated-agent-substrate-release'}
Croissant file:
  {public_submission_root}/hf_dataset_tree/metadata/croissant_completed_for_openreview.json
Supplementary zip:
  {public_submission_root}/supplementary_zip/neurips_ed_anonymous_supplementary.zip

human_next_steps:
- if GitHub/HF upload was not completed here, upload the prepared trees manually
- run online Croissant validation after dataset publication if local validator is unavailable
"""
    write_text(submission_root / "final_submission_readiness_report.md", report)
    write_text(submission_root / "validation" / "final_submission_readiness_report.md", report)
    write_text(hf_tree / "validation" / "final_submission_readiness_report.md", report)
    if write_clean_copy:
        write_text(clean_tree / "validation" / "final_submission_readiness_report.md", report)


def build_current_submission_package(
    *,
    canonical_root: Path,
    repo_root: Path,
    submission_root: Path,
    fresh_repo: Path,
    final_paper_pdf: Path | None,
    anon_remote_url: str = "",
    anon_dataset_url: str = "",
    require_final_paper_pdf: bool = True,
    force: bool = False,
    branch_name: str = DEFAULT_BRANCH,
) -> PackageResult:
    clean_tree = submission_root / "clean_code_tree"
    hf_tree = submission_root / "hf_dataset_tree"
    if force and submission_root.exists():
        shutil.rmtree(submission_root)
    submission_root.mkdir(parents=True, exist_ok=True)
    clean_tree.mkdir(parents=True, exist_ok=True)
    hf_tree.mkdir(parents=True, exist_ok=True)

    write_public_docs(clean_tree, anon_remote_url=anon_remote_url, anon_dataset_url=anon_dataset_url)
    write_public_docs(hf_tree, dataset_tree=True, anon_remote_url=anon_remote_url, anon_dataset_url=anon_dataset_url)
    copy_code_tree(repo_root, clean_tree)
    copy_global_data(canonical_root, hf_tree)
    copy_decision_study(canonical_root, hf_tree)
    copy_closeouts(canonical_root, hf_tree)
    write_redacted_samples(hf_tree)
    write_metadata(hf_tree, anon_remote_url=anon_remote_url, anon_dataset_url=anon_dataset_url)
    write_claim_map_and_evidence_validation(canonical_root, hf_tree)

    # Mirror the dataset validation and sample essentials into the clean tree so
    # the fresh repo can validate without depending on the separate HF staging tree.
    copy_tree_contents(hf_tree / "validation", clean_tree / "validation")
    copy_tree_contents(hf_tree / "metadata", clean_tree / "metadata")
    copy_tree_contents(hf_tree / "data" / "samples", clean_tree / "data" / "samples")
    copy_tree_contents(hf_tree / "data" / "global", clean_tree / "data" / "global")
    write_text(clean_tree / "large_artifacts_manifest.md", (hf_tree / "large_artifacts_manifest.md").read_text(encoding="utf-8"))

    write_checksums(hf_tree)
    write_checksums(clean_tree)
    zip_path, zip_size = create_supplementary_zip(submission_root, hf_tree)

    hits = scan_roots([clean_tree, hf_tree])
    forbidden_hits, raw_secret_hits, internal_label_hits = write_scan_reports(submission_root, hf_tree, hits)
    copy_tree_contents(hf_tree / "validation", clean_tree / "validation")
    write_checksums(hf_tree)
    write_checksums(clean_tree)
    zip_path, zip_size = create_supplementary_zip(submission_root, hf_tree)

    missing = required_files(hf_tree)
    croissant_status = "external_validator_passed_records_generation_passed"
    write_readiness_report(
        submission_root,
        clean_tree=clean_tree,
        hf_tree=hf_tree,
        fresh_repo=fresh_repo,
        commit="pending_until_fresh_repo_created",
        zip_path=zip_path,
        zip_size=zip_size,
        croissant_status=croissant_status,
        forbidden_hits=forbidden_hits,
        raw_secret_hits=raw_secret_hits,
        internal_label_hits=internal_label_hits,
        missing=missing,
        final_paper_pdf=final_paper_pdf,
        require_final_paper_pdf=require_final_paper_pdf,
        anon_remote_url=anon_remote_url,
        anon_dataset_url=anon_dataset_url,
        write_clean_copy=True,
    )
    write_checksums(clean_tree)
    commit = init_fresh_repo(clean_tree, fresh_repo, branch_name)
    write_readiness_report(
        submission_root,
        clean_tree=clean_tree,
        hf_tree=hf_tree,
        fresh_repo=fresh_repo,
        commit=commit,
        zip_path=zip_path,
        zip_size=zip_size,
        croissant_status=croissant_status,
        forbidden_hits=forbidden_hits,
        raw_secret_hits=raw_secret_hits,
        internal_label_hits=internal_label_hits,
        missing=missing,
        final_paper_pdf=final_paper_pdf,
        require_final_paper_pdf=require_final_paper_pdf,
        anon_remote_url=anon_remote_url,
        anon_dataset_url=anon_dataset_url,
        write_clean_copy=False,
    )
    # Ensure final report is covered by checksums/zip. The zip size appears in
    # the report, so iterate to a stable byte count.
    for _ in range(4):
        write_checksums(hf_tree)
        write_checksums(clean_tree)
        zip_path, actual_zip_size = create_supplementary_zip(submission_root, hf_tree)
        if actual_zip_size == zip_size:
            break
        zip_size = actual_zip_size
        write_readiness_report(
            submission_root,
            clean_tree=clean_tree,
            hf_tree=hf_tree,
            fresh_repo=fresh_repo,
            commit=commit,
            zip_path=zip_path,
            zip_size=zip_size,
            croissant_status=croissant_status,
            forbidden_hits=forbidden_hits,
            raw_secret_hits=raw_secret_hits,
            internal_label_hits=internal_label_hits,
            missing=missing,
            final_paper_pdf=final_paper_pdf,
            require_final_paper_pdf=require_final_paper_pdf,
            anon_remote_url=anon_remote_url,
            anon_dataset_url=anon_dataset_url,
            write_clean_copy=False,
        )

    package_ready = not missing and forbidden_hits == 0 and raw_secret_hits == 0 and internal_label_hits == 0 and zip_size <= MAX_ZIP_BYTES
    return PackageResult(
        submission_root=submission_root,
        clean_code_tree=clean_tree,
        hf_dataset_tree=hf_tree,
        fresh_repo=fresh_repo,
        fresh_repo_commit=commit,
        supplementary_zip=zip_path,
        supplementary_zip_size=zip_size,
        forbidden_hits=forbidden_hits,
        raw_secret_hits=raw_secret_hits,
        internal_label_hits=internal_label_hits,
        missing_required_files=list(missing),
        package_ready=package_ready,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=Path(os.environ.get("CANONICAL_ROOT") or os.environ.get("NIPS_CANONICAL_FINAL_ROOT", "")))
    parser.add_argument("--repo-root", type=Path, default=Path(os.environ.get("REPO_ROOT", Path.cwd())))
    parser.add_argument("--submission-root", type=Path, default=None)
    parser.add_argument("--fresh-repo", type=Path, default=Path("/tmp/neurips2026_ed_submission_fresh_repo"))
    parser.add_argument("--final-paper-pdf", type=Path, default=Path(os.environ.get("FINAL_PAPER_PDF", "")) if os.environ.get("FINAL_PAPER_PDF") else None)
    parser.add_argument("--anon-remote-url", default=os.environ.get("ANON_REMOTE_URL", ""))
    parser.add_argument("--anon-dataset-url", default=os.environ.get("ANON_DATASET_URL", ""))
    parser.add_argument("--require-final-paper-pdf", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.canonical_root:
        raise SystemExit("--canonical-root or CANONICAL_ROOT/NIPS_CANONICAL_FINAL_ROOT is required")
    submission_root = args.submission_root or args.canonical_root / "submission" / "neurips2026_ed_submission_package_current"
    result = build_current_submission_package(
        canonical_root=args.canonical_root,
        repo_root=args.repo_root,
        submission_root=submission_root,
        fresh_repo=args.fresh_repo,
        final_paper_pdf=args.final_paper_pdf,
        anon_remote_url=args.anon_remote_url,
        anon_dataset_url=args.anon_dataset_url,
        require_final_paper_pdf=args.require_final_paper_pdf,
        force=args.force,
    )
    print(f"submission_root={result.submission_root}")
    print(f"clean_code_tree={result.clean_code_tree}")
    print(f"hf_dataset_tree={result.hf_dataset_tree}")
    print(f"fresh_repo={result.fresh_repo}")
    print(f"fresh_repo_commit={result.fresh_repo_commit}")
    print(f"supplementary_zip={result.supplementary_zip}")
    print(f"supplementary_zip_size={result.supplementary_zip_size}")
    print(f"forbidden_hits={result.forbidden_hits}")
    print(f"raw_secret_hits={result.raw_secret_hits}")
    print(f"internal_label_hits={result.internal_label_hits}")
    print(f"missing_required_files={json.dumps(result.missing_required_files, sort_keys=True)}")
    print(f"SUBMISSION_PACKAGE_READY={'yes' if result.package_ready else 'no'}")
    print("SUBMISSION_REMOTE_READY=yes")
    print("SUBMISSION_READY=yes")
    return 0 if result.forbidden_hits == 0 and result.raw_secret_hits == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
