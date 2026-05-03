#!/usr/bin/env python3
"""Build the anonymous NeurIPS E&D artifact submission tree.

This builder is packaging-only. It never runs experiments and it never uses the
current repository's Git metadata. When Git export is enabled, a fresh repository
is initialized under a caller-provided scratch directory, committed there, and
optionally bundled back into the submission output.
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
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ANON_ROOT = "artifact_release_root"
ANON_HOME = "ANON_HOME"
DEFAULT_BRANCH = "ed-paper-submission"
MAX_TEXT_COPY_BYTES = 8 * 1024 * 1024
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
    "node_modules",
    "raw",
    "cache",
    "caches",
    "logs",
    "runs",
    "tmp",
    "temporary",
}

SKIP_FILE_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".a", ".png", ".jpg", ".jpeg", ".gif", ".pdf"}

PUBLIC_DOCS = {
    "README.md": """# Anonymous E&D Artifact Submission

This clean export contains sanitized source code, final paper-facing statistics,
figure and table inputs, validation reports, redacted samples, metadata, and
checksums for the anonymous artifact submission.

The bundle separates executable substrate evidence from traffic-source results.
Calibration-only rows are not part of main quantitative aggregates.
""",
    "INSTALL.md": """# Install

Use a recent Python 3 environment with pandas, pytest, and PyYAML available for
local validation. Large external runtimes are intentionally not included in this
anonymous source bundle.
""",
    "RUNME.md": """# Runme

Smoke validation:

```bash
python validate_bundle.py
```

This checks required files, metadata, checksums, and public redaction status. It
does not run benchmark experiments.
""",
    "REPRODUCE.md": """# Reproduce

The included surfaces and figure inputs are final evidence-gated products. The
source code under `code/` documents how the surfaces were produced. Runtime-heavy
experiments are outside this anonymous packaging step.
""",
    "SECURITY_AND_REDACTION.md": """# Security And Redaction

This export excludes private browser state, credential-bearing request metadata,
model weights, raw live-session material, and unreleased runtime images. Private
filesystem roots are replaced with anonymous path tokens.
""",
    "RESPONSIBLE_USE.md": """# Responsible Use

The artifact is intended for benchmark reproducibility review. It should not be
used to recover private runtime state or identify operators, allocation details,
or deployment infrastructure.
""",
    "LICENSES.md": """# Licenses

Code is provided for anonymous review under the project license terms. External
benchmark tasks and third-party assets retain their original licenses and access
conditions.
""",
    "MAINTENANCE.md": """# Maintenance

The public export is a sanitized snapshot. Future public releases should repeat
the redaction scan and checksum generation before publication.
""",
}

PUBLIC_LABEL_REPLACEMENTS = [
    ("targeted attribution audit", "targeted attribution audit"),
    ("targeted attribution audit", "targeted attribution audit"),
    ("targeted_failure_attribution", "targeted_failure_attribution"),
    ("targeted attribution quality check", "targeted attribution QC"),
    ("targeted attribution quality check", "targeted attribution QC"),
    ("targeted_attribution_quality_check", "targeted_attribution_qc"),
    ("SWE verifier-control repair", "SWE verifier-control repair"),
    ("SWE verifier-control repair", "SWE verifier-control repair"),
    ("swe_verifier_control_repair", "swe_verifier_control_repair"),
    ("SWE driver-surface recompute", "SWE driver-surface recompute"),
    ("SWE driver-surface recompute", "SWE driver-surface recompute"),
    ("swe_driver_surface_recompute", "swe_driver_surface_recompute"),
    ("SWE verifier closeout", "SWE verifier closeout"),
    ("SWE verifier closeout", "SWE verifier closeout"),
    ("swe_verifier_closeout", "swe_verifier_closeout"),
    ("bounded stronger-traffic-source sanity check", "bounded stronger-traffic-source sanity check"),
    ("bounded stronger-traffic-source sanity check", "bounded stronger-traffic-source sanity check"),
    ("bounded_stronger_traffic_source_sanity", "bounded_stronger_traffic_source_sanity"),
    ("targeted attribution audit", "targeted attribution audit"),
    ("targeted attribution audit", "targeted attribution audit"),
    ("targeted_failure_attribution", "targeted_failure_attribution"),
    ("initial evidence pass", "initial evidence pass"),
    ("secondary evidence pass", "secondary evidence pass"),
    ("closeout pass", "closeout pass"),
    ("closeout", "closeout"),
    ("closeout", "closeout"),
    ("closeout", "closeout"),
    ("pending SWE verifier-control repair", "pending verifier-control repair"),
    ("SWE verifier-control repair", "verifier-control repair"),
    ("SWE driver-surface recompute", "SWE driver-surface recompute"),
    ("closeout", "closeout"),
    ("reviewer_risk_fix", "closeout"),
    ("evidence-stratified", "evidence-stratified"),
    ("round_aware", "evidence_stratified"),
    ("recompute_required_status", "recompute_required_status"),
]

PRIVATE_REPLACEMENTS = [
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
    ("ANON_USER", "ANON_USER"),
    ("anonymous_user", "anonymous_user"),
    ("anonymous_user", "anonymous_user"),
    ("anonymous_allocation", "anonymous_allocation"),
]

SECRET_REPLACEMENTS = [
    ("REDACTED_SECRET_ENV", "REDACTED_SECRET_ENV"),
    ("REDACTED_REQUEST_METADATA_ENV", "REDACTED_REQUEST_METADATA_ENV"),
    ("REDACTED_BROWSER_STATE_LABEL", "REDACTED_BROWSER_STATE_LABEL"),
    ("REDACTED_BROWSER_STATE_LABEL", "REDACTED_BROWSER_STATE_LABEL"),
    ("REDACTED_BROWSER_STATE_LABEL", "REDACTED_BROWSER_STATE_LABEL"),
    ("REDACTED_BROWSER_STATE_LABEL", "REDACTED_BROWSER_STATE_LABEL"),
    ("REDACTED_BROWSER_STATE_LABEL", "REDACTED_BROWSER_STATE_LABEL"),
    ("credential-bearing request metadata", "credential-bearing request metadata"),
]

FORBIDDEN_PATTERNS = [
    ("artifact_release_root/", "private_root"),
    ("ANON_HOME/", "private_home"),
    ("ANON_USER", "private_user"),
    ("anonymous_user", "private_user"),
    ("anonymous_allocation", "private_allocation"),
    ("REDACTED_SECRET_ENV", "secret_env"),
    ("REDACTED_REQUEST_METADATA_ENV", "secret_env"),
    ("REDACTED_BROWSER_STATE_LABEL", "browser_state"),
    ("REDACTED_BROWSER_STATE_LABEL", "browser_state"),
    ("credential-bearing request metadata", "credential_headers"),
    ("closeout", "internal_closeout_label"),
    ("initial evidence pass", "internal_pass_label"),
    ("secondary evidence pass", "internal_pass_label"),
    ("closeout pass", "internal_pass_label"),
    ("SWE verifier-control repair", "internal_repair_label"),
    ("SWE driver-surface recompute", "internal_recompute_label"),
    ("closeout", "internal_label"),
    ("evidence-stratified", "internal_label"),
    ("recompute_required_status", "internal_label"),
]

ALLOWED_SCAN_OUTPUTS = {
    Path("validation/anonymous_repo_forbidden_string_scan.csv"),
}


@dataclass(frozen=True)
class BundleResult:
    clean_tree: Path
    submission_root: Path
    git_repo_path: Path | None
    git_commit: str
    git_bundle_path: Path | None
    branch_ready: bool
    forbidden_hits: int
    raw_secret_hits: int
    zip_size: int
    branch_name: str
    git_creation_status: str


def sanitize_public_text(text: str) -> str:
    sanitized = text
    for old, new in PRIVATE_REPLACEMENTS:
        sanitized = sanitized.replace(old, new)
    for old, new in PUBLIC_LABEL_REPLACEMENTS:
        sanitized = sanitized.replace(old, new)
    for old, new in SECRET_REPLACEMENTS:
        sanitized = sanitized.replace(old, new)
    sanitized = re.sub(r"git@github\.com:[^\s\"')]+", "ANON_REMOTE_URL", sanitized)
    sanitized = re.sub(r"https://github\.com/[^\s\"')]+", "ANON_REMOTE_URL", sanitized)
    sanitized = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "anonymous@example.com", sanitized)
    return sanitized


def sanitize_path_part(part: str) -> str:
    sanitized = sanitize_public_text(part)
    sanitized = sanitized.replace(" ", "_").replace("/", "_")
    sanitized = re.sub(r"[^A-Za-z0-9._+=-]", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "artifact"


def sanitize_relative_path(path: Path) -> Path:
    return Path(*[sanitize_path_part(part) for part in path.parts])


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def should_skip_path(path: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_FILE_SUFFIXES:
        return True
    name = path.name.lower()
    sensitive_fragments = [
        "credential",
        "credential",
        "passwd",
        "secret",
        "token",
        "private",
        "session",
    ]
    return any(fragment in name for fragment in sensitive_fragments)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sanitize_public_text(text), encoding="utf-8")


def copy_sanitized_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file() or should_skip_path(Path(src.name)):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_text_file(src) and src.stat().st_size <= MAX_TEXT_COPY_BYTES:
        dst.write_text(sanitize_public_text(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
        return True
    if src.stat().st_size <= MAX_TEXT_COPY_BYTES:
        data = src.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return False
        dst.write_text(sanitize_public_text(text), encoding="utf-8")
        return True
    return False


def copy_selected_tree(src_root: Path, dst_root: Path, *, recursive: bool = True) -> int:
    if not src_root.exists() or not src_root.is_dir():
        return 0
    copied = 0
    iterator: Iterable[Path]
    if recursive:
        iterator = (path for path in src_root.rglob("*") if path.is_file())
    else:
        iterator = (path for path in src_root.iterdir() if path.is_file())
    for src in iterator:
        rel = src.relative_to(src_root)
        if should_skip_path(rel):
            continue
        if src.suffix.lower() not in TEXT_SUFFIXES:
            continue
        dst = dst_root / sanitize_relative_path(rel)
        if copy_sanitized_file(src, dst):
            copied += 1
    return copied


def write_public_docs(root: Path) -> None:
    for name, content in PUBLIC_DOCS.items():
        write_text(root / name, content)
    write_text(
        root / "metadata" / "LARGE_ARTIFACTS.md",
        """# Large Artifact Pointers

Full trace archives are externalized for anonymous review unless a separately
redacted archive is provided. Public releases should replace the placeholders
below with anonymous dataset URLs and sha256 hashes.

| artifact | location | sha256 |
| --- | --- | --- |
| redacted full traces | DATASET_URL_PLACEHOLDER | SHA256_PLACEHOLDER |
""",
    )
    write_text(
        root / "validate_bundle.py",
        """#!/usr/bin/env python3
from pathlib import Path

required = [
    "README.md",
    "metadata/croissant.json",
    "checksums/SHA256SUMS",
    "validation/anonymous_repo_scan_report.md",
    "data/global/paper_facing_surface.csv",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))
print("anonymous bundle smoke validation passed")
""",
    )


def copy_final_data(canonical_root: Path, out: Path) -> dict[str, int]:
    counts = {"global": 0, "figure_inputs": 0, "claim_support": 0, "closeouts": 0}
    global_files = [
        "paper_facing_surface.csv",
        "evidence_gate_report.json",
        "excluded_rows.csv",
        "validation_report.json",
        "figure_input_inventory.csv",
        "final_claim_support_matrix.csv",
        "final_claim_support_matrix.md",
    ]
    for name in global_files:
        if copy_sanitized_file(canonical_root / "global" / name, out / "data" / "global" / name):
            counts["global"] += 1

    if copy_sanitized_file(
        canonical_root / "global" / "figure_input_inventory.csv",
        out / "data" / "figure_inputs" / "figure_input_inventory.csv",
    ):
        counts["figure_inputs"] += 1
    for name in ["final_claim_support_matrix.csv", "final_claim_support_matrix.md"]:
        if copy_sanitized_file(canonical_root / "global" / name, out / "data" / "claim_support" / name):
            counts["claim_support"] += 1

    closeout_sources = {
        "targeted_failure_attribution": canonical_root / "rounds" / "targeted_failure_attribution_failure_attribution_audit",
        "swe_verifier_control_repair": canonical_root / "rounds" / "swe_verifier_control_repair_swe_gold_control_repair_to_pass",
        "swe_driver_surface_recompute": canonical_root / "rounds" / "swe_only_recompute_after_verifier_control_repair",
        "bounded_stronger_traffic_source_sanity": canonical_root / "rounds" / "bounded_stronger_traffic_source_sanity_non_degenerate_agent_sanity",
    }
    for public_name, src_dir in closeout_sources.items():
        if not src_dir.exists():
            continue
        for src in sorted(src_dir.iterdir(), key=lambda p: p.name):
            if src.is_file() and src.suffix.lower() in {".csv", ".md", ".json", ".txt"}:
                if copy_sanitized_file(src, out / "data" / "closeouts" / public_name / sanitize_path_part(src.name)):
                    counts["closeouts"] += 1
    return counts


def copy_source_code(source_root: Path, out: Path) -> dict[str, int]:
    code_root = out / "code"
    counts = {
        "scripts_final": copy_selected_tree(source_root / "scripts" / "final", code_root / "scripts" / "final", recursive=False),
        "scripts_repro": copy_selected_tree(source_root / "scripts" / "repro", code_root / "scripts" / "repro", recursive=True),
        "tests": copy_selected_tree(source_root / "tests", code_root / "tests", recursive=False),
        "runtime": copy_selected_tree(source_root / "runtime", code_root / "runtime", recursive=True),
        "runner": copy_selected_tree(source_root / "runner", code_root / "runner", recursive=True),
        "trace": copy_selected_tree(source_root / "trace", code_root / "trace", recursive=True),
        "adapters": copy_selected_tree(source_root / "adapters", code_root / "adapters", recursive=True),
        "evaluators": copy_selected_tree(source_root / "evaluators", code_root / "evaluators", recursive=True),
        "collectors": copy_selected_tree(source_root / "collectors", code_root / "collectors", recursive=True),
        "configs": copy_selected_tree(source_root / "configs", code_root / "configs", recursive=True),
    }
    for rel in ["pyproject.toml", "README.md", "release_manifest.yaml"]:
        copy_sanitized_file(source_root / rel, code_root / sanitize_path_part(rel))
    return counts


def write_redacted_samples(out: Path) -> None:
    sample_event = {
        "event_id": "sample-0001",
        "paper_role": "redacted_sample",
        "family": "sample_family",
        "path_root": ANON_ROOT,
        "payload": {"action": "redacted_action", "observation": "redacted_observation"},
    }
    sample_path = out / "data" / "samples" / "redacted_event_trace_sample.jsonl.gz"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(sample_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(sample_event, sort_keys=True) + "\n")
    (out / "data" / "samples" / "redacted_manifest_sample.json").write_text(
        json.dumps(
            {
                "release_root": ANON_ROOT,
                "manifest_hash": "sha256:REDACTED_SAMPLE",
                "paper_role": "redacted_sample",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "data" / "samples" / "redacted_replay_sample.json").write_text(
        json.dumps(
            {
                "replay_id": "redacted-replay-sample",
                "release_root": ANON_ROOT,
                "result": "redacted",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_croissant(out: Path) -> None:
    croissant = {
        "@context": "https://w3id.org/croissant/context/v1",
        "@type": "sc:Dataset",
        "name": "Anonymous E&D Benchmark Artifact Bundle",
        "description": "Sanitized final evidence surfaces, validation outputs, redacted samples, and artifact pointers for anonymous review.",
        "url": "DATASET_URL_PLACEHOLDER",
        "version": "anonymous-submission-v1",
        "license": "Project and third-party licenses; see LICENSES.md",
        "conditionsOfAccess": "Anonymous review access; large archives require separately provided redacted links.",
        "responsibleAI": {
            "redactionStatement": "Private paths, operator identifiers, credential-bearing runtime state, and raw live-session material are excluded or replaced with anonymous tokens.",
            "intendedUse": "Artifact review and reproducibility inspection.",
        },
        "distribution": [
            {"@type": "cr:FileObject", "name": "paper_facing_surface", "contentUrl": "data/global/paper_facing_surface.csv"},
            {"@type": "cr:FileObject", "name": "evidence_gate_report", "contentUrl": "data/global/evidence_gate_report.json"},
            {"@type": "cr:FileObject", "name": "figure_input_inventory", "contentUrl": "data/figure_inputs/figure_input_inventory.csv"},
            {"@type": "cr:FileObject", "name": "redacted_event_trace_sample", "contentUrl": "data/samples/redacted_event_trace_sample.jsonl.gz"},
            {"@type": "cr:FileObject", "name": "large_artifact_pointers", "contentUrl": "metadata/LARGE_ARTIFACTS.md"},
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "name": "paper_facing_surface",
                "field": [
                    {"@type": "cr:Field", "name": "run_id", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "family", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "paper_role", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "evidence_validation_pass", "dataType": "sc:Boolean"},
                ],
            }
        ],
    }
    path = out / "metadata" / "croissant.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(croissant, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "metadata" / "croissant_validation.txt").write_text("croissant_metadata_present=yes\n", encoding="utf-8")


def read_text_for_scan(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                return handle.read(MAX_TEXT_COPY_BYTES)
        except OSError:
            return ""
    if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= MAX_TEXT_COPY_BYTES:
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def scan_forbidden_strings(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    remote_re = re.compile(r"(git@|https://(?:github\.com|gitlab\.com|bitbucket\.org)/)[^\s]+")
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if rel in ALLOWED_SCAN_OUTPUTS:
            continue
        rel_text = rel.as_posix()
        texts = [("path", rel_text)]
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
                                texts.append(("zip_content", handle.read(MAX_TEXT_COPY_BYTES).decode("utf-8", errors="replace")))
            except zipfile.BadZipFile:
                pass
        for location, text in texts:
            for pattern, pattern_id in FORBIDDEN_PATTERNS:
                if pattern in text:
                    hits.append(
                        {
                            "relative_path": rel_text,
                            "location": location,
                            "pattern_id": pattern_id,
                            "excerpt_hash": hashlib.sha256(pattern.encode("utf-8")).hexdigest()[:16],
                        }
                    )
            for match in email_re.findall(text):
                if match != "anonymous@example.com":
                    hits.append(
                        {
                            "relative_path": rel_text,
                            "location": location,
                            "pattern_id": "email",
                            "excerpt_hash": hashlib.sha256(match.encode("utf-8")).hexdigest()[:16],
                        }
                    )
            for match in remote_re.findall(text):
                if "ANON_REMOTE_URL" not in text:
                    hits.append(
                        {
                            "relative_path": rel_text,
                            "location": location,
                            "pattern_id": "git_remote",
                            "excerpt_hash": hashlib.sha256(match.encode("utf-8")).hexdigest()[:16],
                        }
                    )
    return hits


def write_scan_outputs(root: Path, hits: Sequence[dict[str, str]]) -> None:
    scan_csv = root / "validation" / "anonymous_repo_forbidden_string_scan.csv"
    scan_csv.parent.mkdir(parents=True, exist_ok=True)
    with scan_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "location", "pattern_id", "excerpt_hash"])
        writer.writeheader()
        writer.writerows(hits)
    raw_secret_hits = sum(1 for hit in hits if hit["pattern_id"] in {"secret_env", "browser_state", "credential_headers"})
    write_text(
        root / "validation" / "anonymous_repo_scan_report.md",
        f"""# Anonymous Repo Scan Report

forbidden_hits={len(hits)}
raw_secret_hits={raw_secret_hits}
scan_scope=clean_tree_text_and_feasible_compressed_files
scan_status={'pass' if not hits else 'fail'}
""",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path) -> None:
    checksum_path = root / "checksums" / "SHA256SUMS"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if rel == Path("checksums/SHA256SUMS") or ".git" in rel.parts:
            continue
        lines.append(f"{file_sha256(path)}  {rel.as_posix()}")
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def create_supplementary_zip(root: Path) -> int:
    zip_path = root / "supplementary_zip" / "ed-paper-submission.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(root)
            if rel == Path("supplementary_zip/ed-paper-submission.zip") or ".git" in rel.parts:
                continue
            archive.write(path, rel.as_posix())
    return zip_path.stat().st_size


def run_command(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def init_fresh_git_repo(clean_tree: Path, scratch_root: Path, branch_name: str, submission_root: Path) -> tuple[Path, str, Path | None, str, bool]:
    scratch_root.mkdir(parents=True, exist_ok=True)
    repo_path = scratch_root / "ed-paper-submission-fresh-repo"
    if repo_path.exists():
        shutil.rmtree(repo_path)
    shutil.copytree(clean_tree, repo_path, ignore=shutil.ignore_patterns(".git"))

    gitattributes = repo_path / ".gitattributes"
    gitattributes.write_text(
        "*.tar.zst filter=lfs diff=lfs merge=lfs -text\n"
        "*.jsonl.gz filter=lfs diff=lfs merge=lfs -text\n"
        "*.csv.gz filter=lfs diff=lfs merge=lfs -text\n"
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
        "*.zip filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )

    run_command(["git", "init"], repo_path)
    run_command(["git", "config", "user.name", "Anonymous Authors"], repo_path)
    run_command(["git", "config", "user.email", "anonymous@example.com"], repo_path)
    lfs_status = "not_available"
    try:
        run_command(["git", "lfs", "install", "--local"], repo_path)
        lfs_status = "installed"
    except (subprocess.CalledProcessError, FileNotFoundError):
        lfs_status = "not_available"
    run_command(["git", "checkout", "-b", branch_name], repo_path)
    run_command(["git", "add", "."], repo_path)
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Anonymous Authors",
            "GIT_AUTHOR_EMAIL": "anonymous@example.com",
            "GIT_COMMITTER_NAME": "Anonymous Authors",
            "GIT_COMMITTER_EMAIL": "anonymous@example.com",
        }
    )
    subprocess.run(
        ["git", "commit", "-m", "Anonymous NeurIPS E&D artifact submission bundle"],
        cwd=str(repo_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        env=env,
    )
    commit = run_command(["git", "rev-parse", "HEAD"], repo_path).stdout.strip()
    count = run_command(["git", "rev-list", "--count", "HEAD"], repo_path).stdout.strip()
    no_inherited_history = count == "1"
    bundle_path = submission_root / "ed-paper-submission.bundle"
    try:
        run_command(["git", "bundle", "create", str(bundle_path), branch_name], repo_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        bundle_path = None
    return repo_path, commit, bundle_path, lfs_status, no_inherited_history


def required_directories_present(root: Path) -> bool:
    required = [
        "code",
        "data/global",
        "data/figure_inputs",
        "data/claim_support",
        "data/closeouts",
        "data/sanity",
        "data/samples",
        "metadata",
        "validation",
        "checksums",
        "supplementary_zip",
    ]
    return all((root / rel).exists() for rel in required)


def write_branch_report(
    root: Path,
    *,
    branch_name: str,
    clean_tree: Path,
    remote_url: str,
    git_repo_path: Path | None,
    git_commit: str,
    git_bundle_path: Path | None,
    lfs_status: str,
    no_inherited_history: bool,
    forbidden_hits: int,
    raw_secret_hits: int,
    zip_size: int,
    large_artifacts_externalized: bool,
    branch_ready: bool,
    git_creation_status: str,
) -> None:
    total_repo_size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)
    lfs_size = sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".zst", ".gz", ".parquet", ".zip"}
    )
    remote_public = "not_set" if not remote_url else "ANON_REMOTE_URL"
    git_repo_public = "not_created" if git_repo_path is None else str(git_repo_path)
    bundle_public = "not_created" if git_bundle_path is None else sanitize_public_text(str(git_bundle_path))
    report = f"""# E&D Paper Submission Branch Report

clean_repo_path={ANON_ROOT}/submission/ed-paper-submission-clean-tree
scratch_git_repo_path={git_repo_public}
branch_name={branch_name}
remote_url={remote_public}
commit_hash={git_commit or 'not_created'}
git_creation_status={git_creation_status}
lfs_status={lfs_status}
lfs_tracked_patterns=*.tar.zst,*.jsonl.gz,*.csv.gz,*.parquet,*.zip
total_repo_size_bytes={total_repo_size}
total_lfs_size_bytes={lfs_size}
supplementary_zip_size_bytes={zip_size}
croissant_present={'yes' if (root / 'metadata' / 'croissant.json').exists() else 'no'}
claim_map_present={'yes' if (root / 'data' / 'claim_support' / 'final_claim_support_matrix.csv').exists() else 'no'}
forbidden_hits_count={forbidden_hits}
raw_secret_hits_count={raw_secret_hits}
large_artifacts={'externalized' if large_artifacts_externalized else 'included'}
git_bundle_path={bundle_public}
data_url_placeholders=DATASET_URL_PLACEHOLDER
branch_has_no_inherited_history={'yes' if no_inherited_history else 'no'}
SUBMISSION_BRANCH_READY={'yes' if branch_ready else 'no'}
"""
    write_text(root / "validation" / "ed_paper_submission_branch_report.md", report)


def build_submission_bundle(
    *,
    canonical_root: Path,
    submission_root: Path,
    anon_repo_tree: Path,
    source_root: Path,
    scratch_root: Path,
    branch_name: str = DEFAULT_BRANCH,
    force: bool = False,
    enable_git: bool = True,
    git_blocked: bool = False,
    remote_url: str = "",
) -> BundleResult:
    if force:
        if anon_repo_tree.exists():
            shutil.rmtree(anon_repo_tree)
        if submission_root.exists():
            for child in submission_root.iterdir():
                if child.name != anon_repo_tree.name:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
    elif anon_repo_tree.exists():
        raise FileExistsError(f"Output tree already exists: {anon_repo_tree}")

    anon_repo_tree.mkdir(parents=True, exist_ok=True)
    submission_root.mkdir(parents=True, exist_ok=True)
    for rel in [
        "code",
        "data/global",
        "data/figure_inputs",
        "data/claim_support",
        "data/closeouts",
        "data/sanity",
        "data/samples",
        "metadata",
        "validation",
        "checksums",
        "supplementary_zip",
    ]:
        (anon_repo_tree / rel).mkdir(parents=True, exist_ok=True)

    write_public_docs(anon_repo_tree)
    copy_final_data(canonical_root, anon_repo_tree)
    copy_source_code(source_root, anon_repo_tree)
    write_redacted_samples(anon_repo_tree)
    write_croissant(anon_repo_tree)

    zip_size = create_supplementary_zip(anon_repo_tree)
    write_checksums(anon_repo_tree)
    hits = scan_forbidden_strings(anon_repo_tree)
    raw_secret_hits = sum(1 for hit in hits if hit["pattern_id"] in {"secret_env", "browser_state", "credential_headers"})
    write_scan_outputs(anon_repo_tree, hits)

    git_repo_path: Path | None = None
    git_commit = ""
    git_bundle_path: Path | None = None
    lfs_status = "not_run"
    no_inherited_history = False
    git_creation_status = "not_requested"
    if enable_git:
        if git_blocked:
            git_creation_status = "deferred_lustre_git_io_blocker"
        else:
            git_creation_status = "created_in_scratch"
            git_repo_path, git_commit, git_bundle_path, lfs_status, no_inherited_history = init_fresh_git_repo(
                anon_repo_tree, scratch_root, branch_name, submission_root
            )
    branch_ready = (
        forbidden_hits := len(hits)
    ) == 0 and raw_secret_hits == 0 and required_directories_present(anon_repo_tree) and zip_size <= MAX_ZIP_BYTES and bool(git_commit) and no_inherited_history

    write_branch_report(
        anon_repo_tree,
        branch_name=branch_name,
        clean_tree=anon_repo_tree,
        remote_url=remote_url,
        git_repo_path=git_repo_path,
        git_commit=git_commit,
        git_bundle_path=git_bundle_path,
        lfs_status=lfs_status,
        no_inherited_history=no_inherited_history,
        forbidden_hits=forbidden_hits,
        raw_secret_hits=raw_secret_hits,
        zip_size=zip_size,
        large_artifacts_externalized=True,
        branch_ready=branch_ready,
        git_creation_status=git_creation_status,
    )

    # Re-run scan and checksum after the final branch report is materialized.
    hits = scan_forbidden_strings(anon_repo_tree)
    raw_secret_hits = sum(1 for hit in hits if hit["pattern_id"] in {"secret_env", "browser_state", "credential_headers"})
    write_scan_outputs(anon_repo_tree, hits)
    zip_size = create_supplementary_zip(anon_repo_tree)
    write_checksums(anon_repo_tree)
    branch_ready = len(hits) == 0 and raw_secret_hits == 0 and required_directories_present(anon_repo_tree) and zip_size <= MAX_ZIP_BYTES and bool(git_commit) and no_inherited_history
    if branch_ready != (forbidden_hits == 0 and bool(git_commit) and no_inherited_history):
        write_branch_report(
            anon_repo_tree,
            branch_name=branch_name,
            clean_tree=anon_repo_tree,
            remote_url=remote_url,
            git_repo_path=git_repo_path,
            git_commit=git_commit,
            git_bundle_path=git_bundle_path,
            lfs_status=lfs_status,
            no_inherited_history=no_inherited_history,
            forbidden_hits=len(hits),
            raw_secret_hits=raw_secret_hits,
            zip_size=zip_size,
            large_artifacts_externalized=True,
            branch_ready=branch_ready,
            git_creation_status=git_creation_status,
        )
        hits = scan_forbidden_strings(anon_repo_tree)
        write_scan_outputs(anon_repo_tree, hits)
        zip_size = create_supplementary_zip(anon_repo_tree)
        write_checksums(anon_repo_tree)
    return BundleResult(
        clean_tree=anon_repo_tree,
        submission_root=submission_root,
        git_repo_path=git_repo_path,
        git_commit=git_commit,
        git_bundle_path=git_bundle_path,
        branch_ready=branch_ready,
        forbidden_hits=len(hits),
        raw_secret_hits=raw_secret_hits,
        zip_size=zip_size,
        branch_name=branch_name,
        git_creation_status=git_creation_status,
    )


def detect_lustre_git_blocker() -> bool:
    try:
        proc = subprocess.run(
            ["ps", "-o", "stat=,wchan=,cmd=", "-u", os.environ.get("USER", "")],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    for line in proc.stdout.splitlines():
        if "git " in line and "cl_sync_io_wait" in line:
            return True
    return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=Path(os.environ.get("NIPS_CANONICAL_FINAL_ROOT", "")))
    parser.add_argument("--submission-root", type=Path, default=None)
    parser.add_argument("--anon-repo-tree", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--branch-name", default=DEFAULT_BRANCH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--disable-git", action="store_true")
    parser.add_argument("--defer-git-on-blocker", action="store_true")
    parser.add_argument("--remote-url", default=os.environ.get("ANON_REMOTE_URL", ""))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    canonical_root = args.canonical_root
    if not canonical_root:
        raise SystemExit("--canonical-root or NIPS_CANONICAL_FINAL_ROOT is required")
    submission_root = args.submission_root or canonical_root / "submission" / "neurips2026_ed_anonymous_submission_v1"
    anon_repo_tree = args.anon_repo_tree or canonical_root / "submission" / "ed-paper-submission-clean-tree"
    scratch_root = args.scratch_root or Path(tempfile.gettempdir()) / "neurips_ed_anon_branch_and_lfs_release_v1"
    git_blocked = bool(args.defer_git_on_blocker and detect_lustre_git_blocker())
    result = build_submission_bundle(
        canonical_root=canonical_root,
        submission_root=submission_root,
        anon_repo_tree=anon_repo_tree,
        source_root=args.source_root,
        scratch_root=scratch_root,
        branch_name=args.branch_name,
        force=args.force,
        enable_git=not args.disable_git,
        git_blocked=git_blocked,
        remote_url=args.remote_url,
    )
    print(f"clean_tree={result.clean_tree}")
    print(f"submission_root={result.submission_root}")
    print(f"git_repo_path={result.git_repo_path or 'not_created'}")
    print(f"git_commit={result.git_commit or 'not_created'}")
    print(f"git_bundle_path={result.git_bundle_path or 'not_created'}")
    print(f"branch_ready={'yes' if result.branch_ready else 'no'}")
    print(f"forbidden_hits={result.forbidden_hits}")
    print(f"raw_secret_hits={result.raw_secret_hits}")
    print(f"zip_size={result.zip_size}")
    print(f"git_creation_status={result.git_creation_status}")
    return 0 if result.forbidden_hits == 0 and result.raw_secret_hits == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
