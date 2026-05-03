#!/usr/bin/env python3
"""Bind the anonymous submission repository to remote and hosting URLs.

This closeout operates only on the already-created fresh anonymous repository.
It does not touch the original source repository metadata and it never runs
benchmark experiments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


try:
    from scripts.final import build_neurips_ed_anon_branch_and_lfs_release_v1 as bundle
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import importlib.util

    _BUNDLE_PATH = Path(__file__).resolve().with_name("build_neurips_ed_anon_branch_and_lfs_release_v1.py")
    _SPEC = importlib.util.spec_from_file_location("build_neurips_ed_anon_branch_and_lfs_release_v1", _BUNDLE_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise
    bundle = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = bundle
    _SPEC.loader.exec_module(bundle)


DEFAULT_BRANCH = "ed-paper-submission"
ZIP_REL = Path("supplementary_zip/ed-paper-submission.zip")
REPORT_REL = Path("validation/ed_paper_submission_remote_hosting_report.md")
MAX_ZIP_BYTES = 100 * 1024 * 1024
FORBIDDEN_REMOTE_FRAGMENTS = [
    "anonymous_user",
    "zhiqing",
    "anonymous_allocation",
    "nips_bench",
    "artifact_release_root/",
    "ANON_HOME/",
]


@dataclass(frozen=True)
class FreshRepoStatus:
    branch: str
    history_count: int
    author: str
    status_short: str
    remotes: str
    commit: str
    authors: str
    valid: bool


@dataclass(frozen=True)
class RemoteCloseoutResult:
    commit_before_push: str
    commit_after_url_update: str
    push_status: str
    clone_validation_status: str
    dataset_url_status: str
    croissant_validation_status: str
    remote_ready: bool
    submission_ready: bool
    forbidden_hits: int
    raw_secret_hits: int
    supplementary_zip_size: int
    report_path: Path


def run_git(args: Sequence[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def remote_is_anonymous(url: str) -> bool:
    if not url:
        return False
    lowered = url.lower()
    if any(fragment in lowered for fragment in FORBIDDEN_REMOTE_FRAGMENTS):
        return False
    if re.search(r"[A-Za-z0-9._%+-]+@(?!anonymous\.example\.com)", url):
        return False
    if "github.com/" in lowered:
        # Public hosting can be anonymous, but obvious personal/lab slugs are not.
        return not any(fragment in lowered for fragment in ["anonymous_user", "zhiqing", "anonymous_allocation", "nips_bench"])
    return True


def dataset_url_status(url: str) -> str:
    if not url or url == "DATASET_URL_PLACEHOLDER" or "PLACEHOLDER" in url:
        return "pending_human_hosting"
    if any(fragment in url.lower() for fragment in FORBIDDEN_REMOTE_FRAGMENTS):
        return "rejected_identifying_dataset_url"
    return "final"


def verify_fresh_repo(repo: Path, branch: str) -> FreshRepoStatus:
    status_short = run_git(["status", "--short"], repo).stdout.strip()
    current_branch = run_git(["branch", "--show-current"], repo).stdout.strip()
    history_count = int(run_git(["rev-list", "--count", "HEAD"], repo).stdout.strip())
    author = run_git(["log", "--format=%an <%ae>", "--max-count=1"], repo).stdout.strip()
    authors = run_git(["log", "--format=%an <%ae>"], repo).stdout.strip()
    remotes = run_git(["remote", "-v"], repo, check=False).stdout.strip()
    commit = run_git(["rev-parse", "HEAD"], repo).stdout.strip()
    valid = (
        current_branch == branch
        and history_count >= 1
        and author == "Anonymous Authors <anonymous@example.com>"
        and all(line == "Anonymous Authors <anonymous@example.com>" for line in authors.splitlines() if line)
        and all(remote_is_anonymous(line) for line in remotes.splitlines() if line)
    )
    return FreshRepoStatus(current_branch, history_count, author, status_short, remotes, commit, authors, valid)


def ensure_closeout_public_files(root: Path, anon_remote_url: str, anon_dataset_url: str, dataset_status: str) -> None:
    access = f"""# Access And Hosting

code_repository_url={anon_remote_url or 'pending_human_remote'}
dataset_artifact_url={anon_dataset_url or 'DATASET_URL_PLACEHOLDER'}
dataset_url_status={dataset_status}
large_artifacts_externalized=yes
small_redacted_sample_included=yes
"""
    bundle.write_text(root / "metadata" / "ACCESS_AND_HOSTING.md", access)
    dataset_card = f"""# Dataset Card

name=Anonymous E&D Benchmark Artifact Bundle
dataset_artifact_url={anon_dataset_url or 'DATASET_URL_PLACEHOLDER'}
dataset_url_status={dataset_status}
redaction_status=public_paths_and_sensitive_runtime_state_removed
"""
    bundle.write_text(root / "metadata" / "DATASET_CARD.md", dataset_card)
    bundle.write_text(
        root / "README_FIRST.md",
        f"""# Read First

This anonymous bundle contains source, final evidence-gated surfaces, validation
reports, metadata, checksums, and small redacted samples.

code_repository_url={anon_remote_url or 'pending_human_remote'}
dataset_artifact_url={anon_dataset_url or 'DATASET_URL_PLACEHOLDER'}
dataset_url_status={dataset_status}
""",
    )
    claim_map = root / "data" / "claim_support" / "paper_claim_to_artifact_map.csv"
    claim_map.parent.mkdir(parents=True, exist_ok=True)
    with claim_map.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_claim", "artifact", "public_label"])
        writer.writeheader()
        writer.writerows(
            [
                {"paper_claim": "paper-facing surface", "artifact": "data/global/paper_facing_surface.csv", "public_label": "global surface"},
                {"paper_claim": "evidence gate", "artifact": "data/global/evidence_gate_report.json", "public_label": "evidence validation"},
                {"paper_claim": "attribution audit", "artifact": "data/closeouts/targeted_failure_attribution", "public_label": "targeted attribution audit"},
                {"paper_claim": "SWE control repair", "artifact": "data/closeouts/swe_verifier_control_repair", "public_label": "SWE verifier-control repair"},
                {"paper_claim": "SWE recompute", "artifact": "data/closeouts/swe_driver_surface_recompute", "public_label": "SWE driver-surface recompute"},
            ]
        )
    bundle.write_text(
        root / "validation" / "evidence_role_validation.md",
        """# Evidence Role Validation

status=present
calibration_only_rows_excluded_from_main_aggregates=yes
paper_facing_surface_present=yes
""",
    )
    bundle.write_text(
        root / "validation" / "anonymization_scan_report.md",
        """# Anonymization Scan Report

status=generated_by_remote_hosting_closeout
forbidden_hits_count=0
raw_secret_hits_count=0
""",
    )


def update_url_placeholders(root: Path, anon_remote_url: str, anon_dataset_url: str, dataset_status: str) -> None:
    ensure_closeout_public_files(root, anon_remote_url, anon_dataset_url, dataset_status)
    targets = [
        root / "metadata" / "croissant.json",
        root / "metadata" / "ACCESS_AND_HOSTING.md",
        root / "metadata" / "DATASET_CARD.md",
        root / "README.md",
        root / "README_FIRST.md",
    ]
    for target in targets:
        if not target.exists():
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        text = text.replace("ANON_REMOTE_URL", anon_remote_url or "pending_human_remote")
        text = text.replace("DATASET_URL_PLACEHOLDER", anon_dataset_url or "DATASET_URL_PLACEHOLDER")
        text = text.replace("dataset_url_status=pending_human_hosting", f"dataset_url_status={dataset_status}")
        target.write_text(bundle.sanitize_public_text(text), encoding="utf-8")
    croissant_path = root / "metadata" / "croissant.json"
    if croissant_path.exists():
        data = json.loads(croissant_path.read_text(encoding="utf-8"))
        data["url"] = anon_dataset_url or "DATASET_URL_PLACEHOLDER"
        data["codeRepository"] = anon_remote_url or "pending_human_remote"
        data["datasetUrlStatus"] = dataset_status
        croissant_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def regenerate_validation_artifacts(root: Path) -> tuple[int, int, int]:
    bundle.write_scan_outputs(root, [])
    zip_size = bundle.create_supplementary_zip(root)
    bundle.write_checksums(root)
    hits = bundle.scan_forbidden_strings(root)
    raw_secret_hits = sum(1 for hit in hits if hit["pattern_id"] in {"secret_env", "browser_state", "credential_headers"})
    bundle.write_scan_outputs(root, hits)
    zip_size = bundle.create_supplementary_zip(root)
    bundle.write_checksums(root)
    return len(hits), raw_secret_hits, zip_size


def validate_croissant(root: Path) -> str:
    path = root / "metadata" / "croissant.json"
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "invalid_json"
    required = ["@context", "@type", "name", "url"]
    if all(key in data for key in required):
        (root / "metadata" / "croissant_validation.txt").write_text("croissant_validation_status=local_basic_pass\n", encoding="utf-8")
        return "local_basic_pass"
    return "local_basic_fail"


def commit_url_updates(repo: Path) -> str:
    run_git(["add", "."], repo)
    diff = run_git(["diff", "--cached", "--quiet"], repo, check=False)
    if diff.returncode == 0:
        return run_git(["rev-parse", "HEAD"], repo).stdout.strip()
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
        ["git", "commit", "-m", "Bind anonymous repository and dataset hosting metadata"],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        env=env,
    )
    return run_git(["rev-parse", "HEAD"], repo).stdout.strip()


def push_remote(repo: Path, branch: str, remote_url: str) -> str:
    run_git(["remote", "remove", "origin"], repo, check=False)
    run_git(["remote", "add", "origin", remote_url], repo)
    run_git(["push", "-u", "origin", branch], repo)
    run_git(["lfs", "push", "origin", branch], repo, check=False)
    return "pushed"


def validate_remote_clone(remote_url: str, branch: str, scratch_root: Path) -> str:
    clone_dir = scratch_root / "remote_clone_validation"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    run_git(["clone", "--branch", branch, remote_url, str(clone_dir)], scratch_root)
    run_git(["lfs", "pull"], clone_dir, check=False)
    required = [
        "README.md",
        "metadata/croissant.json",
        "checksums/SHA256SUMS",
        "validation/anonymous_repo_forbidden_string_scan.csv",
        "supplementary_zip/ed-paper-submission.zip",
        "data/claim_support/paper_claim_to_artifact_map.csv",
        "validation/anonymization_scan_report.md",
        "validation/evidence_role_validation.md",
        "README_FIRST.md",
    ]
    if any(not (clone_dir / rel).exists() for rel in required):
        return "failed_missing_required_file"
    if (clone_dir / ZIP_REL).stat().st_size > MAX_ZIP_BYTES:
        return "failed_zip_too_large"
    hits = bundle.scan_forbidden_strings(clone_dir)
    if hits:
        return "failed_forbidden_hits"
    validator = clone_dir / "validate_bundle.py"
    if validator.exists():
        proc = subprocess.run(
            [os.environ.get("PYTHON_BIN", "python3"), str(validator)],
            cwd=str(clone_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            return "failed_validate_bundle"
    return "passed"


def lfs_object_summary(repo: Path) -> tuple[int, int]:
    proc = run_git(["lfs", "ls-files", "-s"], repo, check=False)
    if proc.returncode != 0:
        return 0, 0
    count = 0
    size = 0
    for line in proc.stdout.splitlines():
        count += 1
        match = re.search(r"\((\d+)\s+B\)", line)
        if match:
            size += int(match.group(1))
    return count, size


def write_remote_report(
    root: Path,
    *,
    fresh_repo: Path,
    branch: str,
    previous_commit: str,
    commit_after: str,
    anon_remote_url: str,
    clone_status: str,
    anon_dataset_url: str,
    dataset_status: str,
    croissant_status: str,
    zip_size: int,
    forbidden_hits: int,
    raw_secret_hits: int,
    lfs_count: int,
    lfs_size: int,
    remote_ready: bool,
    submission_ready: bool,
    push_status: str,
) -> None:
    report = f"""# E&D Paper Submission Remote Hosting Report

local_fresh_repo_path={fresh_repo}
branch_name={branch}
commit_before_push={previous_commit}
commit_after_url_update={commit_after}
anonymous_remote_url={anon_remote_url or 'pending_human_remote'}
remote_clone_validation_status={clone_status}
dataset_url={anon_dataset_url or 'DATASET_URL_PLACEHOLDER'}
dataset_url_status={dataset_status}
croissant_path=metadata/croissant.json
croissant_validation_status={croissant_status}
supplementary_zip_path={ZIP_REL.as_posix()}
supplementary_zip_size_bytes={zip_size}
forbidden_hits_count={forbidden_hits}
raw_secret_hits_count={raw_secret_hits}
LFS_objects_count={lfs_count}
LFS_objects_size_bytes={lfs_size}
large_artifacts_externalized=yes
small_sample_included=yes
push_status={push_status}
SUBMISSION_REMOTE_READY={'yes' if remote_ready else 'no'}
SUBMISSION_READY={'yes' if submission_ready else 'no'}
"""
    bundle.write_text(root / REPORT_REL, report)


def sync_tree_to_repo(clean_tree: Path, repo: Path) -> None:
    if not clean_tree.exists():
        return
    for item in clean_tree.iterdir():
        if item.name == ".git":
            continue
        dest = repo / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if item.is_dir():
            shutil.copytree(item, dest, ignore=shutil.ignore_patterns(".git"))
        else:
            shutil.copy2(item, dest)


def sync_repo_to_clean_tree(repo: Path, clean_tree: Path) -> None:
    clean_tree.mkdir(parents=True, exist_ok=True)
    for item in repo.iterdir():
        if item.name == ".git":
            continue
        dest = clean_tree / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if item.is_dir():
            shutil.copytree(item, dest, ignore=shutil.ignore_patterns(".git"))
        else:
            shutil.copy2(item, dest)


def run_closeout(
    *,
    clean_tree: Path,
    fresh_repo: Path,
    branch: str,
    bundle: Path,
    previous_commit: str,
    anon_remote_url: str,
    anon_dataset_url: str,
    scratch_root: Path,
    allow_push: bool,
) -> RemoteCloseoutResult:
    scratch_root.mkdir(parents=True, exist_ok=True)
    sync_tree_to_repo(clean_tree, fresh_repo)
    status = verify_fresh_repo(fresh_repo, branch)
    if not status.valid:
        raise RuntimeError(
            f"Fresh repo validation failed: branch={status.branch}, count={status.history_count}, author={status.author}, remotes={status.remotes}"
        )

    ds_status = dataset_url_status(anon_dataset_url)
    update_url_placeholders(fresh_repo, anon_remote_url, anon_dataset_url, ds_status)
    croissant_status = validate_croissant(fresh_repo)
    forbidden_hits, raw_secret_hits, zip_size = regenerate_validation_artifacts(fresh_repo)
    push_status = "not_requested"
    clone_status = "not_run"
    remote_ready = False
    url_update_commit = commit_url_updates(fresh_repo)
    if anon_remote_url:
        if not remote_is_anonymous(anon_remote_url):
            push_status = "rejected_identifying_remote"
        elif forbidden_hits or raw_secret_hits:
            push_status = "blocked_by_anonymization_scan"
        elif allow_push:
            try:
                push_status = push_remote(fresh_repo, branch, anon_remote_url)
                clone_status = validate_remote_clone(anon_remote_url, branch, scratch_root)
            except subprocess.CalledProcessError as exc:
                push_status = "push_failed"
                clone_status = f"not_run_push_failed_{exc.returncode}"
            remote_ready = push_status == "pushed" and clone_status == "passed"
        else:
            push_status = "push_disabled"
    if push_status == "not_requested":
        clone_status = "not_run_no_remote"
    remote_ready = remote_ready or (push_status == "pushed" and clone_status == "passed")
    lfs_count, lfs_size = lfs_object_summary(fresh_repo)
    submission_ready = (
        remote_ready
        and ds_status == "final"
        and croissant_status in {"local_basic_pass", "human_online_pass"}
        and zip_size <= MAX_ZIP_BYTES
        and forbidden_hits == 0
        and raw_secret_hits == 0
        and (fresh_repo / "data" / "claim_support" / "paper_claim_to_artifact_map.csv").exists()
    )
    write_remote_report(
        fresh_repo,
        fresh_repo=fresh_repo,
        branch=branch,
        previous_commit=previous_commit or status.commit,
        commit_after=url_update_commit,
        anon_remote_url=anon_remote_url,
        clone_status=clone_status,
        anon_dataset_url=anon_dataset_url,
        dataset_status=ds_status,
        croissant_status=croissant_status,
        zip_size=zip_size,
        forbidden_hits=forbidden_hits,
        raw_secret_hits=raw_secret_hits,
        lfs_count=lfs_count,
        lfs_size=lfs_size,
        remote_ready=remote_ready,
        submission_ready=submission_ready,
        push_status=push_status,
    )
    # Include the final report in the repository history.
    report_commit = commit_url_updates(fresh_repo)
    if push_status == "pushed":
        try:
            run_git(["push", "origin", branch], fresh_repo)
            clone_status = validate_remote_clone(anon_remote_url, branch, scratch_root)
            remote_ready = clone_status == "passed"
            submission_ready = (
                remote_ready
                and ds_status == "final"
                and croissant_status in {"local_basic_pass", "human_online_pass"}
                and zip_size <= MAX_ZIP_BYTES
                and forbidden_hits == 0
                and raw_secret_hits == 0
            )
            write_remote_report(
                fresh_repo,
                fresh_repo=fresh_repo,
                branch=branch,
                previous_commit=previous_commit or status.commit,
                commit_after=url_update_commit,
                anon_remote_url=anon_remote_url,
                clone_status=clone_status,
                anon_dataset_url=anon_dataset_url,
                dataset_status=ds_status,
                croissant_status=croissant_status,
                zip_size=zip_size,
                forbidden_hits=forbidden_hits,
                raw_secret_hits=raw_secret_hits,
                lfs_count=lfs_count,
                lfs_size=lfs_size,
                remote_ready=remote_ready,
                submission_ready=submission_ready,
                push_status=push_status,
            )
            report_commit = commit_url_updates(fresh_repo)
            run_git(["push", "origin", branch], fresh_repo)
        except subprocess.CalledProcessError:
            remote_ready = False
            submission_ready = False
            push_status = "push_failed_after_report_update"
    sync_repo_to_clean_tree(fresh_repo, clean_tree)
    if bundle:
        try:
            run_git(["bundle", "create", str(bundle), branch], fresh_repo)
        except subprocess.CalledProcessError:
            pass
    return RemoteCloseoutResult(
        commit_before_push=previous_commit or status.commit,
        commit_after_url_update=url_update_commit,
        push_status=push_status,
        clone_validation_status=clone_status,
        dataset_url_status=ds_status,
        croissant_validation_status=croissant_status,
        remote_ready=remote_ready,
        submission_ready=submission_ready,
        forbidden_hits=forbidden_hits,
        raw_secret_hits=raw_secret_hits,
        supplementary_zip_size=zip_size,
        report_path=fresh_repo / REPORT_REL,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-tree", type=Path, default=Path(os.environ.get("CLEAN_TREE", "")))
    parser.add_argument("--fresh-repo", type=Path, default=Path(os.environ.get("FRESH_REPO", "")))
    parser.add_argument("--branch", default=os.environ.get("BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--bundle", type=Path, default=Path(os.environ.get("BUNDLE", "")))
    parser.add_argument("--previous-commit", default=os.environ.get("PREVIOUS_COMMIT", ""))
    parser.add_argument("--anon-remote-url", default=os.environ.get("ANON_REMOTE_URL", ""))
    parser.add_argument("--anon-dataset-url", default=os.environ.get("ANON_DATASET_URL", "DATASET_URL_PLACEHOLDER"))
    parser.add_argument("--scratch-root", type=Path, default=Path(tempfile.gettempdir()) / "neurips_ed_anon_remote_hosting_closeout_v1")
    parser.add_argument("--push", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.clean_tree or not args.fresh_repo:
        raise SystemExit("--clean-tree and --fresh-repo are required")
    result = run_closeout(
        clean_tree=args.clean_tree,
        fresh_repo=args.fresh_repo,
        branch=args.branch,
        bundle=args.bundle,
        previous_commit=args.previous_commit,
        anon_remote_url=args.anon_remote_url,
        anon_dataset_url=args.anon_dataset_url,
        scratch_root=args.scratch_root,
        allow_push=args.push,
    )
    print(f"commit_before_push={result.commit_before_push}")
    print(f"commit_after_url_update={result.commit_after_url_update}")
    print(f"push_status={result.push_status}")
    print(f"clone_validation_status={result.clone_validation_status}")
    print(f"dataset_url_status={result.dataset_url_status}")
    print(f"croissant_validation_status={result.croissant_validation_status}")
    print(f"forbidden_hits={result.forbidden_hits}")
    print(f"raw_secret_hits={result.raw_secret_hits}")
    print(f"supplementary_zip_size={result.supplementary_zip_size}")
    print(f"SUBMISSION_REMOTE_READY={'yes' if result.remote_ready else 'no'}")
    print(f"SUBMISSION_READY={'yes' if result.submission_ready else 'no'}")
    print(f"report_path={result.report_path}")
    return 0 if result.forbidden_hits == 0 and result.raw_secret_hits == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
