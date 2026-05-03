"""Probe metadata for external benchmark repositories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


KNOWN_MARKERS: dict[str, tuple[str, ...]] = {
    "browsergym": ("pyproject.toml", "browsergym"),
    "swebench": ("pyproject.toml", "swebench"),
}


@dataclass(slots=True)
class RepoProbe:
    name: str
    root: Path | None
    present: bool
    git_head: str | None
    dirty: bool | None
    markers: list[str]
    error: str | None = None
    head_ref: str | None = None

    @property
    def repo_root(self) -> Path | None:
        return self.root

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "root": None if self.root is None else str(self.root),
            "present": self.present,
            "git_head": self.git_head,
            "dirty": self.dirty,
            "markers": list(self.markers),
            "error": self.error,
            "head_ref": self.head_ref,
        }


def _run_git(repo_root: Path, *args: str) -> tuple[bool, str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, output


def _git_head(repo_root: Path) -> tuple[str | None, str | None, str | None]:
    ok, commit = _run_git(repo_root, "rev-parse", "HEAD")
    if not ok:
        return None, None, commit or "git rev-parse HEAD failed"

    ok, head_ref = _run_git(repo_root, "symbolic-ref", "-q", "--short", "HEAD")
    return commit, head_ref if ok and head_ref else "DETACHED", None


def _git_dirty(repo_root: Path) -> tuple[bool | None, str | None]:
    ok, status = _run_git(repo_root, "status", "--porcelain")
    if not ok:
        return None, status or "git status --porcelain failed"
    return bool(status), None


def _missing_markers(name: str, repo_root: Path) -> list[str]:
    return [marker for marker in KNOWN_MARKERS.get(name, ()) if not (repo_root / marker).exists()]


def probe_repo(name: str, repo_root: Path | None) -> RepoProbe:
    if repo_root is None or not repo_root.exists():
        return RepoProbe(
            name=name,
            root=repo_root,
            present=False,
            git_head=None,
            dirty=None,
            markers=[],
            error=f"{name} repo_root does not exist: {repo_root}",
        )

    repo_root = repo_root.resolve(strict=False)
    missing_markers = _missing_markers(name, repo_root)
    if missing_markers:
        found_markers = [marker for marker in KNOWN_MARKERS.get(name, ()) if (repo_root / marker).exists()]
        return RepoProbe(
            name=name,
            root=repo_root,
            present=False,
            git_head=None,
            dirty=None,
            markers=found_markers,
            error=f"{name} missing required markers: {', '.join(missing_markers)}",
        )

    head, head_ref, git_error = _git_head(repo_root)
    if git_error is not None:
        return RepoProbe(
            name=name,
            root=repo_root,
            present=False,
            git_head=None,
            dirty=None,
            markers=list(KNOWN_MARKERS.get(name, ())),
            error=f"{name} is not a git repository: {git_error}",
        )

    dirty, dirty_error = _git_dirty(repo_root)
    if dirty_error is not None:
        return RepoProbe(
            name=name,
            root=repo_root,
            present=False,
            git_head=head,
            dirty=None,
            markers=list(KNOWN_MARKERS.get(name, ())),
            error=f"{name} dirty-state probe failed: {dirty_error}",
            head_ref=head_ref,
        )

    return RepoProbe(
        name=name,
        root=repo_root,
        present=True,
        git_head=head,
        dirty=dirty,
        markers=list(KNOWN_MARKERS.get(name, ())),
        head_ref=head_ref,
    )
