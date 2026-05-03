"""Repository materialization helpers for the SWE-Gym mock slice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

from adapters.swe_gym.tasks import SweTaskSpec


@dataclass(slots=True)
class RepoLayout:
    seed_root: Path
    runtime_root: Path
    file_path: Path
    dependency_lock: Path


def create_seed_repo(task: SweTaskSpec, seed_root: Path) -> RepoLayout:
    seed_root.mkdir(parents=True, exist_ok=True)
    file_path = seed_root / task.file_name
    file_path.write_text(task.buggy_source, encoding="utf-8")
    dependency_lock = seed_root / "requirements.lock"
    dependency_lock.write_text(f"python=={sys.version.split()[0]}\n", encoding="utf-8")
    (seed_root / "README.md").write_text(
        f"# {task.repo_name}\n\nSynthetic repo for {task.instance_id}\n",
        encoding="utf-8",
    )
    return RepoLayout(
        seed_root=seed_root,
        runtime_root=seed_root,
        file_path=file_path,
        dependency_lock=dependency_lock,
    )


def checkout_repo(seed_root: Path, runtime_root: Path) -> RepoLayout:
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    shutil.copytree(seed_root, runtime_root)
    return RepoLayout(
        seed_root=seed_root,
        runtime_root=runtime_root,
        file_path=runtime_root / "app.py",
        dependency_lock=runtime_root / "requirements.lock",
    )


def apply_patch(task: SweTaskSpec, runtime_root: Path) -> Path:
    target = runtime_root / task.file_name
    target.write_text(task.patched_source, encoding="utf-8")
    return target
