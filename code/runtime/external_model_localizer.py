"""Helpers for pinning external Hugging Face models into the shared local cache tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_cache_dir_name(repo_id: str) -> str:
    owner, name = repo_id.split("/", 1)
    return f"models--{owner}--{name}"


def _repo_root(cache_root: Path) -> Path:
    hub_root = cache_root / "hub"
    if hub_root.exists():
        return hub_root
    return cache_root


def expected_snapshot_path(repo_id: str, *, revision: str, hf_home: Path) -> Path:
    return hf_home / "hub" / repo_cache_dir_name(repo_id) / "snapshots" / revision


def link_staged_repo_into_shared_cache(
    *,
    repo_id: str,
    revision: str,
    shared_hf_home: Path,
    staged_hf_home: Path,
) -> Path:
    shared_repo_dir = shared_hf_home / "hub" / repo_cache_dir_name(repo_id)
    staged_repo_dir = _repo_root(staged_hf_home) / repo_cache_dir_name(repo_id)
    staged_snapshot = staged_repo_dir / "snapshots" / revision
    if not staged_snapshot.exists():
        raise FileNotFoundError(f"staged snapshot missing: {staged_snapshot}")

    shared_repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if shared_repo_dir.exists() or shared_repo_dir.is_symlink():
        if not shared_repo_dir.is_symlink() or shared_repo_dir.resolve(strict=False) != staged_repo_dir.resolve(strict=False):
            raise FileExistsError(f"shared cache path already exists with different target: {shared_repo_dir}")
    else:
        shared_repo_dir.symlink_to(staged_repo_dir)
    return expected_snapshot_path(repo_id, revision=revision, hf_home=shared_hf_home)


def _pin_model_paths(model: dict[str, Any], snapshot_path: Path) -> None:
    snapshot_str = str(snapshot_path)
    model["model_path"] = snapshot_str
    model["snapshot_path"] = snapshot_str
    model["model_path_or_hf_id"] = snapshot_str


def pin_local_snapshot_in_inventory(
    *,
    inventory_path: Path,
    model_id: str,
    snapshot_path: Path,
) -> None:
    payload = yaml.safe_load(inventory_path.read_text(encoding="utf-8")) or {}
    models = list(payload.get("models") or [])
    for model in models:
        if str(model.get("model_id")) == model_id:
            _pin_model_paths(model, snapshot_path)
            break
    else:
        raise KeyError(f"model_id not found in inventory: {model_id}")
    inventory_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def pin_local_snapshot_in_external_candidates(
    *,
    candidates_path: Path,
    model_id: str,
    snapshot_path: Path,
) -> None:
    payload = yaml.safe_load(candidates_path.read_text(encoding="utf-8")) or {}
    models = list(payload.get("models") or payload.get("external_models") or [])
    for model in models:
        if str(model.get("model_id")) == model_id:
            _pin_model_paths(model, snapshot_path)
            break
    else:
        raise KeyError(f"model_id not found in candidates: {model_id}")
    candidates_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
