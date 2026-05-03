"""Official-compatible SWE harness helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import yaml

from adapters.swe_gym.real_task_loader import PatchCondition, RealSweTaskSpec
from runtime.container_backend.base import ImageRef, stable_sha256


def load_image_manifest(manifest_path: Path) -> dict:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"invalid SWE image manifest: {manifest_path}")
    return payload


def resolve_image_ref(image_manifest: dict, image_key: str, image_root: Path) -> ImageRef:
    images = image_manifest.get("images") or {}
    if image_key not in images:
        raise ValueError(f"missing image key {image_key!r} in SWE image manifest")
    spec = images[image_key]
    image_source = str(spec.get("image_source") or "docker://python:3.11-slim")
    sif_ref = spec.get("sif_path")
    sif_path = str(Path(sif_ref).expanduser().resolve(strict=False)) if sif_ref else None
    image_digest = str(spec.get("image_digest") or f"sha256:{stable_sha256(image_source)[:32]}")
    sif_hash = str(spec.get("sif_hash") or f"sha256:{stable_sha256(sif_path or image_source)[-32:]}")
    return ImageRef(
        image_name=str(spec.get("image_name") or spec.get("name") or image_key),
        image_source=image_source,
        image_digest=image_digest,
        sif_path=sif_path,
        sif_hash=sif_hash,
    )


def apply_patch_condition(repo_root: Path, condition: PatchCondition) -> None:
    if condition.kind == "replace_file":
        if not condition.file:
            raise ValueError(f"replace_file patch condition missing file for {condition.name}")
        target = repo_root / condition.file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(condition.content or "", encoding="utf-8")
        return

    if not condition.patch:
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".diff", delete=False) as handle:
        handle.write(condition.patch)
        patch_path = Path(handle.name)
    try:
        completed = subprocess.run(
            ["git", "apply", "--verbose", str(patch_path)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"failed to apply patch condition {condition.name}: {completed.stderr or completed.stdout}")
    finally:
        patch_path.unlink(missing_ok=True)


def condition_by_name(task: RealSweTaskSpec, name: str) -> PatchCondition:
    for condition in task.patch_conditions:
        if condition.name == name:
            return condition
    raise KeyError(name)
