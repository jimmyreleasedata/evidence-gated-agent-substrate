"""Generic model/backend registry helpers for the final NeurIPS expansion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import os

import yaml
from huggingface_hub import hf_hub_download, model_info
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from runtime.external_paths import DEFAULT_SOSP26_ROOT


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_QWEN_REGISTRY = REPO_ROOT / "manifests" / "models" / "qwen_local_models.yaml"
DEFAULT_LOCAL_AND_HF_MODELS = REPO_ROOT / "manifests" / "models" / "local_and_hf_models.yaml"
DEFAULT_MODEL_BACKEND_PAIRS = REPO_ROOT / "manifests" / "models" / "model_backend_pairs.yaml"
DEFAULT_EXTERNAL_HF_CANDIDATES = REPO_ROOT / "manifests" / "models" / "external_hf_candidates.yaml"
DEFAULT_EXTERNAL_MODEL_CACHE = Path(os.environ.get("NIPS_EXTERNAL_MODEL_CACHE") or str(DEFAULT_SOSP26_ROOT / "cache" / "hf"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _merge_external_config_metadata(row: dict[str, Any], config_path: Path) -> None:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        row["config_probe_error"] = str(exc)
        return

    row["config_hash"] = _sha256_text(json.dumps(payload, sort_keys=True))
    row["model_path_hash"] = row["config_hash"]
    row["architectures"] = list(payload.get("architectures") or [])
    row["model_type"] = str(payload.get("model_type") or "")
    row["max_context_len"] = payload.get("max_position_embeddings")
    row["dtype"] = str(payload.get("torch_dtype") or row.get("dtype") or "unknown")
    quantization_config = payload.get("quantization_config") or {}
    row["quantization"] = str(quantization_config.get("quant_method") or row.get("quantization") or "unknown")
    row["custom_code_required"] = bool(payload.get("auto_map"))


def _merge_external_model_info_metadata(row: dict[str, Any], info: Any) -> None:
    safetensors = getattr(info, "safetensors", None)
    if safetensors is not None:
        total = getattr(safetensors, "total", None)
        params = dict(getattr(safetensors, "parameters", None) or {})
        if total is not None:
            row["parameter_count_total"] = int(total)
        if params:
            row["parameter_count_by_dtype"] = params
            bytes_total = (
                int(params.get("F32", 0)) * 4
                + int(params.get("BF16", 0)) * 2
                + int(params.get("F16", 0)) * 2
                + int(params.get("F8_E4M3", 0))
                + int(params.get("F8_E5M2", 0))
            )
            if bytes_total > 0:
                footprint_gb = round(bytes_total / (1024**3), 3)
                row["weight_footprint_gb_estimate"] = footprint_gb
                row.setdefault("memory_estimate_gb", footprint_gb)
    siblings = list(getattr(info, "siblings", []) or [])
    filenames = [str(getattr(item, "rfilename", "") or "") for item in siblings]
    safetensor_files = [name for name in filenames if name.endswith(".safetensors")]
    index_files = [name for name in filenames if name.endswith(".safetensors.index.json")]
    if safetensor_files:
        row["weight_file_count"] = len(safetensor_files)
    if index_files:
        row["weight_index_files"] = index_files


def default_external_model_candidates() -> list[dict[str, Any]]:
    env_value = str(os.environ.get("NIPS_EXTERNAL_MODEL_CANDIDATES") or "").strip()
    requested = [item.strip() for item in env_value.split(",") if item.strip()]
    default_rows = [
        {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "model_family": "llama",
            "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
            "optional": False,
            "license_access_status": "pending",
            "readiness_status": "pending",
            "size": "8B",
        },
        {
            "model_id": "google/gemma-3-12b-it",
            "model_family": "gemma",
            "hf_id": "google/gemma-3-12b-it",
            "optional": False,
            "license_access_status": "pending",
            "readiness_status": "pending",
            "size": "12B",
        },
        {
            "model_id": "mistralai/Ministral-3-8B-Instruct-2512",
            "model_family": "mistral",
            "hf_id": "mistralai/Ministral-3-8B-Instruct-2512",
            "optional": False,
            "license_access_status": "pending",
            "readiness_status": "pending",
            "size": "8B",
        },
    ]
    if not requested:
        return default_rows
    by_id = {row["model_id"]: row for row in default_rows}
    return [dict(by_id.get(model_id, {"model_id": model_id, "model_family": "unknown", "hf_id": model_id, "optional": True, "license_access_status": "pending", "readiness_status": "pending", "size": "unknown"})) for model_id in requested]


def resolve_external_model_candidates(
    candidates: list[dict[str, Any]],
    *,
    hf_token_available: bool | None = None,
    model_info_fn: Any = model_info,
    download_probe_fn: Any = None,
) -> list[dict[str, Any]]:
    if hf_token_available is None:
        hf_token_available = bool(os.environ.get("REDACTED_SECRET_ENV"))
    if download_probe_fn is None:
        cache_dir = DEFAULT_EXTERNAL_MODEL_CACHE / "hf_probe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        def download_probe_fn(repo_id: str) -> str:
            return str(hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=str(cache_dir)))

    resolved: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        repo_id = str(row.get("hf_id") or row["model_id"])
        row.setdefault("download_probe_status", "pending")
        try:
            info = model_info_fn(repo_id)
            row["model_revision"] = str(getattr(info, "sha", "") or "")
            tags = list(getattr(info, "tags", []) or [])
            row["tags"] = tags
            _merge_external_model_info_metadata(row, info)
            gated = getattr(info, "gated", False)
            private = bool(getattr(info, "private", False))
            if gated and not hf_token_available:
                row["license_access_status"] = "gated_requires_token"
            elif private:
                row["license_access_status"] = "private_repo"
            else:
                row["license_access_status"] = "public_accessible"
            try:
                row["download_probe_path"] = str(download_probe_fn(repo_id))
                row["download_probe_status"] = "ok"
                row["readiness_status"] = "download_probe_ok"
                _merge_external_config_metadata(row, Path(row["download_probe_path"]))
            except GatedRepoError:
                row["download_probe_status"] = "gated_download_denied"
                row["readiness_status"] = "missing_external_model_access"
            except PermissionError:
                row["download_probe_status"] = "gated_download_denied"
                row["readiness_status"] = "missing_external_model_access"
            except Exception as exc:
                row["download_probe_status"] = type(exc).__name__
                row["readiness_status"] = "metadata_reachable"
                row["download_probe_error"] = str(exc)
        except GatedRepoError:
            row["license_access_status"] = "gated_requires_token"
            row["readiness_status"] = "missing_external_model_access"
            row["download_probe_status"] = "gated_metadata_denied"
        except RepositoryNotFoundError:
            row["license_access_status"] = "missing_repo"
            row["readiness_status"] = "missing_external_model_access"
            row["download_probe_status"] = "missing_repo"
        except Exception as exc:
            row["license_access_status"] = "metadata_probe_failed"
            row["readiness_status"] = "metadata_probe_failed"
            row["download_probe_status"] = type(exc).__name__
            row["download_probe_error"] = str(exc)
        resolved.append(row)
    return resolved


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_local_qwen_models(path: Path = DEFAULT_LOCAL_QWEN_REGISTRY) -> list[dict[str, Any]]:
    payload = load_yaml(path)
    return [dict(model) for model in payload.get("models", [])]


def _size_sort_key(model: dict[str, Any]) -> tuple[int, float, str]:
    family = str(model.get("model_family") or "")
    family_rank = {"qwen": 0, "llama": 1, "gemma": 2, "mistral": 3}.get(family, 9)
    size_value = str(model.get("model_size") or model.get("size") or "")
    numeric = 0.0
    for token in size_value.replace("-", " ").split():
        if token.endswith("B"):
            try:
                numeric = float(token[:-1])
                break
            except ValueError:
                continue
    return (family_rank, numeric, str(model.get("model_id") or ""))


def build_local_and_hf_inventory(
    local_models: list[dict[str, Any]],
    external_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model in local_models:
        rows.append(
            {
                "model_id": str(model["model_id"]),
                "model_family": str(model.get("model_family") or "qwen"),
                "model_source": "local_snapshot",
                "model_path": str(model.get("model_path") or model.get("snapshot_path") or ""),
                "snapshot_path": str(model.get("snapshot_path") or model.get("model_path") or ""),
                "model_path_or_hf_id": str(model.get("model_path") or model.get("snapshot_path") or ""),
                "model_revision": str(model.get("checkpoint_id") or ""),
                "tokenizer_hash": str(model.get("tokenizer_hash") or ""),
                "config_hash": str(model.get("model_path_hash") or ""),
                "model_path_hash": str(model.get("model_path_hash") or ""),
                "license_access_status": "local_available",
                "readiness_status": str(model.get("healthcheck_status") or model.get("load_status") or "pending"),
                "size": str(model.get("model_size") or ""),
                "dtype": str(model.get("dtype") or "unknown"),
                "quantization": str(model.get("quantization") or "unknown"),
                "max_context_len": model.get("max_context_len"),
                "memory_estimate_gb": model.get("required_gpu_count"),
                "required_gpu_count": model.get("required_gpu_count"),
                "tensor_parallel_size": model.get("tensor_parallel_size"),
                "served_model_name": str(model.get("served_model_name") or ""),
                "optional": False,
            }
        )
    for candidate in external_candidates:
        rows.append(
            {
                "model_id": str(candidate["model_id"]),
                "model_family": str(candidate.get("model_family") or "unknown"),
                "model_source": "huggingface",
                "model_path_or_hf_id": str(candidate.get("hf_id") or candidate["model_id"]),
                "model_revision": str(candidate.get("model_revision") or ""),
                "tokenizer_hash": str(candidate.get("tokenizer_hash") or ""),
                "config_hash": str(candidate.get("config_hash") or ""),
                "model_path_hash": str(candidate.get("model_path_hash") or ""),
                "license_access_status": str(candidate.get("license_access_status") or "pending"),
                "readiness_status": str(candidate.get("readiness_status") or "pending"),
                "size": str(candidate.get("size") or ""),
                "dtype": str(candidate.get("dtype") or "unknown"),
                "quantization": str(candidate.get("quantization") or "unknown"),
                "max_context_len": candidate.get("max_context_len"),
                "memory_estimate_gb": candidate.get("memory_estimate_gb"),
                "optional": bool(candidate.get("optional")),
            }
        )
    rows = sorted(rows, key=_size_sort_key)
    return {
        "catalog_id": "local_and_hf_models",
        "catalog_version": "0.1.0",
        "models": rows,
        "defaults": {
            "paper_core_model_ids": select_paper_core_models(rows),
        },
    }


def select_paper_core_models(models: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    for model in models:
        if model.get("model_family") == "qwen" and model.get("model_source") == "local_snapshot" and model.get("readiness_status") == "ready":
            selected.append(str(model["model_id"]))
    for model in models:
        if model.get("model_family") in {"llama", "gemma", "mistral"} and model.get("readiness_status") == "ready":
            selected.append(str(model["model_id"]))
    return selected


def _model_is_probe_eligible(model: dict[str, Any]) -> bool:
    readiness = str(model.get("readiness_status") or "pending")
    return readiness in {"ready", "download_probe_ok"}


def build_model_backend_pairs(models: list[dict[str, Any]], *, backends: list[str]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for model in models:
        readiness = str(model.get("readiness_status") or "pending")
        for backend in backends:
            blocked = not _model_is_probe_eligible(model)
            pairs.append(
                {
                    "model_id": str(model["model_id"]),
                    "model_family": str(model.get("model_family") or "unknown"),
                    "backend": backend,
                    "candidate_status": "blocked" if blocked else "pending_healthcheck",
                    "block_reason": readiness if blocked else "",
                }
            )
    return {
        "catalog_id": "model_backend_pairs",
        "catalog_version": "0.1.0",
        "pairs": pairs,
    }
