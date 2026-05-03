"""Backend launch helpers for the appendix-only RL closure slice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import shutil
import shlex
import subprocess
import time
import urllib.error
import urllib.request

import yaml
import re

from runtime.external_paths import DEFAULT_SOSP26_ROOT, discover_runtime_stack_paths

RUNTIME_STACK = discover_runtime_stack_paths()
SOSP26_ROOT = RUNTIME_STACK.sosp26_root or DEFAULT_SOSP26_ROOT
DEFAULT_SGLANG_PY = RUNTIME_STACK.sglang_python or (SOSP26_ROOT / "env" / "conda-ray" / "bin" / "python")
DEFAULT_VLLM_PY = SOSP26_ROOT / "env" / "vllm_qwen35_isolated_clean" / "bin" / "python"
DEFAULT_VLLM_SIF = SOSP26_ROOT / "containers" / "vllm-openai.sif"
DEFAULT_HF_HOME = SOSP26_ROOT / "cache" / "hf"
DEFAULT_HF_HUB_CACHE = DEFAULT_HF_HOME / "hub"
DEFAULT_MODEL_CACHE = Path("artifact_release_root/model_cache")
DEFAULT_EXTERNAL_MODEL_CACHE = Path(os.environ.get("NIPS_EXTERNAL_MODEL_CACHE") or str(DEFAULT_HF_HOME))
DEFAULT_ENV_BIN = SOSP26_ROOT / "env" / "conda-ray" / "bin"
DEFAULT_CUDA_LIB = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib")
DEFAULT_COMPILER_LIB = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/lib64")


@dataclass(slots=True)
class BackendProcess:
    spec: dict
    process: subprocess.Popen[str]
    log_path: Path


def _open_url_no_proxy(url: str, timeout: float = 5.0):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(url, timeout=timeout)


def resolve_dense_qwen_model(manifest_path: Path, model_id: str = "Qwen/Qwen2.5-14B-Instruct") -> dict:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for model in payload.get("models", []):
        if model.get("model_id") == model_id:
            return dict(model)
    raise KeyError(f"dense Qwen model {model_id} not found in manifest")


def served_model_alias(model_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    return f"{normalized}-local"


def _common_thread_env() -> dict[str, str]:
    return {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Keep loopback warmup and readiness probes off the site proxy.
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }


def _uses_remote_hf_model(model: dict) -> bool:
    return str(model.get("model_source") or "") == "huggingface" and not str(model.get("snapshot_path") or model.get("model_path") or "").strip()


def _hf_cache_paths(model: dict) -> tuple[Path, Path]:
    if _uses_remote_hf_model(model):
        hf_home = DEFAULT_EXTERNAL_MODEL_CACHE
        hf_home.mkdir(parents=True, exist_ok=True)
        hf_hub_cache = hf_home / "hub"
        hf_hub_cache.mkdir(parents=True, exist_ok=True)
        return hf_home, hf_hub_cache
    return DEFAULT_HF_HOME, DEFAULT_HF_HUB_CACHE


def _offline_hf_env(model: dict) -> dict[str, str]:
    if _uses_remote_hf_model(model):
        return {
            "TRANSFORMERS_OFFLINE": "0",
            "HF_HUB_OFFLINE": "0",
            "HF_DATASETS_OFFLINE": "0",
        }
    return {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def _proxy_env(model: dict) -> dict[str, str]:
    if _uses_remote_hf_model(model):
        return {}
    return {
        "HTTP_PROXY": "",
        "http_proxy": "",
        "HTTPS_PROXY": "",
        "https_proxy": "",
        "ALL_PROXY": "",
        "all_proxy": "",
    }


def _model_cache_root() -> Path:
    root = Path(os.environ.get("NIPS_MODEL_CACHE") or DEFAULT_MODEL_CACHE)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _runtime_home_root() -> Path:
    root = _model_cache_root() / "runtime" / "home"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _runtime_xdg_root() -> Path:
    root = _model_cache_root() / "runtime" / "xdg"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _torch_home() -> Path:
    root = _model_cache_root() / "torch"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _triton_cache_dir() -> Path:
    root = _model_cache_root() / "triton"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _vllm_cache_root() -> Path:
    root = _model_cache_root() / "vllm"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _torchinductor_cache_dir() -> Path:
    root = _vllm_cache_root() / "torchinductor"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _runtime_isolation_env(*, cache_home: Path, hf_home: Path, hf_hub_cache: Path) -> dict[str, str]:
    home_root = _runtime_home_root()
    xdg_root = _runtime_xdg_root()
    xdg_config_home = xdg_root / "config"
    xdg_state_home = xdg_root / "state"
    xdg_data_home = xdg_root / "data"
    transformers_cache = hf_home / "transformers"
    hf_datasets_cache = hf_home / "datasets"
    torch_home = _torch_home()
    triton_cache_dir = _triton_cache_dir()
    for path in (
        home_root,
        cache_home,
        xdg_config_home,
        xdg_state_home,
        xdg_data_home,
        transformers_cache,
        hf_datasets_cache,
        torch_home,
        triton_cache_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(home_root),
        "XDG_CACHE_HOME": str(cache_home),
        "XDG_CONFIG_HOME": str(xdg_config_home),
        "XDG_STATE_HOME": str(xdg_state_home),
        "XDG_DATA_HOME": str(xdg_data_home),
        "TRANSFORMERS_CACHE": str(transformers_cache),
        "HUGGINGFACE_HUB_CACHE": str(hf_hub_cache),
        "HF_DATASETS_CACHE": str(hf_datasets_cache),
        "TORCH_HOME": str(torch_home),
        "TRITON_CACHE_DIR": str(triton_cache_dir),
    }


def _model_ref(model: dict) -> str:
    return " ".join(
        str(part)
        for part in (
            model.get("model_id", ""),
            model.get("snapshot_path", ""),
            model.get("model_path", ""),
        )
    )


def _is_qwen30b(model: dict) -> bool:
    return "Qwen3-30B-A3B-Instruct-2507" in _model_ref(model)


def _vllm_gpu_memory_utilization(model: dict) -> str | None:
    for key in ("NIPS_VLLM_GPU_MEMORY_UTILIZATION", "VLLM_GPU_MEMORY_UTILIZATION"):
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value
    if _is_qwen30b(model):
        return str(os.environ.get("NIPS_VLLM_GPU_MEMORY_UTILIZATION_QWEN30B") or "0.5")
    return None


def _default_vllm_max_model_len(model: dict) -> str:
    for key in ("NIPS_VLLM_MAX_MODEL_LEN", "MAX_MODEL_LEN"):
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value
    if _is_qwen30b(model):
        return str(os.environ.get("NIPS_VLLM_MAX_MODEL_LEN_QWEN30B") or "65536")
    return "32768"


def _vllm_max_num_seqs(model: dict) -> str | None:
    for key in ("NIPS_VLLM_MAX_NUM_SEQS", "VLLM_MAX_NUM_SEQS"):
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value
    if _is_qwen30b(model):
        return str(os.environ.get("NIPS_VLLM_MAX_NUM_SEQS_QWEN30B") or "64")
    return None


def _vllm_extra_args(model: dict) -> list[str]:
    extra = str(os.environ.get("NIPS_VLLM_EXTRA_ARGS") or os.environ.get("VLLM_EXTRA_ARGS") or "").strip()
    if extra:
        return shlex.split(extra)
    if _is_qwen30b(model):
        return ["--tool-call-parser", "hermes", "--enable-auto-tool-choice"]
    return []


def _vllm_ready_timeout_s(model: dict) -> int:
    if _is_qwen30b(model):
        return int(os.environ.get("NIPS_VLLM_READY_TIMEOUT_QWEN30B") or "600")
    return int(os.environ.get("NIPS_VLLM_READY_TIMEOUT") or "180")


def _backend_cuda_visible_devices(*, backend: str, default: str) -> str:
    backend_key = f"NIPS_{backend.upper()}_CUDA_VISIBLE_DEVICES"
    return str(
        os.environ.get(backend_key)
        or os.environ.get("NIPS_MODEL_CUDA_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or default
    )


def build_vllm_launch_spec(model: dict, repo_root: Path, port: int = 18080, host: str = "127.0.0.1") -> dict:
    model_path = str(model.get("snapshot_path") or model.get("model_path") or model.get("model_path_or_hf_id") or model["model_id"])
    model_alias = served_model_alias(str(model["model_id"]))
    tensor_parallel_size = max(1, int(model.get("tensor_parallel_size") or model.get("required_gpu_count") or 1))
    gpu_memory_utilization = _vllm_gpu_memory_utilization(model)
    max_model_len = _default_vllm_max_model_len(model)
    max_num_seqs = _vllm_max_num_seqs(model)
    if _is_qwen30b(model):
        tensor_parallel_size = max(tensor_parallel_size, 4)
    extra_args = _vllm_extra_args(model)
    model_cache_root = _model_cache_root()
    vllm_cache_root = _vllm_cache_root()
    torchinductor_cache_dir = _torchinductor_cache_dir()
    hf_home, hf_hub_cache = _hf_cache_paths(model)
    runtime_isolation_env = _runtime_isolation_env(
        cache_home=model_cache_root,
        hf_home=hf_home,
        hf_hub_cache=hf_hub_cache,
    )
    apptainer = shutil.which("apptainer") or "apptainer"
    env = {
        "REPO": str(SOSP26_ROOT),
        "PYTHON_BIN": str(DEFAULT_VLLM_PY),
        "VLLM_SIF": str(DEFAULT_VLLM_SIF),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_hub_cache),
        "VLLM_CACHE_ROOT": str(vllm_cache_root),
        "TORCHINDUCTOR_CACHE_DIR": str(torchinductor_cache_dir),
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_DO_NOT_TRACK": "1",
        **runtime_isolation_env,
        **_offline_hf_env(model),
        **_proxy_env(model),
        **_common_thread_env(),
    }
    env["CUDA_VISIBLE_DEVICES"] = _backend_cuda_visible_devices(backend="vllm", default="0")
    inner = (
        "set -euo pipefail\n"
        "PY=$(command -v python3 || command -v python)\n"
        "exec \"$PY\" -u -m vllm.entrypoints.openai.api_server "
        f"--host {host} --port {port} --model {model_path} "
        f"--served-model-name {model_alias} --tensor-parallel-size {tensor_parallel_size} "
        f"--max-model-len {max_model_len} --disable-log-requests"
    )
    if gpu_memory_utilization is not None:
        inner += f" --gpu-memory-utilization {gpu_memory_utilization}"
    if max_num_seqs is not None:
        inner += f" --max-num-seqs {max_num_seqs}"
    if extra_args:
        inner += " " + " ".join(shlex.quote(arg) for arg in extra_args)
    launch_command = [
        apptainer,
        "exec",
        "--nv",
        "--cleanenv",
        "--bind",
        f"{SOSP26_ROOT}:{SOSP26_ROOT}",
        "--bind",
        f"{hf_home}:{hf_home}",
        "--bind",
        f"{model_cache_root}:{model_cache_root}",
        "--env",
        f"HOME={env['HOME']}",
        "--env",
        f"HF_HOME={hf_home}",
        "--env",
        f"HF_HUB_CACHE={hf_hub_cache}",
        "--env",
        f"XDG_CACHE_HOME={env['XDG_CACHE_HOME']}",
        "--env",
        f"XDG_CONFIG_HOME={env['XDG_CONFIG_HOME']}",
        "--env",
        f"XDG_STATE_HOME={env['XDG_STATE_HOME']}",
        "--env",
        f"XDG_DATA_HOME={env['XDG_DATA_HOME']}",
        "--env",
        f"TRANSFORMERS_CACHE={env['TRANSFORMERS_CACHE']}",
        "--env",
        f"HUGGINGFACE_HUB_CACHE={env['HUGGINGFACE_HUB_CACHE']}",
        "--env",
        f"HF_DATASETS_CACHE={env['HF_DATASETS_CACHE']}",
        "--env",
        f"TORCH_HOME={env['TORCH_HOME']}",
        "--env",
        f"TRITON_CACHE_DIR={env['TRITON_CACHE_DIR']}",
        "--env",
        f"VLLM_CACHE_ROOT={vllm_cache_root}",
        "--env",
        f"TORCHINDUCTOR_CACHE_DIR={torchinductor_cache_dir}",
        "--env",
        f"TRANSFORMERS_OFFLINE={env['TRANSFORMERS_OFFLINE']}",
        "--env",
        f"HF_HUB_OFFLINE={env['HF_HUB_OFFLINE']}",
        "--env",
        f"HF_DATASETS_OFFLINE={env['HF_DATASETS_OFFLINE']}",
        "--env",
        "TOKENIZERS_PARALLELISM=false",
        "--env",
        "OPENBLAS_NUM_THREADS=1",
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "MKL_NUM_THREADS=1",
        "--env",
        "NUMEXPR_NUM_THREADS=1",
        str(DEFAULT_VLLM_SIF),
        "bash",
        "-lc",
        inner,
    ]
    if "CUDA_VISIBLE_DEVICES" in env:
        launch_command[launch_command.index("--env"):launch_command.index("--env")] = ["--env", f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}"]
    return {
        "backend": "vllm",
        "backend_impl": "vllm_openai",
        "model_id": str(model["model_id"]),
        "model_path": model_path,
        "served_model_name": model_alias,
        "port": port,
        "api_base_url": f"http://{host}:{port}/v1",
        "ready_url": f"http://{host}:{port}/v1/models",
        "ready_timeout_s": _vllm_ready_timeout_s(model),
        "env": env,
        "launch_command": launch_command,
        # Reuse the same Apptainer/SIF launch path across models to avoid
        # drifting into a second host-side vLLM stack with different wheels.
        "host_fallback_command": [],
        "cwd": str(repo_root),
    }


def build_sglang_launch_spec(model: dict, repo_root: Path, port: int = 30080, host: str = "127.0.0.1") -> dict:
    model_path = str(model.get("snapshot_path") or model.get("model_path") or model.get("model_path_or_hf_id") or model["model_id"])
    model_alias = str(model.get("served_model_name") or served_model_alias(str(model["model_id"])))
    sglang_cache_root = _model_cache_root() / "sglang"
    sglang_cache_root.mkdir(parents=True, exist_ok=True)
    hf_home, hf_hub_cache = _hf_cache_paths(model)
    runtime_isolation_env = _runtime_isolation_env(
        cache_home=sglang_cache_root,
        hf_home=hf_home,
        hf_hub_cache=hf_hub_cache,
    )
    path = f"{DEFAULT_ENV_BIN}:{os.environ.get('PATH', '')}".rstrip(":")
    ld_library_path = f"{DEFAULT_CUDA_LIB}:{DEFAULT_COMPILER_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    library_path = f"{DEFAULT_CUDA_LIB}:{os.environ.get('LIBRARY_PATH', '')}".rstrip(":")
    env = {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_hub_cache),
        "CUDA_VISIBLE_DEVICES": _backend_cuda_visible_devices(backend="sglang", default="1"),
        "PATH": path,
        "LD_LIBRARY_PATH": ld_library_path,
        "LIBRARY_PATH": library_path,
        **runtime_isolation_env,
        **_offline_hf_env(model),
        **_proxy_env(model),
        **_common_thread_env(),
    }
    launch_command = [
        str(DEFAULT_SGLANG_PY),
        "-u",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        model_alias,
        "--disable-cuda-graph",
    ]
    return {
        "backend": "sglang",
        "backend_impl": "sglang_openai",
        "model_id": str(model["model_id"]),
        "model_path": model_path,
        "served_model_name": model_alias,
        "port": port,
        "api_base_url": f"http://{host}:{port}/v1",
        "ready_url": f"http://{host}:{port}/v1/models",
        "ready_timeout_s": 360,
        "env": env,
        "launch_command": launch_command,
        "cwd": str(repo_root),
    }


def wait_for_ready(ready_url: str, timeout_s: int = 240, poll_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with _open_url_no_proxy(ready_url, timeout=5.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"backend did not become ready: {ready_url}")


def _wait_for_ready_with_process(
    process: subprocess.Popen[str],
    ready_url: str,
    *,
    timeout_s: int = 240,
    poll_s: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"backend process exited before ready: {ready_url}")
        try:
            with _open_url_no_proxy(ready_url, timeout=5.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"backend did not become ready: {ready_url}")


def start_backend_process(spec: dict, log_path: Path) -> BackendProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in spec.get("env", {}).items()})
    handle = log_path.open("w", encoding="utf-8")
    command = [str(item) for item in spec["launch_command"]]
    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=spec.get("cwd"),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_ready_with_process(
            process,
            spec["ready_url"],
            timeout_s=int(spec.get("ready_timeout_s", 240)),
        )
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        fallback_command = spec.get("host_fallback_command")
        if spec.get("backend") == "vllm" and fallback_command:
            env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
            process = subprocess.Popen(  # noqa: S603
                [str(item) for item in fallback_command],
                cwd=spec.get("cwd"),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                _wait_for_ready_with_process(
                    process,
                    spec["ready_url"],
                    timeout_s=int(spec.get("ready_timeout_s", 240)),
                )
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise
        else:
            raise
    return BackendProcess(spec=spec, process=process, log_path=log_path)


def stop_backend_process(backend_process: BackendProcess) -> None:
    if backend_process.process.poll() is None:
        backend_process.process.terminate()
        try:
            backend_process.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            backend_process.process.kill()
            backend_process.process.wait(timeout=20)


def fetch_models(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15.0) as response:
        return json.loads(response.read().decode("utf-8"))
