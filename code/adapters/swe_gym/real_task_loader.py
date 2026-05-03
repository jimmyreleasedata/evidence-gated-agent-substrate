"""Load real SWE slice task metadata."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any

import yaml

from adapters.swe_gym.evidence import SYNTHETIC_COMMIT_PREFIX, SYNTHETIC_PREFIX


@dataclass(slots=True)
class PatchCondition:
    name: str
    kind: str
    file: str | None = None
    content: str | None = None
    patch: str | None = None


@dataclass(slots=True)
class RealSweTaskSpec:
    task_id: str
    instance_id: str
    repo: str
    version: str
    repo_snapshot_path: Path
    base_commit: str
    problem_statement_hash: str
    eval_spec_hash: str
    harness_version: str
    upstream_dataset_version: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    test_patch: str
    build_command: list[str]
    test_command: list[str]
    patch_conditions: list[PatchCondition]
    image_key: str

    @property
    def local_repo_path(self) -> Path:
        return self.repo_snapshot_path

    @property
    def test_patch_hash(self) -> str:
        return _sha256_text(self.test_patch)


def _read_structured(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _coerce_command(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field_name} must be a non-empty list[str]")
    return [str(item) for item in value]


def _normalize_eval_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [text]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _coerce_patch_conditions(payload: Any) -> list[PatchCondition]:
    if not isinstance(payload, dict) or len(payload) < 2:
        raise ValueError("patch_conditions must include at least two named conditions")
    names = set(payload)
    if "oracle" not in names:
        raise ValueError("patch_conditions must include oracle")
    if "noop" not in names and "known_bad" not in names:
        raise ValueError("patch_conditions must include noop or known_bad")

    conditions: list[PatchCondition] = []
    for name, spec in payload.items():
        if not isinstance(spec, dict):
            raise ValueError(f"invalid patch condition entry for {name}: {spec!r}")
        kind = str(spec.get("type") or "").strip()
        if kind not in {"replace_file", "git_apply"}:
            raise ValueError(f"unsupported patch condition type for {name}: {kind!r}")
        conditions.append(
            PatchCondition(
                name=str(name),
                kind=kind,
                file=str(spec.get("file")) if spec.get("file") is not None else None,
                content=str(spec.get("content")) if spec.get("content") is not None else None,
                patch=str(spec.get("patch")) if spec.get("patch") is not None else None,
            )
        )
    return conditions


def _task_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_structured(path)
    if isinstance(payload, dict) and "tasks" in payload:
        tasks = payload["tasks"] or []
        if not isinstance(tasks, list):
            raise ValueError("tasks must be a list")
        return [dict(item) for item in tasks]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise ValueError(f"unsupported SWE real task selection payload in {path}")


def _build_command_for_repo(repo: str) -> list[str]:
    bootstrap_packages = ["pip", "wheel", "pytest"]
    install_target = "."
    if repo == "astropy/astropy":
        bootstrap_packages.extend(["setuptools<69", "setuptools_scm", "extension-helpers"])
        install_target = ".[test]"
    else:
        bootstrap_packages.append("setuptools")
    bootstrap = " ".join(shlex.quote(package) for package in bootstrap_packages)
    return [
        "bash",
        "-lc",
        "python3.11 -m venv .nips-swe-venv && . .nips-swe-venv/bin/activate && "
        f"python -m pip install -U {bootstrap} && "
        f"python -m pip install -e {shlex.quote(install_target)}",
    ]


def _load_official_repo_spec(repo: str, version: str | None) -> dict[str, Any] | None:
    if not version:
        return None
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    except Exception:
        return None
    repo_specs = MAP_REPO_VERSION_TO_SPECS.get(repo)
    if not repo_specs:
        return None
    spec = repo_specs.get(version)
    if not isinstance(spec, dict):
        return None
    resolved = dict(spec)
    if repo == "astropy/astropy":
        pip_packages = [str(pkg) for pkg in resolved.get("pip_packages") or []]
        for extra_pkg in [
            "setuptools_scm",
            "extension-helpers",
            "oldest-supported-numpy",
            "Cython<3",
        ]:
            if extra_pkg not in pip_packages:
                pip_packages.append(extra_pkg)
        resolved["pip_packages"] = pip_packages
    return resolved


def _official_build_command(spec: dict[str, Any]) -> list[str]:
    def _root_only_preinstall(command: str) -> bool:
        root_only_markers = (
            "apt-get",
            "apt ",
            "yum ",
            "dnf ",
            "apk ",
            "locale-gen",
            "update-locale",
            "/etc/",
            "/var/lib/apt",
        )
        return any(marker in command for marker in root_only_markers)

    def _guard_preinstall(command: str) -> str:
        if not _root_only_preinstall(command):
            return command
        quoted = shlex.quote(command)
        return (
            "if [ \"$(id -u)\" = \"0\" ]; then "
            f"{command}; "
            "else "
            f"echo \"[nips-swe] skipping root-only pre_install: {quoted}\"; "
            "fi"
        )

    def _package_install_commands(value: Any) -> list[str]:
        package_entries: list[str] = []
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if text:
                package_entries.extend(token.strip() for token in shlex.split(text) if token.strip())
        elif isinstance(value, (list, tuple)):
            for item in value:
                text = str(item).strip()
                if text:
                    package_entries.append(text)
        commands: list[str] = []
        pip_requirements: list[str] = []
        requirement_files: list[str] = []
        for entry in package_entries:
            if any(token in entry for token in ("&&", "||", ";", "|", "$(", "`")):
                continue
            if entry.endswith((".yml", ".yaml")):
                continue
            if entry.endswith((".txt", ".in")) or "/" in entry:
                requirement_files.append(entry)
            else:
                pip_requirements.append(entry)
        if pip_requirements:
            commands.append("python -m pip install " + " ".join(shlex.quote(pkg) for pkg in pip_requirements))
        for requirement_file in requirement_files:
            commands.append(f"python -m pip install -r {shlex.quote(requirement_file)}")
        return commands

    python_version = str(spec.get("python") or "3.11").strip()
    env_path = "$PWD/.nips-swe-conda"
    commands = [
        "source /opt/conda/etc/profile.d/conda.sh",
        f"conda create -y -p {env_path} python={shlex.quote(python_version)}",
        f"conda install -y -p {env_path} -c conda-forge compilers make pkg-config",
        f"conda activate {env_path}",
    ]
    pip_packages = []
    for pkg in ["pip", "wheel", "pytest", *(str(item) for item in spec.get("pip_packages") or [])]:
        pkg = str(pkg).strip()
        if pkg and pkg not in pip_packages:
            pip_packages.append(pkg)
    if pip_packages:
        commands.append("python -m pip install " + " ".join(shlex.quote(pkg) for pkg in pip_packages))
    commands.extend(_package_install_commands(spec.get("packages")))
    commands.extend(_guard_preinstall(str(cmd).strip()) for cmd in spec.get("pre_install") or [] if str(cmd).strip())
    install_cmd = str(spec.get("install") or "").strip()
    if install_cmd:
        commands.append(install_cmd)
    return ["bash", "-lc", " && ".join(commands)]


def _official_test_command(spec: dict[str, Any], quoted_targets: str) -> list[str]:
    test_cmd = str(spec.get("test_cmd") or "pytest -q").strip()
    if test_cmd.startswith("pytest "):
        test_cmd = test_cmd.replace("pytest ", "python -m pytest ", 1)
    elif test_cmd == "pytest":
        test_cmd = "python -m pytest"
    try:
        from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
    except Exception:
        START_TEST_OUTPUT = "START_TEST_OUTPUT"
        END_TEST_OUTPUT = "END_TEST_OUTPUT"
    return [
        "bash",
        "-lc",
        "source /opt/conda/etc/profile.d/conda.sh"
        " && conda activate \"$PWD/.nips-swe-conda\""
        f" && printf '%s\\n' {shlex.quote(START_TEST_OUTPUT)}"
        f" && PYTHONPATH=. {test_cmd} {quoted_targets}"
        " ; status=$?"
        f" ; printf '%s\\n' {shlex.quote(END_TEST_OUTPUT)}"
        " ; exit $status",
    ]


def build_real_task_row_from_official_instance(
    instance: dict[str, Any],
    *,
    repo_snapshot_path: Path,
    oracle_patch: str,
    noop_patch: str,
    test_targets: list[str],
    image_key: str,
) -> dict[str, Any]:
    quoted_targets = " ".join(shlex.quote(target) for target in test_targets)
    if not quoted_targets:
        raise ValueError("test_targets must be non-empty")
    official_spec = _load_official_repo_spec(str(instance["repo"]), str(instance.get("version") or "").strip() or None)
    build_command = _official_build_command(official_spec) if official_spec else _build_command_for_repo(str(instance["repo"]))
    test_command = _official_test_command(official_spec, quoted_targets) if official_spec else [
        "bash",
        "-lc",
        f". .nips-swe-venv/bin/activate && PYTHONPATH=. pytest -q {quoted_targets}",
    ]

    return {
        "task_id": str(instance.get("task_id") or instance["instance_id"]),
        "instance_id": str(instance["instance_id"]),
        "repo": str(instance["repo"]),
        "version": str(instance.get("version") or ""),
        "repo_snapshot_path": str(repo_snapshot_path),
        "base_commit": str(instance["base_commit"]),
        "problem_statement_hash": str(
            instance.get("problem_statement_hash") or _sha256_text(str(instance.get("problem_statement") or ""))
        ),
        "eval_spec_hash": str(instance.get("eval_spec_hash") or _sha256_text(quoted_targets)),
        "harness_version": str(instance.get("harness_version") or "official_or_compatible"),
        "upstream_dataset_version": f"{instance.get('dataset_name', 'official')}:{instance.get('split', 'test')}",
        "fail_to_pass": _normalize_eval_list(instance.get("FAIL_TO_PASS")),
        "pass_to_pass": _normalize_eval_list(instance.get("PASS_TO_PASS")),
        "test_patch": str(instance.get("test_patch") or ""),
        "build_command": build_command,
        "test_command": test_command,
        "patch_conditions": {
            "oracle": {"type": "git_apply", "patch": oracle_patch},
            "noop": {"type": "git_apply", "patch": noop_patch},
        },
        "image_key": image_key,
    }


def load_real_task_specs(path: Path) -> list[RealSweTaskSpec]:
    rows = _task_rows(path)
    specs: list[RealSweTaskSpec] = []
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        instance_id = str(row.get("instance_id") or "").strip()
        repo = str(row.get("repo") or "").strip()
        base_commit = str(row.get("base_commit") or "").strip()
        if not instance_id or instance_id.startswith(SYNTHETIC_PREFIX):
            raise ValueError(f"invalid official instance_id: {instance_id!r}")
        if not repo:
            raise ValueError(f"missing repo for {instance_id}")
        if not base_commit or base_commit.startswith(SYNTHETIC_COMMIT_PREFIX):
            raise ValueError(f"invalid base_commit for {instance_id}: {base_commit!r}")
        repo_snapshot_path = Path(str(row.get("repo_snapshot_path") or "")).expanduser().resolve(strict=False)
        if not repo_snapshot_path.exists():
            raise FileNotFoundError(f"repo_snapshot_path missing for {instance_id}: {repo_snapshot_path}")

        specs.append(
            RealSweTaskSpec(
                task_id=task_id or instance_id,
                instance_id=instance_id,
                repo=repo,
                version=str(row.get("version") or ""),
                repo_snapshot_path=repo_snapshot_path,
                base_commit=base_commit,
                problem_statement_hash=str(row.get("problem_statement_hash") or _sha256_text(str(row.get("problem_statement") or ""))),
                eval_spec_hash=str(row.get("eval_spec_hash") or _sha256_text(str(row.get("eval_spec") or ""))),
                harness_version=str(row.get("harness_version") or "official_or_compatible"),
                upstream_dataset_version=str(row.get("upstream_dataset_version") or "official_selected_slice"),
                fail_to_pass=_normalize_eval_list(row.get("fail_to_pass")),
                pass_to_pass=_normalize_eval_list(row.get("pass_to_pass")),
                test_patch=str(row.get("test_patch") or ""),
                build_command=_coerce_command(row.get("build_command"), "build_command"),
                test_command=_coerce_command(row.get("test_command"), "test_command"),
                patch_conditions=_coerce_patch_conditions(row.get("patch_conditions")),
                image_key=str(row.get("image_key") or repo),
            )
        )
    return specs
