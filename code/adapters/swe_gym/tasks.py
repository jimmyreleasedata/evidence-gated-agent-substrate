"""Small SWE-Gym slice used for smoke and slice validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SweTaskSpec:
    task_id: str
    instance_id: str
    repo_name: str
    repo_commit: str
    module_name: str
    file_name: str
    buggy_source: str
    patched_source: str
    test_assertion: str


V1_TASKS = [
    SweTaskSpec(
        task_id="fix_answer",
        instance_id="mock__calc-001",
        repo_name="mock_calc",
        repo_commit="deadbeef001",
        module_name="app",
        file_name="app.py",
        buggy_source="def answer():\n    return 41\n",
        patched_source="def answer():\n    return 42\n",
        test_assertion="import app; assert app.answer() == 42",
    ),
    SweTaskSpec(
        task_id="fix_greeting",
        instance_id="mock__hello-002",
        repo_name="mock_hello",
        repo_commit="deadbeef002",
        module_name="app",
        file_name="app.py",
        buggy_source="def greeting():\n    return 'helo world'\n",
        patched_source="def greeting():\n    return 'hello world'\n",
        test_assertion="import app; assert app.greeting() == 'hello world'",
    ),
    SweTaskSpec(
        task_id="fix_flag_parser",
        instance_id="mock__flags-003",
        repo_name="mock_flags",
        repo_commit="deadbeef003",
        module_name="app",
        file_name="app.py",
        buggy_source="def parse_flag(value):\n    return value == 'yes'\n",
        patched_source="def parse_flag(value):\n    return value.lower() in {'yes', 'true', '1'}\n",
        test_assertion="import app; assert app.parse_flag('TRUE') is True",
    ),
    SweTaskSpec(
        task_id="fix_ini_parser",
        instance_id="mock__ini-004",
        repo_name="mock_ini",
        repo_commit="deadbeef004",
        module_name="app",
        file_name="app.py",
        buggy_source=(
            "def parse_ini_flag(text):\n"
            "    config = {}\n"
            "    for raw in text.splitlines():\n"
            "        line = raw.strip()\n"
            "        if not line or line.startswith('#'):\n"
            "            continue\n"
            "        if '=' not in line:\n"
            "            continue\n"
            "        key, value = line.split('=', 1)\n"
            "        config[key.strip()] = value.strip()\n"
            "    return config.get('enabled') == 'yes'\n"
        ),
        patched_source=(
            "def parse_ini_flag(text):\n"
            "    config = {}\n"
            "    for raw in text.splitlines():\n"
            "        line = raw.strip()\n"
            "        if not line or line.startswith('#'):\n"
            "            continue\n"
            "        if '=' not in line:\n"
            "            continue\n"
            "        key, value = line.split('=', 1)\n"
            "        config[key.strip().lower()] = value.strip().lower()\n"
            "    return config.get('enabled') in {'yes', 'true', '1', 'on'}\n"
        ),
        test_assertion="import app; assert app.parse_ini_flag('enabled = ON\\n') is True",
    ),
    SweTaskSpec(
        task_id="fix_path_normalizer",
        instance_id="mock__paths-005",
        repo_name="mock_paths",
        repo_commit="deadbeef005",
        module_name="app",
        file_name="app.py",
        buggy_source=(
            "def normalize_path(path):\n"
            "    return path.replace('\\\\', '/')\n"
        ),
        patched_source=(
            "def normalize_path(path):\n"
            "    normalized = path.replace('\\\\', '/')\n"
            "    parts = [part for part in normalized.split('/') if part not in {'', '.'}]\n"
            "    return '/'.join(parts)\n"
        ),
        test_assertion=(
            "import app; "
            "assert app.normalize_path('.\\\\configs//prod/settings.ini') == 'configs/prod/settings.ini'"
        ),
    ),
    SweTaskSpec(
        task_id="fix_csv_loader",
        instance_id="mock__csv-006",
        repo_name="mock_csv",
        repo_commit="deadbeef006",
        module_name="app",
        file_name="app.py",
        buggy_source=(
            "def load_csv_flag(text):\n"
            "    rows = [line.split(',') for line in text.splitlines() if line.strip()]\n"
            "    return rows[1][1] == 'yes'\n"
        ),
        patched_source=(
            "def load_csv_flag(text):\n"
            "    rows = [line.split(',') for line in text.splitlines() if line.strip()]\n"
            "    if len(rows) < 2 or len(rows[1]) < 2:\n"
            "        return False\n"
            "    return rows[1][1].strip().lower() in {'yes', 'true', '1', 'on'}\n"
        ),
        test_assertion=(
            "import app; "
            "assert app.load_csv_flag('name,enabled\\nservice, TRUE\\n') is True"
        ),
    ),
    SweTaskSpec(
        task_id="fix_json_loader",
        instance_id="mock__json-007",
        repo_name="mock_json",
        repo_commit="deadbeef007",
        module_name="app",
        file_name="app.py",
        buggy_source=(
            "import json\n\n"
            "def load_flag(text):\n"
            "    payload = json.loads(text)\n"
            "    return payload['enabled'] == 'yes'\n"
        ),
        patched_source=(
            "import json\n\n"
            "def load_flag(text):\n"
            "    payload = json.loads(text)\n"
            "    return str(payload.get('enabled', '')).strip().lower() in {'yes', 'true', '1', 'on'}\n"
        ),
        test_assertion=(
            "import app; "
            "assert app.load_flag('{\"enabled\": \"TRUE\"}') is True"
        ),
    ),
    SweTaskSpec(
        task_id="fix_env_parser",
        instance_id="mock__env-008",
        repo_name="mock_env",
        repo_commit="deadbeef008",
        module_name="app",
        file_name="app.py",
        buggy_source=(
            "def parse_env_flag(text):\n"
            "    values = {}\n"
            "    for raw in text.splitlines():\n"
            "        if '=' not in raw:\n"
            "            continue\n"
            "        key, value = raw.split('=', 1)\n"
            "        values[key] = value\n"
            "    return values.get('ENABLED') == 'yes'\n"
        ),
        patched_source=(
            "def parse_env_flag(text):\n"
            "    values = {}\n"
            "    for raw in text.splitlines():\n"
            "        if '=' not in raw:\n"
            "            continue\n"
            "        key, value = raw.split('=', 1)\n"
            "        values[key.strip().upper()] = value.strip().lower()\n"
            "    return values.get('ENABLED') in {'yes', 'true', '1', 'on'}\n"
        ),
        test_assertion=(
            "import app; "
            "assert app.parse_env_flag('ENABLED = ON\\n') is True"
        ),
    ),
]

TASKS_BY_ID = {task.task_id: task for task in V1_TASKS}


def get_task(task_id: str) -> SweTaskSpec:
    try:
        return TASKS_BY_ID[task_id]
    except KeyError as exc:
        raise KeyError(f"unknown SWE task_id: {task_id}") from exc


def default_task_ids() -> list[str]:
    return [task.task_id for task in V1_TASKS]
