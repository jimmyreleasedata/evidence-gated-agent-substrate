from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_runbook_covers_real_pilot_scope() -> None:
    runbook = Path("docs/runbooks/webarena_real_controller_pilot.md")
    assert runbook.exists()
    content = runbook.read_text(encoding="utf-8")
    assert "11-task" in content
    assert "hook_a_only" in content
    assert "hook_b_only" in content
    assert "clean" in content
    assert "medium" in content
    assert "single backend first" in content
    assert "not the old mock controller study" in content


def test_controller_study_config_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from adapters.webarena_verified.real_controller_study import ControllerStudyConfig

    dataset_path = tmp_path / "webarena-verified-hard.json"
    available_tasks = tmp_path / "available_task_ids.txt"
    REDACTED_BROWSER_STATE_LABEL = tmp_path / "REDACTED_BROWSER_STATE_LABEL.json"
    extra_headers = tmp_path / "extra_headers.json"
    upstream_root = tmp_path / "upstream-root"
    _write_json(dataset_path, [{"task_id": 124, "sites": ["shopping"], "start_urls": ["__SHOPPING__"], "eval": []}])
    available_tasks.write_text("124\n", encoding="utf-8")
    _write_json(REDACTED_BROWSER_STATE_LABEL, {"REDACTED_BROWSER_STATE_LABEL": []})
    _write_json(extra_headers, {"X-Test": "1"})
    monkeypatch.setenv("NIPS_WEBARENA_VERIFIED_ROOT", str(upstream_root))
    monkeypatch.setenv("NIPS_WEBARENA_VERIFIED_DATASET", str(dataset_path))
    monkeypatch.setenv("NIPS_WEBARENA_VERIFIED_AVAILABLE_TASKS", str(available_tasks))
    monkeypatch.setenv("NIPS_BENCH_WEBARENA_STORAGE_STATE", str(REDACTED_BROWSER_STATE_LABEL))
    monkeypatch.setenv("REDACTED_REQUEST_METADATA_ENV", str(extra_headers))
    monkeypatch.setenv("NIPS_WEBARENA_VERIFIED_COMMIT", "abc123")
    monkeypatch.setenv("NIPS_WEBARENA_VERIFIED_EVALUATOR_VERSION", "official")
    monkeypatch.setenv("WA_SHOPPING", "https://shopping.example")
    monkeypatch.setenv("WA_REDDIT", "https://reddit.example")
    monkeypatch.setenv("WA_GITLAB", "https://gitlab.example")
    monkeypatch.setenv("WA_HOMEPAGE", "https://shopping.example")
    monkeypatch.setenv("PBS_JOBID", "12345.evaluation_environment")

    config = ControllerStudyConfig.from_env(
        output_root=tmp_path / "pilot",
        controllers=("hook_a_only", "hook_b_only"),
        regimes=("clean", "medium"),
        backends=("vllm",),
        seeds=(0, 1, 2),
    )

    assert config.controllers == ("hook_a_only", "hook_b_only")
    assert config.regimes == ("clean", "medium")
    assert config.backends == ("vllm",)
    assert config.seeds == (0, 1, 2)
    assert config.upstream_commit == "abc123"
    assert config.evaluator_version == "official"
    assert config.pbs_job_id == "12345.evaluation_environment"


def test_controller_policies_are_defined_locally() -> None:
    from adapters.webarena_verified.real_controller_study import ControllerPolicy, controller_policy_for_arm

    hook_a = controller_policy_for_arm("hook_a_only")
    hook_b = controller_policy_for_arm("hook_b_only")
    assert isinstance(hook_a, ControllerPolicy)
    assert isinstance(hook_b, ControllerPolicy)
    assert hook_a.arm == "hook_a_only"
    assert hook_b.arm == "hook_b_only"
    assert hook_a.strategy != hook_b.strategy


def test_supported_task_ids_cover_real_live_subset() -> None:
    from adapters.webarena_verified.real_controller_study import SUPPORTED_TASK_IDS

    assert {"105", "106", "124", "125", "142", "143", "149", "156", "163"} <= SUPPORTED_TASK_IDS


def test_price_range_parser_extracts_min_and_max_currency_values() -> None:
    from adapters.webarena_verified.real_controller_study import _price_range_from_product_text

    text = """
    Wireless Earphone Basic
    $19.99
    Wireless Earphone Premium
    $298.00
    Sale
    $0.01
    """

    assert _price_range_from_product_text(text) == {"min": 0.01, "max": 298.0}


def test_pilot_aggregation_emits_reversal_summary(tmp_path: Path) -> None:
    from adapters.webarena_verified.real_controller_study import write_controller_study_outputs

    rows = [
        {
            "backend": "vllm",
            "controller": "hook_a_only",
            "regime": "clean",
            "seed": 0,
            "task_id": "124",
            "reward_auc_over_wallclock": 0.60,
            "queue_wait_p99_ms": 10.0,
            "p99_latency_ms": 110.0,
        },
        {
            "backend": "vllm",
            "controller": "hook_b_only",
            "regime": "clean",
            "seed": 0,
            "task_id": "124",
            "reward_auc_over_wallclock": 0.55,
            "queue_wait_p99_ms": 12.0,
            "p99_latency_ms": 115.0,
        },
        {
            "backend": "vllm",
            "controller": "hook_a_only",
            "regime": "medium",
            "seed": 0,
            "task_id": "124",
            "reward_auc_over_wallclock": 0.40,
            "queue_wait_p99_ms": 75.0,
            "p99_latency_ms": 230.0,
        },
        {
            "backend": "vllm",
            "controller": "hook_b_only",
            "regime": "medium",
            "seed": 0,
            "task_id": "124",
            "reward_auc_over_wallclock": 0.50,
            "queue_wait_p99_ms": 35.0,
            "p99_latency_ms": 180.0,
        },
    ]

    outputs = write_controller_study_outputs(tmp_path, rows)
    summary_path = Path(outputs["decision_reversal_summary"])
    assert summary_path.exists()
    with summary_path.open(encoding="utf-8", newline="") as handle:
        payload = list(csv.DictReader(handle))
    assert payload[0]["clean_ordering"] == "hook_a_only > hook_b_only"
    assert payload[0]["stressed_ordering"] == "hook_b_only > hook_a_only"
    assert payload[0]["reversal"] == "true"

    with Path(outputs["controller_trace"]).open(encoding="utf-8", newline="") as handle:
        trace_fields = set((csv.DictReader(handle).fieldnames or []))
    assert {
        "family",
        "paper_role",
        "decision_label",
        "driver_id",
        "driver_type",
        "driver_version",
        "model_family",
        "model_id",
        "model_backend",
        "backend_engine",
        "policy_version",
        "prompt_template_hash",
        "action_parser_version",
        "budget",
        "implementation_source",
        "manifest_hash",
        "manifest_path",
        "trace_schema_version",
        "suite_version",
        "replay_class",
        "release_root",
        "source_root",
        "evaluator_version",
        "terminal_outcome_present",
        "terminal_outcome",
        "pass_rate",
        "passed",
        "trace_complete",
        "main_aggregate_eligible",
    } <= trace_fields


def test_pbs_pilot_script_uses_real_live_path() -> None:
    path = Path("scripts/evaluation_environment/pbs_webarena_real_controller_pilot.sh")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "#PBS -q debug" in content or "#PBS -q debug-scaling" in content
    assert "#PBS -l select=1:system=evaluation_environment" in content
    assert "webarena_gcp_exports.sh" in content
    assert "run_webarena_real_controller_pilot.sh" in content
    assert "expected_slug" not in content
    assert "backend=mock" not in content


def test_single_cell_live_controller_run_writes_controller_metadata(tmp_path: Path) -> None:
    from adapters.webarena_verified.real_controller_study import (
        ControllerCell,
        ControllerStudyConfig,
        run_controller_cell,
    )
    from adapters.webarena_verified.real_live_capture import BrowserCaptureArtifact

    dataset_path = tmp_path / "webarena-verified-hard.json"
    available_tasks = tmp_path / "available_task_ids.txt"
    REDACTED_BROWSER_STATE_LABEL = tmp_path / "REDACTED_BROWSER_STATE_LABEL.json"
    extra_headers = tmp_path / "extra_headers.json"
    _write_json(
        dataset_path,
        [
            {
                "task_id": 124,
                "sites": ["shopping"],
                "start_urls": ["__SHOPPING__"],
                "intent": "Find the product page.",
                "eval": [],
            }
        ],
    )
    available_tasks.write_text("124\n", encoding="utf-8")
    _write_json(REDACTED_BROWSER_STATE_LABEL, {"REDACTED_BROWSER_STATE_LABEL": []})
    _write_json(extra_headers, {"X-Test": "1"})
    config = ControllerStudyConfig(
        output_root=tmp_path / "pilot",
        dataset_path=dataset_path,
        available_task_ids_path=available_tasks,
        REDACTED_BROWSER_STATE_LABEL_path=REDACTED_BROWSER_STATE_LABEL,
        extra_headers_path=extra_headers,
        upstream_root=tmp_path / "upstream-root",
        upstream_commit="abc123",
        evaluator_version="official",
        wa_urls={
            "shopping": "https://shopping.example",
            "reddit": "https://reddit.example",
            "gitlab": "https://gitlab.example",
            "homepage": "https://shopping.example",
        },
        controllers=("hook_a_only", "hook_b_only"),
        regimes=("clean", "medium"),
        backends=("vllm",),
        seeds=(0, 1, 2),
        pbs_job_id="12345.evaluation_environment",
    )
    cell = ControllerCell(task_id="124", controller="hook_a_only", regime="clean", backend="vllm", seed=0)

    def fake_browser_capture(task: dict, task_root: Path, _config) -> BrowserCaptureArtifact:
        trace_path = task_root / "network.har"
        trace_path.write_text('{"log":{"entries":[]}}', encoding="utf-8")
        screenshot_path = task_root / "screenshots" / "step-000.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(b"png")
        return BrowserCaptureArtifact(
            start_url="https://shopping.example",
            final_url="https://shopping.example/product",
            title="One Stop Market",
            browser_version="Chromium 125.0.6422.26",
            driver_version="playwright-1.44.0",
            action_log=[{"step": 1, "action": "goto", "target": "https://shopping.example", "status": "ok"}],
            screenshot_paths=[screenshot_path],
            network_trace_path=trace_path,
            browser_trace_path=None,
            gcp_public_host="shopping.example",
        )

    def fake_agent_response(task: dict, _task_root: Path, _config, current_cell: ControllerCell) -> dict:
        assert current_cell == cell
        return {
            "task_type": "RETRIEVE",
            "status": "SUCCESS",
            "retrieved_data": ["demo"],
            "error_details": None,
        }

    def fake_evaluator(task: dict, task_root: Path, _config) -> dict:
        payload = json.loads((task_root / "agent_response.json").read_text(encoding="utf-8"))
        assert sorted(payload.keys()) == ["retrieved_data", "status", "task_type"]
        assert payload["status"] == "SUCCESS"
        return {
            "task_id": task["task_id"],
            "passed": True,
            "score": 1.0,
            "status": "SUCCESS",
            "evaluator_version": "official",
            "evaluator_id": "webarena_verified_real_evaluator",
        }

    summary = run_controller_cell(
        config,
        cell,
        browser_capture_fn=fake_browser_capture,
        agent_response_fn=fake_agent_response,
        evaluator_fn=fake_evaluator,
    )

    assert summary["implementation_source"] == "real_upstream_live"
    assert summary["controller"] == "hook_a_only"
    assert summary["regime"] == "clean"
    assert summary["backend"] == "vllm"
    assert summary["seed"] == 0
    assert summary["stress_overlay"] is False
    assert summary["task_success"] is True
    assert summary["evidence_validation_pass"] is True
    assert "hook_a_only" in summary["summary_path"]
    assert "clean" in summary["summary_path"]
    assert "seed_0" in summary["summary_path"]
    assert summary["family"] == "webarena_verified"
    assert summary["paper_role"] == "paper_facing_decision_evidence"
    assert summary["main_aggregate_eligible"] is False
    assert summary["decision_label"] == "hook_a_only"
    assert summary["driver_id"] == "webarena_real_controller:hook_a_only:clean:vllm"
    assert summary["driver_type"] == "controller"
    assert summary["driver_version"] == "real_controller_study_v1"
    assert summary["model_family"] == "controller_policy"
    assert summary["model_id"] == "webarena_real_controller_vllm"
    assert summary["model_backend"] == "vllm"
    assert summary["backend_engine"] == "vllm"
    assert summary["budget"] == 0 or summary["budget"]
    assert summary["prompt_template_hash"].startswith("sha256:")
    assert summary["action_parser_version"] == "direct_controller_action_v1"
    assert summary["manifest_hash"].startswith("sha256:")
    assert summary["trace_schema_version"] == "1.0.0"
    assert summary["suite_version"] == "0.1.0"
    assert summary["replay_class"] == "R1"
    assert summary["release_root"] == str(config.output_root)
    assert summary["source_root"] == str(config.output_root)
    assert summary["terminal_outcome_present"] is True
    assert summary["terminal_outcome"] == "pass"
    assert summary["pass_rate"] == 1.0
    assert summary["passed"] is True
    assert summary["trace_complete"] is True


def test_controller_agent_response_omits_experiment_metadata() -> None:
    from adapters.webarena_verified.real_controller_study import (
        ControllerCell,
        _agent_response_from_solver_payload,
    )

    payload = _agent_response_from_solver_payload(
        {
            "task_type": "RETRIEVE",
            "status": "SUCCESS",
            "retrieved_data": ["demo"],
            "error_details": None,
        },
        ControllerCell(task_id="163", controller="hook_a_only", regime="clean", backend="vllm", seed=0),
    )

    assert payload == {
        "task_type": "RETRIEVE",
        "status": "SUCCESS",
        "retrieved_data": ["demo"],
    }


def test_default_controller_agent_response_uses_solver_result(tmp_path: Path) -> None:
    from adapters.webarena_verified.real_controller_study import (
        ControllerCell,
        ControllerStudyConfig,
        run_controller_cell,
    )
    from adapters.webarena_verified.real_live_capture import BrowserCaptureArtifact

    dataset_path = tmp_path / "webarena-verified-hard.json"
    available_tasks = tmp_path / "available_task_ids.txt"
    REDACTED_BROWSER_STATE_LABEL = tmp_path / "REDACTED_BROWSER_STATE_LABEL.json"
    extra_headers = tmp_path / "extra_headers.json"
    _write_json(
        dataset_path,
        [
            {
                "task_id": 163,
                "sites": ["shopping"],
                "start_urls": ["__SHOPPING__"],
                "intent": "Find a review title.",
                "eval": [],
            }
        ],
    )
    available_tasks.write_text("163\n", encoding="utf-8")
    _write_json(REDACTED_BROWSER_STATE_LABEL, {"REDACTED_BROWSER_STATE_LABEL": []})
    _write_json(extra_headers, {"X-Test": "1"})
    config = ControllerStudyConfig(
        output_root=tmp_path / "pilot",
        dataset_path=dataset_path,
        available_task_ids_path=available_tasks,
        REDACTED_BROWSER_STATE_LABEL_path=REDACTED_BROWSER_STATE_LABEL,
        extra_headers_path=extra_headers,
        upstream_root=tmp_path / "upstream-root",
        upstream_commit="abc123",
        evaluator_version="official",
        wa_urls={
            "shopping": "https://shopping.example",
            "reddit": "https://reddit.example",
            "gitlab": "https://gitlab.example",
            "homepage": "https://shopping.example",
        },
        controllers=("hook_a_only",),
        regimes=("clean",),
        backends=("vllm",),
        seeds=(0,),
    )
    cell = ControllerCell(task_id="163", controller="hook_a_only", regime="clean", backend="vllm", seed=0)

    def fake_browser_capture(task: dict, task_root: Path, _config) -> BrowserCaptureArtifact:
        trace_path = task_root / "network.har"
        trace_path.write_text('{"log":{"entries":[]}}', encoding="utf-8")
        screenshot_path = task_root / "screenshots" / "step-000.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(b"png")
        _write_json(
            task_root / "solver_result.json",
            {
                "task_type": "RETRIEVE",
                "status": "SUCCESS",
                "retrieved_data": ["Memory Card Came Defective"],
                "error_details": None,
            },
        )
        return BrowserCaptureArtifact(
            start_url="https://shopping.example",
            final_url="https://shopping.example/product",
            title="One Stop Market",
            browser_version="Chromium 125.0.6422.26",
            driver_version="playwright-1.44.0",
            action_log=[{"step": 1, "action": "goto", "target": "https://shopping.example", "status": "ok"}],
            screenshot_paths=[screenshot_path],
            network_trace_path=trace_path,
            browser_trace_path=None,
            gcp_public_host="shopping.example",
        )

    def fake_evaluator(_task: dict, task_root: Path, _config) -> dict:
        payload = json.loads((task_root / "agent_response.json").read_text(encoding="utf-8"))
        assert payload["status"] == "SUCCESS"
        assert payload["retrieved_data"] == ["Memory Card Came Defective"]
        return {
            "passed": True,
            "score": 1.0,
            "status": "SUCCESS",
            "evaluator_version": "official",
            "evaluator_id": "webarena_verified_real_evaluator",
        }

    summary = run_controller_cell(
        config,
        cell,
        browser_capture_fn=fake_browser_capture,
        evaluator_fn=fake_evaluator,
    )

    assert summary["task_success"] is True
    assert summary["terminal_outcome"] == "pass"
