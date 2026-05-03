from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from tests.test_decision_robustness_closeout_v1 import controller_row


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_dataset(path: Path, task_ids: list[str]) -> None:
    payload = [
        {"task_id": task_id, "sites": ["shopping"], "start_urls": ["__SHOPPING__"], "eval": []}
        for task_id in task_ids
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v2_builder_freezes_unique_9_task_slice_and_336_planned_rows(tmp_path: Path):
    mod = load_module(
        "scripts/closeout/build_decision_robustness_budget9_v2.py",
        "build_decision_robustness_budget9_v2",
    )
    root = tmp_path / "canonical"
    available = tmp_path / "available.txt"
    dataset = tmp_path / "dataset.json"
    task_ids = ["105", "106", "124", "125", "142", "143", "147", "148", "149", "156", "163"]
    available.write_text("\n".join(task_ids) + "\n", encoding="utf-8")
    write_dataset(dataset, task_ids)

    result = mod.build_budget9_v2(root, available_tasks_path=available, dataset_path=dataset)

    closeout = root / "decision_robustness_closeout_v2_budget9"
    frozen = json.loads((closeout / "manifests" / "frozen_9task_slice.json").read_text(encoding="utf-8"))
    planned = pd.read_csv(closeout / "manifests" / "planned_cells.csv")
    unsupported = pd.read_csv(closeout / "unsupported_cells.csv")
    assert result["frozen_task_count"] == 9
    assert frozen["task_ids"] == ["105", "106", "124", "125", "142", "143", "149", "156", "163"]
    assert len(frozen["task_ids"]) == len(set(frozen["task_ids"]))
    assert len(planned) == 336
    assert set(planned["budget"]) == {5, 7, 9}
    assert planned.groupby("budget").size().to_dict() == {5: 80, 7: 112, 9: 144}
    assert "147" in set(unsupported["task_id"].astype(str))
    assert "148" in set(unsupported["task_id"].astype(str))


def test_v2_builder_rejects_duplicate_or_truncated_budget9_slice(tmp_path: Path):
    mod = load_module(
        "scripts/closeout/build_decision_robustness_budget9_v2.py",
        "build_decision_robustness_budget9_v2_dupes",
    )
    root = tmp_path / "canonical"
    available = tmp_path / "available.txt"
    dataset = tmp_path / "dataset.json"
    available.write_text("105\n106\n124\n124\n142\n143\n149\n156\n163\n", encoding="utf-8")
    write_dataset(dataset, ["105", "106", "124", "142", "143", "149", "156", "163"])

    try:
        mod.build_budget9_v2(root, available_tasks_path=available, dataset_path=dataset)
    except ValueError as exc:
        assert "fewer than 9 unique" in str(exc)
    else:
        raise AssertionError("duplicate task ids must not satisfy budget=9")


def test_v2_gate_and_aggregate_expect_336_rows(tmp_path: Path):
    build = load_module(
        "scripts/closeout/build_decision_robustness_budget9_v2.py",
        "build_decision_robustness_budget9_v2_gate",
    )
    gate_mod = load_module(
        "scripts/closeout/validate_decision_robustness_budget9_v2.py",
        "validate_decision_robustness_budget9_v2",
    )
    agg_mod = load_module(
        "scripts/closeout/aggregate_decision_robustness_budget9_v2.py",
        "aggregate_decision_robustness_budget9_v2",
    )
    root = tmp_path / "canonical"
    available = tmp_path / "available.txt"
    dataset = tmp_path / "dataset.json"
    task_ids = ["105", "106", "124", "125", "142", "143", "149", "156", "163"]
    available.write_text("\n".join(task_ids) + "\n", encoding="utf-8")
    write_dataset(dataset, task_ids)
    build.build_budget9_v2(root, available_tasks_path=available, dataset_path=dataset)
    closeout = root / "decision_robustness_closeout_v2_budget9"

    rows = []
    for budget in (5, 7, 9):
        for task_id in task_ids[:budget]:
            for backend in ("vllm", "sglang"):
                for seed in (0, 1):
                    for regime in ("clean", "medium"):
                        for controller in ("hook_a_only", "hook_b_only"):
                            auc = 0.9 if (regime == "clean" and controller == "hook_a_only") else 0.4
                            if regime == "medium":
                                auc = 0.8 if controller == "hook_b_only" else 0.2
                            rows.append(
                                controller_row(
                                    root,
                                    task_id=task_id,
                                    backend=backend,
                                    regime=regime,
                                    controller=controller,
                                    seed=seed,
                                    budget=budget,
                                    auc=auc,
                                    paper_role="decision_robustness_closeout",
                                )
                            )
    run_dir = closeout / "runs" / "real_complete"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / "controller_trace.csv", index=False)

    gate = gate_mod.validate_budget9_v2(root)
    claim = agg_mod.aggregate_budget9_v2(root)

    assert gate["expected_rows"] == 336
    assert gate["admitted_rows"] == 336
    assert gate["excluded_rows"] == 0
    assert claim["claim_level"] == "STRONG"
    assert (closeout / "planned_vs_executed.csv").exists()
    assert (closeout / "gate_report.json").exists()
    assert (closeout / "claim_matrix.csv").exists()
    assert (closeout / "summary.md").exists()


def run_direct_decision_robustness_budget9_v2_tests() -> None:
    import tempfile

    for test in (
        test_v2_builder_freezes_unique_9_task_slice_and_336_planned_rows,
        test_v2_builder_rejects_duplicate_or_truncated_budget9_slice,
        test_v2_gate_and_aggregate_expect_336_rows,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
