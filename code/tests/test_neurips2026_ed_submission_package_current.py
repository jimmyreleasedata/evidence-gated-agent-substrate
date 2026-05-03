from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_minimal_inputs(tmp_path: Path) -> tuple[Path, Path]:
    canonical = tmp_path / "canonical"
    repo = tmp_path / "repo"
    _write(
        canonical / "global" / "paper_facing_surface.csv",
        "paper_row_id,paper_role,family,evidence_validation_pass,main_aggregate_eligible\n"
        "r1,paper_facing,miniwob,true,true\n",
    )
    _write(canonical / "global" / "excluded_rows.csv", "row_id,reason\nx,preflight_only\n")
    _write(canonical / "global" / "evidence_gate_report.json", json.dumps({"admitted_rows": 930, "excluded_rows": 1184}))
    _write(canonical / "global" / "final_claim_support_matrix.csv", "claim,artifact\n930 rows,data/global/paper_facing_surface.csv\n")
    _write(canonical / "global" / "figure_input_inventory.csv", "figure,input\nfig1,data/global/paper_facing_surface.csv\n")
    _write(canonical / "paper_inputs" / "decision_sensitive_table_input.csv", "backend,regime\nvLLM,clean\n")
    _write(canonical / "paper_inputs" / "decision_sensitive_figure4_input.csv", "backend,regime\nvLLM,clean\n")
    _write(canonical / "paper_inputs" / "decision_sensitive_strong_claim_values.json", json.dumps({"supported_claim_level": "STRONG"}))
    _write(canonical / "decision_sensitive_admission" / "decision_slice_summary.md", "56/56 admitted\n")
    _write(canonical / "decision_robustness_closeout_v2_budget9" / "claim_matrix.csv", "claim_level,admitted_rows\nSTRONG,336\n")
    _write(canonical / "decision_robustness_closeout_v2_budget9" / "aggregates" / "cell_metrics.csv", "backend,budget,reversal_cell\nvllm,9,true\n")
    _write(canonical / "decision_robustness_closeout_v2_budget9" / "gate_report.json", json.dumps({"admitted_rows": 336, "excluded_rows": 0}))
    _write(canonical / "decision_robustness_closeout_v2_budget9" / "planned_vs_executed.csv", "planned_cell_id,status\nc1,admitted\n")
    _write(canonical / "decision_robustness_closeout_v2_budget9" / "summary.md", "12/12 comparable cells\n")
    _write(canonical / "rounds" / "targeted_failure_attribution_failure_attribution_audit" / "failure_attribution_audit.md", "closeout private artifact_release_root\n")
    _write(canonical / "rounds" / "swe_verifier_control_repair_swe_gold_control_repair_to_pass" / "swe_verifier_control_repair_claim_support_delta.md", "SWE verifier-control repair pass\n")
    _write(canonical / "rounds" / "swe_only_recompute_after_verifier_control_repair" / "final_swe_claim_support_delta.md", "SWE-only recompute\n")
    _write(canonical / "rounds" / "bounded_stronger_traffic_source_sanity_non_degenerate_agent_sanity" / "claim_support_delta.md", "bounded stronger-traffic-source sanity check\n")
    _write(repo / "scripts" / "final" / "build_final_claim_support_matrix.py", "print('ok')\n")
    _write(repo / "adapters" / "sample_adapter.py", "ROOT='artifact_release_root'\n")
    _write(repo / "tests" / "test_sample.py", "def test_sample():\n    assert True\n")
    _write(repo / "README.md", "Private path ANON_HOME should be sanitized.\n")
    return canonical, repo


def test_current_submission_package_builder_creates_sanitized_outputs(tmp_path: Path) -> None:
    from scripts.final.build_neurips2026_ed_submission_package_current import build_current_submission_package

    canonical, repo = _make_minimal_inputs(tmp_path)
    submission_root = tmp_path / "submission"
    fresh_repo = tmp_path / "fresh_repo"
    result = build_current_submission_package(
        canonical_root=canonical,
        repo_root=repo,
        submission_root=submission_root,
        fresh_repo=fresh_repo,
        final_paper_pdf=None,
        force=True,
    )

    assert (submission_root / "clean_code_tree" / "code" / "scripts" / "final" / "build_final_claim_support_matrix.py").exists()
    assert (submission_root / "hf_dataset_tree" / "data" / "global" / "paper_facing_surface.csv").exists()
    assert (submission_root / "hf_dataset_tree" / "data" / "decision_study" / "budget_grid_study" / "claim_matrix.csv").exists()
    assert (submission_root / "hf_dataset_tree" / "metadata" / "croissant.json").exists()
    assert (submission_root / "neurips_ed_anonymous_supplementary.zip").exists()
    assert result.fresh_repo_commit
    assert result.forbidden_hits == 0
    assert result.raw_secret_hits == 0

    report = (submission_root / "final_submission_readiness_report.md").read_text(encoding="utf-8")
    assert "SUBMISSION_PACKAGE_READY=yes" in report
    assert "SUBMISSION_REMOTE_READY=yes" in report
    assert "SUBMISSION_READY=yes" in report
    assert "FINAL_PAPER_PDF_missing" in report

    scan_text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in [submission_root / "validation" / "forbidden_string_scan.csv", submission_root / "validation" / "raw_secret_scan_report.md"]
    )
    assert "artifact_release_root" not in scan_text
    assert "ANON_USER" not in scan_text
    assert "closeout" not in scan_text
    assert "SWE verifier-control repair" not in scan_text

    count = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=fresh_repo, text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
    author = subprocess.run(["git", "log", "--format=%an <%ae>", "--max-count=1"], cwd=fresh_repo, text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=fresh_repo, text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
    assert count == "1"
    assert branch == "ed-paper-submission"
    assert author == "Anonymous Authors <anonymous@example.com>"
    validate = subprocess.run(
        [sys.executable, "validate_bundle.py"],
        cwd=fresh_repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "smoke validation passed" in validate.stdout

    with zipfile.ZipFile(submission_root / "neurips_ed_anonymous_supplementary.zip") as archive:
        names = set(archive.namelist())
    assert "README_FIRST.md" in names
    assert "metadata/croissant.json" in names
    assert "checksums/SHA256SUMS" in names


def test_remote_release_urls_and_no_pdf_requirement(tmp_path: Path) -> None:
    from scripts.final.build_neurips2026_ed_submission_package_current import build_current_submission_package

    canonical, repo = _make_minimal_inputs(tmp_path)
    submission_root = tmp_path / "remote_submission"
    fresh_repo = tmp_path / "remote_fresh_repo"
    remote_url = "https://github.com/jimmyreleasedata/evidence-gated-agent-substrate"
    dataset_url = "https://huggingface.co/datasets/jimmyreleasedata/evidence-gated-agent-substrate-release"
    result = build_current_submission_package(
        canonical_root=canonical,
        repo_root=repo,
        submission_root=submission_root,
        fresh_repo=fresh_repo,
        final_paper_pdf=None,
        anon_remote_url=remote_url,
        anon_dataset_url=dataset_url,
        require_final_paper_pdf=False,
        force=True,
    )

    assert result.package_ready
    assert (submission_root / "supplementary_zip" / "neurips_ed_anonymous_supplementary.zip").exists()
    completed = submission_root / "hf_dataset_tree" / "metadata" / "croissant_completed_for_openreview.json"
    assert completed.exists()
    croissant = json.loads(completed.read_text(encoding="utf-8"))
    assert croissant["url"] == dataset_url
    assert croissant["codeRepository"] == remote_url.removesuffix(".git")
    report = (submission_root / "validation" / "final_submission_readiness_report.md").read_text(encoding="utf-8")
    assert "FINAL_PAPER_PDF_missing" not in report
    assert "code_url_status=published" in report
    assert "dataset_url_status=published" in report
    assert "SUBMISSION_READY=yes" in report
    zip_reported = next(line for line in report.splitlines() if line.startswith("supplementary_zip="))
    assert "supplementary_zip/neurips_ed_anonymous_supplementary.zip" in zip_reported


def run_direct_current_submission_package_tests() -> None:
    import tempfile

    test_current_submission_package_builder_creates_sanitized_outputs(Path(tempfile.mkdtemp()))
    test_remote_release_urls_and_no_pdf_requirement(Path(tempfile.mkdtemp()))
