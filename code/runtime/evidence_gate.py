"""Strict paper-facing evidence gate for benchmark families."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any

from adapters.miniwob.evidence import infer_implementation_source as miniwob_impl_source
from adapters.swe_gym.evidence import infer_implementation_source as swe_impl_source
from adapters.webarena_verified.evidence import infer_implementation_source as webarena_impl_source


ALLOWED_PAPER_FACING = {"real_upstream", "real_upstream_replay", "real_upstream_live"}


@dataclass(frozen=True, slots=True)
class EvidenceGateResult:
    family: str
    implementation_source: str
    allowed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class EvidenceGateReport:
    csv_path: Path
    allowed: bool
    checked_rows: int
    failures: list[dict[str, Any]]


def _normalized_family(family: str) -> str:
    if family == "webarena":
        return "webarena_verified"
    if family == "swe":
        return "swe_gym"
    return family


def infer_implementation_source(family: str, row: dict[str, Any]) -> str:
    family = _normalized_family(family)
    if family == "miniwob":
        return miniwob_impl_source(row, row)
    if family == "webarena_verified":
        return webarena_impl_source(row, row)
    if family == "swe_gym":
        return swe_impl_source(row, row)
    return str(row.get("implementation_source") or "unsupported")


def _missing_fields(row: dict[str, Any], fields: list[str]) -> list[str]:
    return [field for field in fields if not row.get(field)]


def evaluate_paper_facing_row(family: str, row: dict[str, Any]) -> EvidenceGateResult:
    family = _normalized_family(family)
    if family not in {"miniwob", "webarena_verified", "swe_gym"}:
        implementation_source = str(row.get("implementation_source") or "unscoped")
        return EvidenceGateResult(family, implementation_source, True, "family not governed by benchmark evidence gate")
    implementation_source = infer_implementation_source(family, row)

    if implementation_source == "mock_fixture":
        return EvidenceGateResult(family, implementation_source, False, "mock_fixture evidence is never paper-facing")
    if implementation_source == "synthetic_executable":
        return EvidenceGateResult(
            family,
            implementation_source,
            False,
            "synthetic_executable evidence is appendix-only and cannot enter paper-facing reports",
        )
    if implementation_source not in ALLOWED_PAPER_FACING:
        return EvidenceGateResult(family, implementation_source, False, f"unsupported implementation_source={implementation_source}")

    if family == "webarena_verified":
        replay_class = row.get("replay_class")
        if replay_class == "R1" or implementation_source == "real_upstream_replay":
            missing = _missing_fields(row, ["trace_hash", "evaluator_version"])
            if missing:
                return EvidenceGateResult(family, implementation_source, False, f"missing required WebArena replay fields: {', '.join(missing)}")
        else:
            missing = _missing_fields(row, ["evaluator_version", "task_id"])
            if missing:
                return EvidenceGateResult(family, implementation_source, False, f"missing required WebArena live fields: {', '.join(missing)}")
    elif family == "swe_gym":
        missing = _missing_fields(row, ["replay_class", "instance_id", "repo", "base_commit", "harness_version"])
        if missing:
            return EvidenceGateResult(family, implementation_source, False, f"missing required SWE fields: {', '.join(missing)}")
        if row.get("replay_class") != "R2":
            return EvidenceGateResult(family, implementation_source, False, "paper-facing SWE rows require replay_class=R2")
        image_ref = row.get("image_digest_or_sif_hash") or row.get("image_digest") or row.get("sif_hash")
        if not image_ref:
            return EvidenceGateResult(family, implementation_source, False, "missing required SWE image/SIF hash")
    elif family == "miniwob":
        missing = _missing_fields(row, ["replay_class", "task_id", "upstream_package_version", "task_manifest_hash"])
        if missing:
            return EvidenceGateResult(family, implementation_source, False, f"missing required MiniWoB fields: {', '.join(missing)}")
        if row.get("replay_class") != "R0":
            return EvidenceGateResult(family, implementation_source, False, "paper-facing MiniWoB rows require replay_class=R0")
        browser_ref = row.get("browser_version") or row.get("driver_version") or row.get("backend_version")
        if not browser_ref:
            return EvidenceGateResult(family, implementation_source, False, "missing required MiniWoB browser/backend version")

    return EvidenceGateResult(family, implementation_source, True, "allowed")


def validate_paper_facing_csv(csv_path: Path, family_column: str = "family") -> EvidenceGateReport:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if family_column not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing required family column: {family_column}")
        failures: list[dict[str, Any]] = []
        checked_rows = 0
        for index, row in enumerate(reader, start=2):
            checked_rows += 1
            family = str(row.get(family_column) or "")
            result = evaluate_paper_facing_row(family, row)
            if not result.allowed:
                failures.append(
                    {
                        "line": index,
                        "family": result.family,
                        "implementation_source": result.implementation_source,
                        "reason": result.reason,
                    }
                )

    return EvidenceGateReport(
        csv_path=csv_path,
        allowed=not failures,
        checked_rows=checked_rows,
        failures=failures,
    )
