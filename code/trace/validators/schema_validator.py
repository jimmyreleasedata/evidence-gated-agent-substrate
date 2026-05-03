"""Schema validation for benchmark-suite event traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

from trace.schema.events import TraceEvent


@dataclass(slots=True)
class ValidationIssue:
    line: int
    message: str


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    event_count: int
    issues: list[ValidationIssue] = field(default_factory=list)
    event_type_counts: dict[str, int] = field(default_factory=dict)
    run_ids: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "event_count": self.event_count,
            "issue_count": len(self.issues),
            "issues": [{"line": issue.line, "message": issue.message} for issue in self.issues],
            "event_type_counts": dict(sorted(self.event_type_counts.items())),
            "run_ids": sorted(self.run_ids),
        }

    def raise_for_issues(self) -> None:
        if not self.valid:
            detail = "; ".join(f"line {item.line}: {item.message}" for item in self.issues)
            raise ValueError(detail)


def _validate_rows(rows: list[dict[str, Any]]) -> ValidationResult:
    issues: list[ValidationIssue] = []
    event_type_counts: dict[str, int] = {}
    run_ids: set[str] = set()
    last_step_by_episode: dict[str, int] = {}
    first_event_type: str | None = None
    last_event_type: str | None = None

    for idx, row in enumerate(rows, start=1):
        try:
            event = TraceEvent.model_validate(row)
        except Exception as exc:
            issues.append(ValidationIssue(idx, str(exc)))
            continue

        event_type = str(event.event_type.value)
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        run_ids.add(event.run_id)
        first_event_type = first_event_type or event_type
        last_event_type = event_type

        last_step = last_step_by_episode.get(event.episode_id)
        if last_step is not None and event.step_id < last_step:
            issues.append(
                ValidationIssue(
                    idx,
                    f"step_id regression for episode {event.episode_id}: {event.step_id} < {last_step}",
                )
            )
        last_step_by_episode[event.episode_id] = event.step_id

    if len(run_ids) > 1:
        issues.append(ValidationIssue(0, f"multiple run_ids in one trace: {sorted(run_ids)}"))

    if rows and first_event_type != "run_start":
        issues.append(ValidationIssue(1, f"trace should start with run_start, got {first_event_type}"))
    if rows and last_event_type != "run_end":
        issues.append(
            ValidationIssue(
                len(rows),
                f"trace should end with run_end, got {last_event_type}",
            )
        )

    return ValidationResult(
        valid=not issues,
        event_count=len(rows),
        issues=issues,
        event_type_counts=event_type_counts,
        run_ids=run_ids,
    )


def validate_jsonl(path: Path) -> ValidationResult:
    rows: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                issues.append(ValidationIssue(idx, f"invalid JSON: {exc}"))

    result = _validate_rows(rows)
    result.issues = issues + result.issues
    result.valid = not result.issues
    return result
