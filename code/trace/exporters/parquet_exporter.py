"""Optional Parquet export for benchmark-suite traces."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

from trace.schema.events import TraceEvent


def export_parquet(path: Path, events: Iterable[TraceEvent | dict[str, Any]]) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required for Parquet export. Install benchmark-suite[export]."
        ) from exc

    rows = []
    for event in events:
        if isinstance(event, TraceEvent):
            rows.append(event.to_dict())
        else:
            rows.append(dict(event))

    if not rows:
        raise ValueError("cannot export an empty event list to Parquet")

    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path
