"""Optional DuckDB materialization for benchmark-suite traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_duckdb_report(jsonl_path: Path, db_path: Path) -> dict[str, Any]:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "duckdb is required for DuckDB export. Install benchmark-suite[export]."
        ) from exc

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(
            "CREATE OR REPLACE TABLE events AS SELECT * FROM read_json_auto(?)",
            [str(jsonl_path)],
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE summary_by_event_type AS
            SELECT event_type, count(*) AS count
            FROM events
            GROUP BY 1
            ORDER BY 1
            """
        )
        counts = connection.execute(
            """
            SELECT
              count(*) AS event_count,
              count(DISTINCT run_id) AS run_count,
              count(DISTINCT episode_id) AS episode_count,
              count(DISTINCT task_id) AS task_count
            FROM events
            """
        ).fetchone()
    finally:
        connection.close()

    return {
        "db_path": str(db_path),
        "event_count": counts[0] if counts else 0,
        "run_count": counts[1] if counts else 0,
        "episode_count": counts[2] if counts else 0,
        "task_count": counts[3] if counts else 0,
    }
