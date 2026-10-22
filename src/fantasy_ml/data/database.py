"""Local DuckDB storage for validated NFL fantasy datasets."""

from __future__ import annotations

from pathlib import Path

import duckdb


DEFAULT_SOURCES = {
    "player_stats": Path("data/processed/player_stats_2018_2025.parquet"),
    "training_player_games": Path(
        "data/processed/training_player_games_2018_2025.parquet"
    ),
    "player_game_availability": Path(
        "data/interim/player_game_availability_2018_2025.parquet"
    ),
}


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return con.execute(
        """
        SELECT COUNT(*) > 0
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [table],
    ).fetchone()[0]


def _sql_path(path: Path) -> str:
    """Return a safely quoted absolute path for a DuckDB SQL literal."""
    return str(path.resolve()).replace("'", "''")


def initialize_database(
    db_path: str | Path = "data/fantasy.duckdb",
    sources: dict[str, str | Path] | None = None,
) -> dict[str, dict]:
    """Create/update dataset tables from local Parquet files.

    Unchanged files are skipped. Changed files are imported transactionally
    and replace their corresponding table. Source Parquet files are untouched.
    """
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    configured_sources = sources or DEFAULT_SOURCES

    con = duckdb.connect(str(database_path))
    report: dict[str, dict] = {}

    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS source_registry (
                table_name VARCHAR PRIMARY KEY,
                source_path VARCHAR NOT NULL,
                file_size_bytes BIGINT NOT NULL,
                file_mtime_ns BIGINT NOT NULL,
                row_count BIGINT NOT NULL,
                imported_at TIMESTAMP NOT NULL
            )
        """)

        for table, configured_path in configured_sources.items():
            # Table names come from the fixed configuration, not user input.
            if not table.replace("_", "").isalnum():
                raise ValueError(f"Invalid table name: {table!r}")

            source_path = Path(configured_path).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Parquet source does not exist: {source_path}"
                )

            stat = source_path.stat()
            previous = con.execute(
                """
                SELECT source_path, file_size_bytes, file_mtime_ns, row_count
                FROM source_registry
                WHERE table_name = ?
                """,
                [table],
            ).fetchone()

            unchanged = (
                previous is not None
                and previous[0] == str(source_path)
                and previous[1] == stat.st_size
                and previous[2] == stat.st_mtime_ns
                and _table_exists(con, table)
            )

            if unchanged:
                report[table] = {
                    "status": "skipped",
                    "rows": previous[3],
                    "source": str(source_path),
                }
                continue

            path_literal = _sql_path(source_path)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    f'CREATE OR REPLACE TABLE "{table}" AS '
                    f"SELECT * FROM read_parquet('{path_literal}')"
                )
                row_count = con.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]

                con.execute(
                    """
                    INSERT INTO source_registry
                        (table_name, source_path, file_size_bytes,
                         file_mtime_ns, row_count, imported_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT (table_name) DO UPDATE SET
                        source_path = excluded.source_path,
                        file_size_bytes = excluded.file_size_bytes,
                        file_mtime_ns = excluded.file_mtime_ns,
                        row_count = excluded.row_count,
                        imported_at = excluded.imported_at
                    """,
                    [
                        table,
                        str(source_path),
                        stat.st_size,
                        stat.st_mtime_ns,
                        row_count,
                    ],
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

            report[table] = {
                "status": "imported",
                "rows": row_count,
                "source": str(source_path),
            }

        return report
    finally:
        con.close()


def get_database_summary(
    db_path: str | Path = "data/fantasy.duckdb",
) -> dict[str, dict]:
    """Return row counts and source metadata for imported datasets."""
    database_path = Path(db_path)
    if not database_path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database_path}")

    con = duckdb.connect(str(database_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT table_name, source_path, row_count, imported_at
            FROM source_registry
            ORDER BY table_name
        """).fetchall()

        return {
            table: {
                "source": source,
                "rows": count,
                "imported_at": imported_at.isoformat(),
            }
            for table, source, count, imported_at in rows
        }
    finally:
        con.close()
