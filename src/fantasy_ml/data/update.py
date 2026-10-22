"""Incrementally update the local NFL fantasy data store."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import nflreadpy
import polars as pl

from fantasy_ml.data.training import build_training_dataset

CURRENT_SEASON = datetime.now(UTC).year

SOURCE_TABLES = {
    "player_stats": "player_stats",
    "weekly_rosters": "weekly_rosters",
    "injury_reports": "injury_reports",
}

HISTORICAL_FILES = {
    "player_stats": Path("processed/player_stats_2018_2025.parquet"),
    "weekly_rosters": Path("raw/nflverse_weekly_rosters_2018_2025.parquet"),
    "injury_reports": Path("raw/nflverse_injury_reports_2018_2025.parquet"),
}

SNAPSHOT_NAMES = {
    "player_stats": "nflverse_player_stats.parquet",
    "weekly_rosters": "nflverse_weekly_rosters.parquet",
    "injury_reports": "nflverse_injury_reports.parquet",
}


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT COUNT(*) > 0
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [table],
        ).fetchone()[0]
    )


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _validate_source(name: str, df: pl.DataFrame, season: int) -> dict:
    """Reject structurally invalid downloads before changing persistent data."""
    if df.is_empty():
        raise ValueError(f"{name}: source returned no rows")

    required_by_source = {
        "player_stats": {
            "season", "week", "season_type", "game_id", "player_id",
            "player_name", "position", "team", "fantasy_points_ppr",
            "passing_yards", "passing_tds", "passing_interceptions",
            "rushing_yards", "rushing_tds", "receptions", "targets",
            "receiving_yards", "receiving_tds", "fumbles_lost_total",
        },
        "weekly_rosters": {
            "season", "week", "gsis_id", "team", "game_type",
        },
        "injury_reports": {
            "season", "week", "gsis_id", "team", "game_type",
            "report_status", "practice_status",
        },
    }

    missing = sorted(required_by_source[name] - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

    seasons = sorted(df["season"].drop_nulls().unique().to_list())
    if seasons != [season]:
        raise ValueError(
            f"{name}: expected only season {season}, received {seasons}"
        )

    invalid_weeks = df.filter(
        pl.col("week").is_null()
        | (pl.col("week") < 1)
        | (pl.col("week") > 22)
    ).height
    if invalid_weeks:
        raise ValueError(f"{name}: found {invalid_weeks} invalid week rows")

    if name == "player_stats":
        keyed = df.filter(pl.col("player_id").is_not_null())
        duplicates = (
            keyed.group_by(
                ["season", "week", "season_type", "player_id"]
            )
            .len()
            .filter(pl.col("len") > 1)
            .height
        )
        if duplicates:
            raise ValueError(
                f"{name}: found {duplicates} duplicate player-week groups"
            )

    # Roster rows with missing IDs and multiple injury reports are preserved.
    return {
        "rows": df.height,
        "columns": len(df.columns),
        "weeks": sorted(df["week"].drop_nulls().unique().to_list()),
    }


def _fetch_current_season(season: int) -> dict[str, pl.DataFrame]:
    """Fetch all source datasets before any persistent writes."""
    return {
        "player_stats": nflreadpy.load_player_stats(season),
        "weekly_rosters": nflreadpy.load_rosters_weekly(season),
        "injury_reports": nflreadpy.load_injuries(season),
    }


def _merge_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    current_snapshot: Path,
    historical_path: Path,
    season: int,
) -> None:
    """Replace one season while retaining every other season."""
    snapshot_sql = _sql_path(current_snapshot)
    temp_table = f"_next_{table}"

    if _table_exists(con, table):
        con.execute(
            f"""
            CREATE TEMP TABLE "{temp_table}" AS
            SELECT * FROM "{table}"
            WHERE season IS DISTINCT FROM {season}
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{snapshot_sql}')
            """
        )
    else:
        if not historical_path.is_file():
            raise FileNotFoundError(
                f"Cannot initialize {table}; historical data missing: "
                f"{historical_path}"
            )

        historical_sql = _sql_path(historical_path)
        con.execute(
            f"""
            CREATE TEMP TABLE "{temp_table}" AS
            SELECT * FROM read_parquet('{historical_sql}')
            WHERE season IS DISTINCT FROM {season}
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{snapshot_sql}')
            """
        )

    con.execute(
        f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM "{temp_table}"'
    )


def update_current_season(
    data_dir: str | Path = "data",
    db_path: str | Path | None = None,
    season: int = CURRENT_SEASON,
) -> dict:
    """Refresh the selected season in stable tables.

    Downloads are saved in timestamped, immutable snapshot directories.
    The three raw tables and derived training table are updated together
    inside one DuckDB transaction. Historical source files are not changed.
    """
    root = Path(data_dir)
    database_path = Path(db_path) if db_path is not None else root / "fantasy.duckdb"
    raw_dir = root / "raw"
    snapshot_root = raw_dir / "snapshots"
    processed_dir = root / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    historical_paths = {
        name: root / relative
        for name, relative in HISTORICAL_FILES.items()
    }

    # Download and validate every source before publishing anything.
    frames = _fetch_current_season(season)
    validations = {
        name: _validate_source(name, frame, season)
        for name, frame in frames.items()
    }

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_dir = snapshot_root / stamp
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    snapshots: dict[str, Path] = {}
    for name, frame in frames.items():
        path = snapshot_dir / SNAPSHOT_NAMES[name]
        frame.write_parquet(path)
        snapshots[name] = path

    training_path = processed_dir / "training_player_games_current.parquet"
    training_temp = processed_dir / ".training_player_games_current.tmp.parquet"
    database_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(database_path))
    try:
        con.execute("BEGIN TRANSACTION")
        try:
            for table, snapshot in snapshots.items():
                _merge_table(
                    con,
                    table,
                    snapshot,
                    historical_paths[table],
                    season,
                )

            # Rebuild derived data from the complete merged statistics table.
            merged_stats = con.execute(
                "SELECT * FROM player_stats"
            ).pl()

            training = build_training_dataset(merged_stats)
            con.register("_next_training_df", training)
            con.execute(
                """
                CREATE OR REPLACE TABLE training_player_games AS
                SELECT * FROM _next_training_df
                """
            )
            con.unregister("_next_training_df")

            training.write_parquet(training_temp)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    finally:
        con.close()

    # Publish the derived file only after the database transaction succeeds.
    os.replace(training_temp, training_path)

    return {
        "season": season,
        "database": str(database_path),
        "snapshots": {
            name: {
                **validations[name],
                "path": str(snapshots[name]),
            }
            for name in snapshots
        },
        "training_player_games": {
            "rows": training.height,
            "path": str(training_path),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(update_current_season(), indent=2, default=str))
