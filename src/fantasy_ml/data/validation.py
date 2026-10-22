
"""Validation helpers for nflreadpy player-stat datasets."""

import polars as pl

PLAYER_WEEK_KEY = ["season", "week", "season_type", "player_id"]

REQUIRED_COLUMNS = {
    *PLAYER_WEEK_KEY,
    "player_name",
    "position",
    "team",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost_total",
}


def validate_player_stats(df: pl.DataFrame) -> dict:
    """Return validation results; raise on structural failures."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    null_key_counts = {
        col: df[col].null_count()
        for col in PLAYER_WEEK_KEY
    }
    duplicate_keys = (
        df.group_by(PLAYER_WEEK_KEY)
        .len()
        .filter(pl.col("len") > 1)
    )

    invalid_seasons = df.filter(
        (pl.col("season") < 1999) | (pl.col("season") > 2100)
    ).height

    invalid_weeks = df.filter(
        (pl.col("week") < 1) | (pl.col("week") > 22)
    ).height

    return {
        "rows": df.height,
        "columns": len(df.columns),
        "seasons": sorted(df["season"].unique().to_list()),
        "missing_columns": missing_columns,
        "null_key_counts": null_key_counts,
        "duplicate_key_count": duplicate_keys.height,
        "invalid_season_rows": invalid_seasons,
        "invalid_week_rows": invalid_weeks,
    }


def season_coverage(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize row and week coverage by season and season type."""
    return (
        df.group_by(["season", "season_type"])
        .agg(
            pl.len().alias("rows"),
            pl.col("week").n_unique().alias("distinct_weeks"),
            pl.col("week").min().alias("min_week"),
            pl.col("week").max().alias("max_week"),
        )
        .sort(["season", "season_type"])
    )
