
"""Build leakage-safe player-game training data."""

from __future__ import annotations

import polars as pl
from fantasy_ml.scoring.ppr import add_fantasy_points

from fantasy_ml.features.weekly import (
    CANDIDATE_FEATURES,
    build_weekly_features,
)

TARGET = "model_fantasy_points"

METADATA_COLUMNS = (
    "season",
    "season_type",
    "week",
    "game_id",
    "player_id",
    "player_name",
)

CONTEXT_FEATURES = (
    "position",
    "team",
    "opponent_team",
)

REQUIRED_COLUMNS = (
    "season",
    "season_type",
    "week",
    "game_id",
    "player_id",
    "fantasy_points_ppr",
)

def _add_model_target(df: pl.DataFrame) -> pl.DataFrame:
    """Create the model target while preserving source PPR scores."""
    if "fantasy_points_ppr" not in df.columns:
        raise ValueError("Missing required source column: fantasy_points_ppr")

    if "position" not in df.columns:
        raise ValueError("Missing required column for model target: position")

    # Keep original PPR scoring for all non-kickers.
    result = df.with_columns(
        pl.col("fantasy_points_ppr").alias("model_fantasy_points")
    )

    if result.filter(pl.col("position") == "K").is_empty():
        return result

    # Calculate the configured default kicker score. The scorer does not
    # overwrite either source fantasy-points column.
    scored = add_fantasy_points(result)

    return scored.with_columns(
        pl.when(pl.col("position") == "K")
        .then(pl.col("calculated_fantasy_points"))
        .otherwise(pl.col("fantasy_points_ppr"))
        .alias("model_fantasy_points")
    ).drop("calculated_fantasy_points")


def build_training_dataset(
    df: pl.DataFrame,
    rolling_windows: tuple[int, ...] = (3, 5),
) -> pl.DataFrame:
    """Return player-game metadata, pregame features, and target.

    Only regular-season games with a known player ID and target are
    retained. Same-game performance statistics are never model inputs.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required training columns: {missing}"
        )

    duplicate_keys = (
        df.filter(pl.col("season_type") == "REG")
        .filter(pl.col("player_id").is_not_null())
        .group_by(["game_id", "player_id"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_keys:
        raise ValueError(
            f"Found {duplicate_keys} duplicate player-game keys"
        )

    # Standard offensive fantasy positions only.
    offensive_positions = ["QB", "RB", "WR", "TE", "K"]

    prepared = _add_model_target(
        df.filter(
            pl.col("position").is_in(offensive_positions)
        )
    )

    features = build_weekly_features(
        prepared,
        rolling_windows=rolling_windows,
    )

    metadata = [
        column
        for column in METADATA_COLUMNS
        if column in features.columns
    ]

    context = [
        column
        for column in CONTEXT_FEATURES
        if column in features.columns
    ]

    rolling = [
        f"{column}_avg_prev_{window}_games"
        for column in CANDIDATE_FEATURES
        if column in features.columns
        for window in rolling_windows
        if f"{column}_avg_prev_{window}_games"
        in features.columns
    ]

    # Explicit allowlist: never pass raw same-game statistics to X.
    selected = list(dict.fromkeys(
        metadata + context + rolling + [TARGET]
    ))

    result = (
        features
        .filter(pl.col("player_id").is_not_null())
        .filter(pl.col(TARGET).is_not_null())
        .select(selected)
        .sort(["season", "week", "game_id", "player_id"])
    )

    return result


def split_features_target(
    dataset: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.Series, pl.DataFrame]:
    """Separate model inputs, target, and evaluation metadata."""
    if TARGET not in dataset.columns:
        raise ValueError(f"Dataset must contain {TARGET!r}")

    metadata_columns = [
        column
        for column in METADATA_COLUMNS
        if column in dataset.columns
    ]

    # Only these columns are eligible to become model inputs.
    feature_columns = [
        column
        for column in CONTEXT_FEATURES
        if column in dataset.columns
    ] + [
        column
        for column in dataset.columns
        if "_avg_prev_" in column
    ]

    if not feature_columns:
        raise ValueError("No model feature columns found")

    X = dataset.select(feature_columns)
    y = dataset.get_column(TARGET)
    metadata = dataset.select(metadata_columns)

    return X, y, metadata



def build_prediction_dataset(
    historical_stats: pl.DataFrame,
    upcoming_rows: pl.DataFrame,
    rolling_windows: tuple[int, ...] = (3, 5),
) -> pl.DataFrame:
    """Build leakage-safe features for one upcoming regular-season week."""
    required = {
        "season", "season_type", "week", "game_id", "player_id",
        "position", "team", "opponent_team",
    }
    missing = sorted(required - set(upcoming_rows.columns))
    if missing:
        raise ValueError(f"Missing upcoming-row columns: {missing}")
    if upcoming_rows.is_empty():
        raise ValueError("upcoming_rows must not be empty")

    target_weeks = upcoming_rows.select("season", "week").unique()
    if target_weeks.height != 1:
        raise ValueError(
            "upcoming_rows must contain exactly one season and week"
        )
    if upcoming_rows.filter(
        pl.col("season_type") != "REG"
    ).height:
        raise ValueError("Upcoming rows must be regular-season games")
    if upcoming_rows.filter(
        pl.col("player_id").is_null() | pl.col("game_id").is_null()
    ).height:
        raise ValueError("Upcoming rows must have player_id and game_id")
    if (
        upcoming_rows.group_by(["season", "week", "player_id"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    ):
        raise ValueError("Duplicate player-week rows in upcoming_rows")

    target_season = target_weeks["season"][0]
    target_week = target_weeks["week"][0]

    history = historical_stats.filter(
        (pl.col("season") < target_season)
        | (
            (pl.col("season") == target_season)
            & (pl.col("week") < target_week)
        )
    )

    history = _add_model_target(history).with_columns(
        pl.lit(False).alias("__is_forecast")
    )

    future = upcoming_rows
    if TARGET not in future.columns:
        future = future.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(TARGET)
        )
    else:
        future = future.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(TARGET)
        )

    future = future.with_columns(
        pl.lit(True).alias("__is_forecast")
    )

    combined = pl.concat([history, future], how="diagonal_relaxed")
    featured = build_weekly_features(
        combined, rolling_windows=rolling_windows
    )
    predictions = featured.filter(pl.col("__is_forecast")).drop(
        "__is_forecast"
    )

    metadata = [
        col for col in METADATA_COLUMNS if col in predictions.columns
    ]
    context = [
        col for col in CONTEXT_FEATURES if col in predictions.columns
    ]
    rolling = [
        f"{col}_avg_prev_{window}_games"
        for col in CANDIDATE_FEATURES
        for window in rolling_windows
        if f"{col}_avg_prev_{window}_games" in predictions.columns
    ]
    selected = list(
        dict.fromkeys(metadata + context + rolling + [TARGET])
    )

    return predictions.select(selected).sort(
        ["season", "week", "game_id", "player_id"]
    )
