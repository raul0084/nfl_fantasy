
"""Build leakage-safe player-game training data."""

from __future__ import annotations

import polars as pl

from fantasy_ml.features.weekly import (
    CANDIDATE_FEATURES,
    build_weekly_features,
)

TARGET = "fantasy_points_ppr"

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
    TARGET,
)


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

    features = build_weekly_features(
        df,
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

    duplicate_keys = (
        result
        .group_by(["game_id", "player_id"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_keys:
        raise ValueError(
            f"Found {duplicate_keys} duplicate player-game keys"
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
