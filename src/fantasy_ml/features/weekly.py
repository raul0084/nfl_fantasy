
"""Leakage-safe weekly player feature engineering."""

import polars as pl


# Statistics available in the dataset that may help predict future production.
CANDIDATE_FEATURES = (
    "fantasy_points_ppr",
    "targets",
    "carries",
    "receptions",
    "receiving_yards",
    "rushing_yards",
    "passing_yards",
    "attempts",
    "target_share",
    "air_yards_share",
    "wopr",
)


def build_weekly_features(
    df: pl.DataFrame,
    rolling_windows: tuple[int, ...] = (3, 5),
) -> pl.DataFrame:
    """Build rolling features using only prior observed player rows.

    Features reset at each season boundary. Rolling windows count prior
    observed rows, not necessarily consecutive calendar weeks.
    """
    required = {
        "season",
        "season_type",
        "week",
        "player_id",
        "fantasy_points_ppr",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if any(
        not isinstance(window, int) or isinstance(window, bool) or window < 1
        for window in rolling_windows
    ):
        raise ValueError("Rolling windows must be positive integers")

    if len(set(rolling_windows)) != len(rolling_windows):
        raise ValueError("Rolling windows must not contain duplicates")

    result = (
        df.filter(pl.col("season_type") == "REG")
        .filter(pl.col("player_id").is_not_null())
        .sort(["player_id", "season", "week"])
    )

    available_features = [
        column
        for column in CANDIDATE_FEATURES
        if column in result.columns
    ]

    expressions = []
    for column in available_features:
        for window in rolling_windows:
            expressions.append(
                pl.col(column)
                .shift(1)
                .rolling_mean(window_size=window, min_samples=1)
                .over(["player_id", "season"])
                .alias(f"{column}_avg_prev_{window}_games")
            )

    if expressions:
        result = result.with_columns(expressions)

    return result
