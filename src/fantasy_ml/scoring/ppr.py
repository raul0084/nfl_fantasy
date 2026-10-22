
"""Configurable fantasy football scoring."""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class ScoringSettings:
    # Passing
    passing_yards_per_point: float = 25.0
    passing_td: float = 4.0
    interception: float = -2.0

    # Rushing and receiving
    rushing_yards_per_point: float = 10.0
    rushing_td: float = 6.0
    receiving_yards_per_point: float = 10.0
    receiving_td: float = 6.0
    reception: float = 1.0

    # Other offensive scoring
    fumble_lost: float = -2.0
    two_point_conversion: float = 2.0
    special_teams_td: float = 6.0

    # Kicker scoring
    fg_0_39: float = 3.0
    fg_40_49: float = 4.0
    fg_50_plus: float = 5.0
    extra_point_made: float = 1.0
    missed_field_goal: float = 0.0


FG_BUCKETS = (
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
)


def add_fantasy_points(
    df: pl.DataFrame,
    settings: ScoringSettings = ScoringSettings(),
) -> pl.DataFrame:
    """Add calculated fantasy points without modifying source score columns.

    When position is available, K rows use kicker scoring and other rows
    use offensive scoring. Without position, all rows use offensive scoring.
    """
    required = {
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "rushing_yards",
        "rushing_tds",
        "receiving_yards",
        "receiving_tds",
        "receptions",
        "fumbles_lost_total",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing scoring columns: {missing}")

    def stat(name: str) -> pl.Expr:
        return pl.col(name).fill_null(0)

    offensive_score = (
        stat("passing_yards") / settings.passing_yards_per_point
        + stat("passing_tds") * settings.passing_td
        + stat("passing_interceptions") * settings.interception
        + stat("rushing_yards") / settings.rushing_yards_per_point
        + stat("rushing_tds") * settings.rushing_td
        + stat("receiving_yards") / settings.receiving_yards_per_point
        + stat("receiving_tds") * settings.receiving_td
        + stat("receptions") * settings.reception
        + stat("fumbles_lost_total") * settings.fumble_lost
    )

    for column in (
        "passing_2pt_conversions",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
    ):
        if column in df.columns:
            offensive_score += stat(column) * settings.two_point_conversion

    if "special_teams_tds" in df.columns:
        offensive_score += stat("special_teams_tds") * settings.special_teams_td

    # Keep the existing behavior for data without position information.
    if "position" not in df.columns:
        return df.with_columns(
            offensive_score.alias("calculated_fantasy_points")
        )

    has_kickers = df.filter(pl.col("position") == "K").height > 0

    if not has_kickers:
        return df.with_columns(
            offensive_score.alias("calculated_fantasy_points")
        )

    kicker_required = set(FG_BUCKETS) | {"pat_made"}
    missing_kicker = sorted(kicker_required - set(df.columns))
    if missing_kicker:
        raise ValueError(
            f"Missing kicker scoring columns: {missing_kicker}"
        )

    # Distance-based scoring: 0–39 yards = 3, 40–49 = 4, 50+ = 5.
    # The 50+ tier includes both the 50–59 and 60+ source buckets.
    kicker_score = (
        (stat("fg_made_0_19") + stat("fg_made_20_29")
         + stat("fg_made_30_39")) * settings.fg_0_39
        + stat("fg_made_40_49") * settings.fg_40_49
        + (stat("fg_made_50_59") + stat("fg_made_60_"))
        * settings.fg_50_plus
        + stat("pat_made") * settings.extra_point_made
    )

    # Missed field goals contribute zero by default. The setting is exposed
    # for configurability if a league uses a nonzero penalty or bonus.
    if settings.missed_field_goal != 0:
        if "fg_missed" not in df.columns:
            raise ValueError(
                "Missing kicker scoring columns: ['fg_missed']"
            )
        kicker_score += stat("fg_missed") * settings.missed_field_goal

    score = (
        pl.when(pl.col("position") == "K")
        .then(kicker_score)
        .otherwise(offensive_score)
    )

    return df.with_columns(score.alias("calculated_fantasy_points"))
