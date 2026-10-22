
"""Configurable fantasy football scoring."""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class ScoringSettings:
    passing_yards_per_point: float = 25.0
    passing_td: float = 4.0
    interception: float = -2.0
    rushing_yards_per_point: float = 10.0
    rushing_td: float = 6.0
    receiving_yards_per_point: float = 10.0
    receiving_td: float = 6.0
    reception: float = 1.0
    fumble_lost: float = -2.0
    two_point_conversion: float = 2.0
    special_teams_td: float = 6.0


def add_fantasy_points(
    df: pl.DataFrame,
    settings: ScoringSettings = ScoringSettings(),
) -> pl.DataFrame:
    """Return a copy with calculated fantasy points under the supplied rules."""
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

    score = (
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

    if "passing_2pt_conversions" in df.columns:
        score += stat("passing_2pt_conversions") * settings.two_point_conversion
    if "rushing_2pt_conversions" in df.columns:
        score += stat("rushing_2pt_conversions") * settings.two_point_conversion
    if "receiving_2pt_conversions" in df.columns:
        score += stat("receiving_2pt_conversions") * settings.two_point_conversion
    if "special_teams_tds" in df.columns:
        score += stat("special_teams_tds") * settings.special_teams_td

    return df.with_columns(score.alias("calculated_fantasy_points"))
