"""Reusable weekly Full-PPR projection pipeline.

Train only on completed historical seasons (2018-2025 by default), use all
available prior player-game data to build pre-kickoff rolling features, and
project players on NFL teams scheduled to play in the requested week.
"""
from __future__ import annotations

import argparse
import math
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import nflreadpy
import polars as pl

from fantasy_ml.data.training import build_prediction_dataset, build_training_dataset
from fantasy_ml.models.ridge import fit_ridge, predict_ridge

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K")
DEFAULT_DB_PATH = Path("data/fantasy.duckdb")
DEFAULT_OUTPUT_DIR = Path("artifacts/predictions")
EASTERN = ZoneInfo("America/New_York")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _kickoff(row: dict[str, Any]) -> datetime | None:
    """Parse nflverse's gameday/gametime, whose listed game time is US Eastern."""
    day, clock = row.get("gameday"), row.get("gametime")
    if _is_missing(day) or _is_missing(clock):
        return None
    try:
        parsed_day = _as_date(day)
        parsed_time = time.fromisoformat(str(clock).strip()[:8])
    except (TypeError, ValueError):
        return None
    return datetime.combine(parsed_day, parsed_time).replace(tzinfo=EASTERN)


def get_next_upcoming_week(
    season: int | None = None,
    now: datetime | None = None,
) -> tuple[int, int]:
    """Return (season, week) for the earliest scheduled REG game not yet kicked off.

    `now` is injectable for deterministic tests. Naive datetimes are interpreted
    as US Eastern time. The function does not assume a fixed NFL week number.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=EASTERN)
    now = now.astimezone(timezone.utc)

    if season is None:
        season = now.astimezone(EASTERN).year

    schedules = nflreadpy.load_schedules(season)
    candidates: list[tuple[datetime, int]] = []

    for row in schedules.iter_rows(named=True):
        if int(row.get("season", season)) != season:
            continue
        if str(row.get("game_type", "")).upper() != "REG":
            continue
        if _is_missing(row.get("week")):
            continue
        kickoff = _kickoff(row)
        if kickoff is None or kickoff.astimezone(timezone.utc) <= now:
            continue
        # Future games should have no final result. The kickoff check is primary,
        # but excluding completed rows also guards inconsistent schedule records.
        result = row.get("result")
        if not _is_missing(result):
            continue
        candidates.append((kickoff, int(row["week"])))

    if not candidates:
        raise ValueError(
            f"No upcoming regular-season kickoff found for season {season}. "
            "Pass --season/--week explicitly or refresh the schedule data."
        )

    kickoff, week = min(candidates, key=lambda item: item[0])
    return season, week


def _read_table(con: duckdb.DuckDBPyConnection, table: str) -> pl.DataFrame:
    # Table names are internal constants, never user input.
    return pl.from_arrow(con.execute(f"SELECT * FROM {table}").fetch_arrow_table())


def _load_stats(db_path: Path, through_season: int) -> pl.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")
    with duckdb.connect(str(db_path), read_only=True) as con:
        available = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        if "player_stats" not in available:
            raise ValueError(f"{db_path} has no player_stats table")
        stats = pl.from_arrow(
            con.execute(
                "SELECT * FROM player_stats WHERE season <= ?",
                [through_season],
            ).fetch_arrow_table()
        )
    if stats.is_empty():
        raise ValueError(f"No player_stats rows found through season {through_season}")
    if "season_type" in stats.columns:
        stats = stats.filter(pl.col("season_type") == "REG")
    return stats



def _load_rosters(
    db_path: Path,
    season: int,
    target_week: int,
    roster_week: int | None = None,
) -> tuple[pl.DataFrame, int]:
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    with duckdb.connect(str(db_path), read_only=True) as con:
        available = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        if "weekly_rosters" not in available:
            raise ValueError(f"{db_path} has no weekly_rosters table")

        if roster_week is not None:
            # Explicit override: use exactly the requested snapshot week.
            roster = pl.from_arrow(
                con.execute(
                    """
                    SELECT *
                    FROM weekly_rosters
                    WHERE season = ? AND week = ?
                    """,
                    [season, roster_week],
                ).fetch_arrow_table()
            )
            if roster.is_empty():
                raise ValueError(
                    f"No weekly_rosters snapshot for season {season}, "
                    f"week {roster_week}."
                )
            used_week = int(roster_week)
        else:
            # Use the newest available snapshot for each team. Bye-week
            # teams may have an older snapshot than teams that played.
            roster = pl.from_arrow(
                con.execute(
                    """
                    WITH latest_team_week AS (
                        SELECT team, MAX(week) AS week
                        FROM weekly_rosters
                        WHERE season = ? AND week <= ?
                        AND team IS NOT NULL
                        GROUP BY team
                    ),
                    team_snapshots AS (
                        SELECT r.*
                        FROM weekly_rosters AS r
                        INNER JOIN latest_team_week AS latest
                            ON r.team = latest.team
                        AND r.week = latest.week
                        WHERE r.season = ?
                    ),
                    ranked_players AS (
                        SELECT *,
                            ROW_NUMBER() OVER (
                                PARTITION BY gsis_id
                                ORDER BY week DESC,
                                        CASE
                                            WHEN status = 'CUT' THEN 1
                                            ELSE 0
                                        END ASC,
                                        team ASC
                            ) AS player_row
                        FROM team_snapshots
                        WHERE gsis_id IS NOT NULL
                    )
                    SELECT * EXCLUDE (player_row)
                    FROM ranked_players
                    WHERE player_row = 1
                    """,
                    [season, target_week, season],
                ).fetch_arrow_table()
            )
            if roster.is_empty():
                raise ValueError(
                    f"No weekly_rosters snapshots for season {season} "
                    f"at or before week {target_week}."
                )
            used_week = int(roster["week"].max())

    required = {"team", "position", "gsis_id", "full_name", "week"}
    missing = sorted(required - set(roster.columns))
    if missing:
        raise ValueError(f"weekly_rosters missing required columns: {missing}")

    roster = roster.filter(
        pl.col("position").is_in(FANTASY_POSITIONS)
        & pl.col("gsis_id").is_not_null()
        & pl.col("team").is_not_null()
    )

    # NFL roster status is not game-day inactive status.
    # Keep it as context; do not claim players are confirmed active.
    return roster, used_week



def _make_upcoming_rows(
    schedule: pl.DataFrame,
    roster: pl.DataFrame,
    season: int,
    week: int,
    roster_week: int,
) -> pl.DataFrame:
    games = schedule.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("game_type") == "REG")
    )
    if games.is_empty():
        raise ValueError(f"No regular-season schedule rows for {season} week {week}")

    game_records: list[dict[str, Any]] = []
    team_to_game: dict[str, tuple[str, str, Any, Any]] = {}
    for game in games.iter_rows(named=True):
        home, away = game.get("home_team"), game.get("away_team")
        game_id = game.get("game_id")
        for team, opponent in ((home, away), (away, home)):
            if not _is_missing(team):
                team_to_game[str(team)] = (
                    str(opponent), str(game_id), game.get("gameday"), game.get("gametime")
                )

    for player in roster.iter_rows(named=True):
        if player.get("position") not in FANTASY_POSITIONS:
            continue
        team = str(player["team"])
        if team not in team_to_game:
            continue  # Bye week or team not present in the selected schedule.
        opponent, game_id, gameday, gametime = team_to_game[team]
        player_id = str(player["gsis_id"])
        item = {
            "season": season,
            "season_type": "REG",
            "week": week,
            "game_id": game_id,
            "player_id": player_id,
            "player_name": player.get("full_name"),
            "position": player.get("position"),
            "team": team,
            "opponent_team": opponent,
            "roster_status": player.get("status"),
            "roster_snapshot_week": roster_week,
            "gameday": gameday,
            "gametime": gametime,
        }
        game_records.append(item)

    if not game_records:
        raise ValueError(
            f"No eligible fantasy-position players from roster snapshot week "
            f"{roster_week} have a scheduled game in {season} week {week}."
        )

    upcoming = pl.DataFrame(game_records, infer_schema_length=None)
    duplicates = upcoming.group_by(["season", "week", "player_id"]).len().filter(
        pl.col("len") > 1
    )
    if not duplicates.is_empty():
        raise ValueError(
            "Roster/schedule join produced duplicate player-week rows; "
            f"examples: {duplicates.head(5).to_dicts()}"
        )
    return upcoming



def project_week(
    season: int,
    week: int,
    db_path: str | Path = DEFAULT_DB_PATH,
    roster_week: int | None = None,
    alpha: float = 1.0,
    train_through_season: int = 2025,
) -> pl.DataFrame:
    """Fit historical Ridge and return projections for eligible players that week.

    Model fitting is restricted to regular-season player games through
    `train_through_season`. Rolling features may use available prior stats
    through the target week, including earlier games in the target season.
    """
    db_path = Path(db_path)

    if not 1 <= week <= 18:
        raise ValueError(f"week must be between 1 and 18; received {week}")

    if train_through_season >= season:
        raise ValueError(
            "train_through_season must be earlier than the projection season "
            "to prevent training on the target season."
        )

    schedules = nflreadpy.load_schedules(season)
    if schedules.is_empty():
        raise ValueError(f"No schedule data returned for season {season}")

    # Load history through the projection season so feature engineering can
    # use all available games before the target week.
    historical_stats = _load_stats(db_path, through_season=season)

    # Keep model training restricted to the configured historical seasons.
    train_stats = historical_stats.filter(
        pl.col("season") <= train_through_season
    )
    training_rows = build_training_dataset(train_stats)

    if training_rows.is_empty():
        raise ValueError(
            f"No training rows available through {train_through_season}; "
            "check player_stats coverage and scoring columns."
        )

    model = fit_ridge(training_rows, alpha=alpha)

    roster, selected_roster_week = _load_rosters(
        db_path,
        season,
        week,
        roster_week=roster_week,
    )

    upcoming = _make_upcoming_rows(
        schedules,
        roster,
        season,
        week,
        selected_roster_week,
    )

    if upcoming.is_empty():
        raise ValueError(
            f"No eligible upcoming players found for season {season}, "
            f"week {week}."
        )

    # Build features. This function may reorder rows, so do not assume
    # prediction order matches the original upcoming roster order.
    prediction_rows = build_prediction_dataset(
        historical_stats=historical_stats,
        upcoming_rows=upcoming.select(
            "season",
            "season_type",
            "week",
            "game_id",
            "player_id",
            "position",
            "team",
            "opponent_team",
            "player_name",
        ),
    )

    predictions = predict_ridge(model, prediction_rows)

    if len(predictions) != prediction_rows.height:
        raise ValueError(
            "Prediction count does not match prediction feature rows: "
            f"{len(predictions)} predictions for "
            f"{prediction_rows.height} rows."
        )

    # Attach predictions to stable player/game keys before joining to roster
    # metadata. This prevents a sort-order difference from assigning one
    # player's prediction to another player.
    prediction_frame = (
        prediction_rows.select("player_id", "game_id")
        .with_columns(
            pl.col("player_id").cast(pl.String),
            pl.Series(
                "projected_points_ppr",
                predictions,
            ),
        )
        .with_columns(
            pl.col("projected_points_ppr").round(2)
        )
    )

    duplicate_prediction_keys = (
        prediction_frame.group_by(["player_id", "game_id"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicate_prediction_keys.is_empty():
        raise ValueError(
            "Prediction features contain duplicate player/game keys: "
            f"{duplicate_prediction_keys.head(5).to_dicts()}"
        )

    # Count historical player-game rows before the target week.
    prior = historical_stats.filter(
        (pl.col("season") < season)
        | (
            (pl.col("season") == season)
            & (pl.col("week") < week)
        )
    )

    prior_counts = (
        prior.group_by("player_id")
        .len(name="prior_game_rows")
        .with_columns(
            pl.col("player_id").cast(pl.String)
        )
    )

    metadata = upcoming.with_columns(
        pl.col("player_id").cast(pl.String)
    )

    output = (
        metadata.join(
            prior_counts,
            on="player_id",
            how="left",
        )
        .join(
            prediction_frame,
            on=["player_id", "game_id"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("prior_game_rows")
            .fill_null(0)
            .cast(pl.Int64),
        )
        .with_columns(
            (pl.col("prior_game_rows") > 0).alias("has_prior_games"),
        )
    )

    missing_predictions = output.filter(
        pl.col("projected_points_ppr").is_null()
    )
    if not missing_predictions.is_empty():
        raise ValueError(
            "Some upcoming player/game rows did not match their predictions: "
            f"{missing_predictions.select('player_id', 'game_id').head(5).to_dicts()}"
        )

    
    output = output.with_columns(
        pl.col("projected_points_ppr").round(2),
        pl.lit("ridge").alias("model"),
        pl.lit(float(alpha)).alias("ridge_alpha"),
    ).sort(
        ["position", "projected_points_ppr"],
        descending=[False, True],
    )


    return output.select(
        "season",
        "week",
        "gameday",
        "gametime",
        "game_id",
        "player_id",
        "player_name",
        "position",
        "team",
        "opponent_team",
        "roster_status",
        "roster_snapshot_week",
        "prior_game_rows",
        "has_prior_games",
        "projected_points_ppr",
        "model",
        "ridge_alpha",
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weekly NFL Full-PPR projections.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--roster-week", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-through-season", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.week is None:
        season, week = get_next_upcoming_week(season=args.season)
    else:
        season = args.season or datetime.now(EASTERN).year
        week = args.week

    projections = project_week(
        season=season,
        week=week,
        db_path=args.db_path,
        roster_week=args.roster_week,
        alpha=args.alpha,
        train_through_season=args.train_through_season,
    )
    output_path = args.output or (
        DEFAULT_OUTPUT_DIR / f"projections_{season}_week_{week:02d}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    projections.write_csv(output_path)
    print(f"Season/week: {season} / {week}")
    print(f"Players projected: {projections.height}")
    print(f"Output: {output_path}")
    print(projections.head(15))


if __name__ == "__main__":
    main()
