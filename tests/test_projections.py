from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from fantasy_ml.models.projections import (
    _make_upcoming_rows,
    get_next_upcoming_week,
)


def test_make_upcoming_rows_excludes_byes_and_nonfantasy_positions():
    schedule = pl.DataFrame(
        {
            "season": [2026],
            "week": [8],
            "game_type": ["REG"],
            "game_id": ["2026_08_BUF_KC"],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "gameday": ["2026-10-29"],
            "gametime": ["20:15"],
        }
    )
    roster = pl.DataFrame(
        {
            "team": ["KC", "BUF", "NYJ", "KC"],
            "position": ["QB", "WR", "RB", "LS"],
            "gsis_id": ["p1", "p2", "p3", "p4"],
            "full_name": ["Quarterback", "Receiver", "Bye Player", "Long Snapper"],
            "status": ["ACT", "ACT", "ACT", "ACT"],
        }
    )

    result = _make_upcoming_rows(schedule, roster, 2026, 8, 7)

    assert set(result["player_id"].to_list()) == {"p1", "p2"}
    assert result.height == 2
    assert set(result["opponent_team"].to_list()) == {"KC", "BUF"}
    assert result["roster_snapshot_week"].to_list() == [7, 7]


def test_next_upcoming_week_uses_future_kickoff(monkeypatch):
    schedule = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [7, 8, 9],
            "game_type": ["REG", "REG", "REG"],
            "gameday": ["2026-10-25", "2026-10-29", "2026-11-05"],
            "gametime": ["13:00", "20:15", "20:15"],
            "result": [3.0, None, None],
        }
    )

    class FakeNFLReadPy:
        @staticmethod
        def load_schedules(season):
            return schedule.filter(pl.col("season") == season)

    monkeypatch.setattr("fantasy_ml.models.projections.nflreadpy", FakeNFLReadPy)
    now = datetime(2026, 10, 26, 12, 0, tzinfo=ZoneInfo("America/New_York"))

    assert get_next_upcoming_week(2026, now=now) == (2026, 8)


def test_next_upcoming_week_errors_when_no_future_kickoff(monkeypatch):
    schedule = pl.DataFrame(
        {
            "season": [2026],
            "week": [7],
            "game_type": ["REG"],
            "gameday": ["2026-10-25"],
            "gametime": ["13:00"],
            "result": [3.0],
        }
    )

    class FakeNFLReadPy:
        @staticmethod
        def load_schedules(season):
            return schedule

    monkeypatch.setattr("fantasy_ml.models.projections.nflreadpy", FakeNFLReadPy)

    with pytest.raises(ValueError, match="No upcoming regular-season kickoff"):
        get_next_upcoming_week(
            2026,
            now=datetime(2026, 10, 26, 12, 0, tzinfo=ZoneInfo("America/New_York")),
        )
