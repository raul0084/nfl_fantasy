
import polars as pl

from fantasy_ml.data.validation import (
    season_coverage,
    validate_player_stats,
)


def sample_stats():
    return pl.DataFrame({
        "season": [2024, 2024],
        "week": [1, 1],
        "season_type": ["REG", "REG"],
        "player_id": ["p1", "p2"],
        "player_name": ["Player One", "Player Two"],
        "position": ["RB", "WR"],
        "team": ["AAA", "BBB"],
        "passing_yards": [0, 0],
        "passing_tds": [0, 0],
        "passing_interceptions": [0, 0],
        "rushing_yards": [50, 0],
        "rushing_tds": [1, 0],
        "receptions": [2, 5],
        "targets": [3, 7],
        "receiving_yards": [10, 80],
        "receiving_tds": [0, 1],
        "fumbles_lost_total": [0, 0],
    })


def test_valid_player_stats_pass():
    result = validate_player_stats(sample_stats())

    assert result["rows"] == 2
    assert result["missing_columns"] == []
    assert result["duplicate_key_count"] == 0


def test_duplicate_player_week_key_is_detected():
    df = pl.concat([sample_stats(), sample_stats().head(1)])

    result = validate_player_stats(df)

    assert result["duplicate_key_count"] == 1


def test_season_coverage():
    result = season_coverage(sample_stats())

    assert result.height == 1
    assert result["season"][0] == 2024
    assert result["rows"][0] == 2
