
import polars as pl
import pytest

from fantasy_ml.features.weekly import build_weekly_features


def sample_stats():
    return pl.DataFrame({
        "season": [2024, 2024, 2024, 2024],
        "season_type": ["REG", "REG", "REG", "REG"],
        "week": [1, 2, 3, 4],
        "player_id": ["p1", "p1", "p1", "p1"],
        "fantasy_points_ppr": [10.0, 20.0, 30.0, 40.0],
    })


def test_rolling_features_exclude_current_week():
    result = build_weekly_features(sample_stats(), rolling_windows=(3,))

    averages = result["fantasy_points_ppr_avg_prev_3_games"].to_list()

    assert averages[0] is None
    assert averages[1] == pytest.approx(10.0)
    assert averages[2] == pytest.approx(15.0)
    assert averages[3] == pytest.approx(20.0)


def test_features_do_not_cross_seasons():
    df = pl.DataFrame({
        "season": [2024, 2025],
        "season_type": ["REG", "REG"],
        "week": [18, 1],
        "player_id": ["p1", "p1"],
        "fantasy_points_ppr": [30.0, 50.0],
    })

    result = build_weekly_features(df, rolling_windows=(3,))

    assert result["fantasy_points_ppr_avg_prev_3_games"].to_list() == [
        None, None
    ]


def test_missing_required_column_raises():
    df = pl.DataFrame({
        "season": [2024],
        "season_type": ["REG"],
        "week": [1],
        "player_id": ["p1"],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        build_weekly_features(df)
