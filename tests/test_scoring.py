
import polars as pl
import pytest

from fantasy_ml.scoring.ppr import ScoringSettings, add_fantasy_points


def test_full_ppr_scoring():
    df = pl.DataFrame({
        "passing_yards": [250],
        "passing_tds": [2],
        "passing_interceptions": [1],
        "rushing_yards": [20],
        "rushing_tds": [0],
        "receiving_yards": [100],
        "receiving_tds": [1],
        "receptions": [8],
        "fumbles_lost_total": [0],
    })

    result = add_fantasy_points(df)

    # 10 + 8 - 2 + 2 + 10 + 6 + 8 = 42
    assert result["calculated_fantasy_points"][0] == pytest.approx(42.0)


def test_custom_scoring_settings():
    df = pl.DataFrame({
        "passing_yards": [100],
        "passing_tds": [1],
        "passing_interceptions": [0],
        "rushing_yards": [0],
        "rushing_tds": [0],
        "receiving_yards": [0],
        "receiving_tds": [0],
        "receptions": [0],
        "fumbles_lost_total": [0],
    })

    settings = ScoringSettings(passing_td=6.0)
    result = add_fantasy_points(df, settings)

    assert result["calculated_fantasy_points"][0] == pytest.approx(10.0)


def test_missing_scoring_column_raises():
    with pytest.raises(ValueError, match="Missing scoring columns"):
        add_fantasy_points(pl.DataFrame({"passing_yards": [100]}))


def test_kicker_distance_scoring_and_source_preservation():
    df = pl.DataFrame({
        "position": ["K"],
        "passing_yards": [0],
        "passing_tds": [0],
        "passing_interceptions": [0],
        "rushing_yards": [0],
        "rushing_tds": [0],
        "receiving_yards": [0],
        "receiving_tds": [0],
        "receptions": [0],
        "fumbles_lost_total": [0],
        "fg_made_0_19": [1],
        "fg_made_20_29": [0],
        "fg_made_30_39": [1],
        "fg_made_40_49": [1],
        "fg_made_50_59": [1],
        "fg_made_60_": [1],
        "pat_made": [2],
        "fg_missed": [3],
        "fantasy_points": [0.0],
        "fantasy_points_ppr": [0.0],
    })

    result = add_fantasy_points(df)

    # Three short field goals, one 40–49, two 50+, two extra points.
    assert result["calculated_fantasy_points"][0] == pytest.approx(22.0)
    assert result["fantasy_points"][0] == 0.0
    assert result["fantasy_points_ppr"][0] == 0.0


def test_custom_kicker_scoring():
    df = pl.DataFrame({
        "position": ["K"],
        "passing_yards": [0],
        "passing_tds": [0],
        "passing_interceptions": [0],
        "rushing_yards": [0],
        "rushing_tds": [0],
        "receiving_yards": [0],
        "receiving_tds": [0],
        "receptions": [0],
        "fumbles_lost_total": [0],
        "fg_made_0_19": [0],
        "fg_made_20_29": [0],
        "fg_made_30_39": [0],
        "fg_made_40_49": [1],
        "fg_made_50_59": [0],
        "fg_made_60_": [0],
        "pat_made": [1],
    })

    settings = ScoringSettings(
        fg_40_49=5.0,
        extra_point_made=2.0,
    )
    result = add_fantasy_points(df, settings)

    assert result["calculated_fantasy_points"][0] == pytest.approx(7.0)
