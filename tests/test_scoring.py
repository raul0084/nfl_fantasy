
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
