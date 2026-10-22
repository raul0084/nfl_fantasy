
import numpy as np
import polars as pl
import pytest

from fantasy_ml.models.ridge import (
    fit_ridge,
    predict_ridge,
    regression_metrics,
)


def sample_training_rows():
    return pl.DataFrame(
        {
            "season": [2021, 2021, 2022, 2022, 2023, 2023],
            "season_type": ["REG"] * 6,
            "week": [1, 1, 1, 1, 1, 1],
            "game_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "player_id": ["p1", "p2", "p1", "p2", "p1", "p2"],
            "player_name": ["A", "B", "A", "B", "A", "B"],
            "position": ["RB", "WR", "RB", "WR", "RB", "WR"],
            "team": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "opponent_team": ["BBB", "AAA", "BBB", "AAA", "BBB", "AAA"],
            "fantasy_points_ppr_avg_prev_3_games": [
                None, None, 8.0, 4.0, 10.0, 5.0
            ],
            "fantasy_points_ppr_avg_prev_5_games": [
                None, None, 7.0, 3.0, 9.0, 4.0
            ],
            "fantasy_points_ppr": [8.0, 4.0, 10.0, 5.0, 11.0, 6.0],
        }
    )


def test_ridge_fits_and_predicts():
    df = sample_training_rows()
    train = df.filter(pl.col("season") < 2023)
    target = df.filter(pl.col("season") == 2023)

    model = fit_ridge(train, alpha=1.0)
    predictions = predict_ridge(model, target)

    assert len(predictions) == target.height
    assert np.isfinite(predictions).all()


def test_ridge_rejects_nonpositive_alpha():
    with pytest.raises(ValueError, match="alpha must be positive"):
        fit_ridge(sample_training_rows(), alpha=0)


def test_regression_metrics():
    result = regression_metrics(
        np.array([1.0, 3.0]),
        np.array([2.0, 3.0]),
    )

    assert result["mae"] == pytest.approx(0.5)
    assert result["rmse"] == pytest.approx(np.sqrt(0.5))
