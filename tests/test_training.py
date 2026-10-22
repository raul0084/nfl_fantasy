
import polars as pl
import pytest

from fantasy_ml.data.training import (
    build_training_dataset,
    split_features_target,
)


def sample_stats():
    return pl.DataFrame(
        {
            "season": [2024] * 4,
            "season_type": ["REG"] * 4,
            "week": [1, 2, 3, 4],
            "game_id": ["g1", "g2", "g3", "g4"],
            "player_id": ["p1"] * 4,
            "player_name": ["Player One"] * 4,
            "position": ["WR"] * 4,
            "team": ["AAA"] * 4,
            "opponent_team": ["BBB"] * 4,
            "fantasy_points_ppr": [10.0, 20.0, 5.0, 15.0],
            "targets": [5, 8, 3, 7],
            "receptions": [3, 6, 2, 5],
            "receiving_yards": [40, 90, 20, 70],
        }
    )


def test_training_target_is_actual_ppr():
    result = build_training_dataset(sample_stats())

    assert result["model_fantasy_points"].to_list() == [
        10.0, 20.0, 5.0, 15.0
    ]


def test_current_game_statistics_are_not_features():
    result = build_training_dataset(sample_stats())
    X, y, metadata = split_features_target(result)

    assert "targets" not in X.columns
    assert "receptions" not in X.columns
    assert "receiving_yards" not in X.columns
    assert "model_fantasy_points" not in X.columns
    assert len(X) == len(y) == len(metadata)


def test_rolling_features_use_only_prior_games():
    result = build_training_dataset(
        sample_stats(),
        rolling_windows=(3,),
    )

    # Week 2's rolling PPR average uses Week 1 only.
    assert result[
        "model_fantasy_points_avg_prev_3_games"
    ].to_list()[1] == pytest.approx(10.0)


def test_missing_player_ids_are_excluded():
    df = sample_stats().with_columns(
        pl.when(pl.col("week") == 4)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("player_id"))
        .alias("player_id")
    )

    result = build_training_dataset(df)

    assert result.height == 3


def test_duplicate_player_game_is_rejected():
    df = pl.concat([sample_stats(), sample_stats().head(1)])

    with pytest.raises(ValueError, match="duplicate player-game"):
        build_training_dataset(df)

def test_prediction_dataset_uses_only_pre_week_history():
    from fantasy_ml.data.training import build_prediction_dataset

    history = sample_stats()

    upcoming = pl.DataFrame({
        "season": [2024],
        "season_type": ["REG"],
        "week": [5],
        "game_id": ["g5"],
        "player_id": ["p1"],
        "player_name": ["Player One"],
        "position": ["WR"],
        "team": ["AAA"],
        "opponent_team": ["BBB"],
    })

    result = build_prediction_dataset(history, upcoming)

    assert result.height == 1
    assert result["week"].to_list() == [5]
    assert result["model_fantasy_points"].to_list() == [None]

    # Prior three observed games: weeks 2, 3 and 4.
    assert result[
        "model_fantasy_points_avg_prev_3_games"
    ][0] == pytest.approx((20.0 + 5.0 + 15.0) / 3)


def test_prediction_dataset_excludes_target_week_history():
    from fantasy_ml.data.training import build_prediction_dataset

    history = sample_stats()

    # Deliberately include a target-week record in historical input.
    target_week_record = pl.DataFrame({
        "season": [2024],
        "season_type": ["REG"],
        "week": [5],
        "game_id": ["g5"],
        "player_id": ["p1"],
        "player_name": ["Player One"],
        "position": ["WR"],
        "team": ["AAA"],
        "opponent_team": ["BBB"],
        "fantasy_points_ppr": [999.0],
        "targets": [99],
        "receptions": [99],
        "receiving_yards": [999],
    })
    history = pl.concat(
        [history, target_week_record],
        how="diagonal_relaxed",
    )

    upcoming = target_week_record.drop(
        "fantasy_points_ppr", "targets", "receptions", "receiving_yards"
    )

    result = build_prediction_dataset(history, upcoming)

    assert result["model_fantasy_points"][0] is None
    assert result[
        "model_fantasy_points_avg_prev_3_games"
    ][0] == pytest.approx((20.0 + 5.0 + 15.0) / 3)

def test_training_keeps_only_offensive_fantasy_positions():
    df = pl.concat(
        [
            sample_stats(),
            sample_stats().head(1).with_columns(
                pl.lit("CB").alias("position"),
                pl.lit("defensive-player").alias("player_id"),
                pl.lit("g5").alias("game_id"),
            ),
        ]
    )

    result = build_training_dataset(df)

    assert set(result["position"].unique().to_list()) == {"WR"}
    assert "defensive-player" not in result["player_id"].to_list()

