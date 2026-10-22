
"""Evaluate a leakage-safe, recency-weighted seasonal PPR baseline."""
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/processed/training_player_games_2018_2025.parquet")
TARGET = "fantasy_points_ppr"
DECAYS = (0.4, 0.6, 0.8)
VALIDATION_SEASON = 2023
TEST_SEASONS = (2024, 2025)


def predict_season(
    history: pl.DataFrame,
    target_rows: pl.DataFrame,
    decay: float,
) -> pl.DataFrame:
    """Predict a season using only rows from strictly earlier seasons."""
    if history.height and history.get_column("season").max() >= (
        target_rows.get_column("season").min()
    ):
        raise ValueError("Leakage detected: history includes target-season data")

    # Give more recent seasons more weight. Each season contributes its
    # mean PPR per recorded player-game, rather than total season points.
    seasonal = (
        history.group_by(["player_id", "season"])
        .agg(pl.col(TARGET).mean().alias("season_mean"))
        .with_columns(
            (decay ** (
                target_rows.get_column("season").min()
                - pl.col("season") - 1
            )).alias("weight")
        )
        .group_by("player_id")
        .agg(
            (
                (pl.col("season_mean") * pl.col("weight")).sum()
                / pl.col("weight").sum()
            ).alias("player_prediction")
        )
    )

    # Fallbacks use historical data only.
    position_means = history.group_by("position").agg(
        pl.col(TARGET).mean().alias("position_prediction")
    )
    overall_mean = (
        history.get_column(TARGET).mean()
        if history.height
        else 0.0
    )

    return (
        target_rows
        .join(seasonal, on="player_id", how="left")
        .join(position_means, on="position", how="left")
        .with_columns(
            pl.coalesce([
                pl.col("player_prediction"),
                pl.col("position_prediction"),
                pl.lit(overall_mean),
            ]).alias("prediction")
        )
    )


def evaluate(
    history: pl.DataFrame,
    target_rows: pl.DataFrame,
    decay: float,
    label: str,
) -> dict:
    scored = predict_season(history, target_rows, decay)
    actual = scored.get_column(TARGET).to_numpy()
    predicted = scored.get_column("prediction").to_numpy()

    return {
        "period": label,
        "decay": decay,
        "rows": len(actual),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "actual_mean": float(actual.mean()),
        "prediction_mean": float(predicted.mean()),
    }


def evaluate_position_baseline(
    history: pl.DataFrame,
    target_rows: pl.DataFrame,
    label: str,
) -> dict:
    """Predict using only historical position averages."""
    position_means = history.group_by("position").agg(
        pl.col(TARGET).mean().alias("prediction")
    )
    overall_mean = history.get_column(TARGET).mean()

    scored = (
        target_rows.join(position_means, on="position", how="left")
        .with_columns(
            pl.col("prediction").fill_null(overall_mean)
        )
    )

    actual = scored.get_column(TARGET).to_numpy()
    predicted = scored.get_column("prediction").to_numpy()

    return {
        "period": label,
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
    }



def main() -> None:
    df = (
        pl.read_parquet(DATA_PATH)
        .filter(pl.col("season_type") == "REG")
        .filter(pl.col("player_id").is_not_null())
        .filter(pl.col(TARGET).is_not_null())
        .sort(["season", "week", "game_id", "player_id"])
    )

    # Tune on 2023 only. Each validation prediction sees seasons through 2022.
    train_history = df.filter(pl.col("season") < VALIDATION_SEASON)
    validation = df.filter(pl.col("season") == VALIDATION_SEASON)

    validation_results = [
        evaluate(train_history, validation, decay, "validation_2023")
        for decay in DECAYS
    ]
    results = pl.DataFrame(validation_results).sort("mae")

    print("\n=== VALIDATION: 2023 (DECAY SELECTION) ===")
    print(results)

    best_decay = results.row(0, named=True)["decay"]
    print(f"\nSelected decay: {best_decay}")

    # Sequential test: each target season sees only earlier seasons.
    # In particular, 2025 predictions may use 2024 results because those
    # results would be known before the 2025 season.
    test_results = []
    for season in TEST_SEASONS:
        history = df.filter(pl.col("season") < season)
        target_rows = df.filter(pl.col("season") == season)
        test_results.append(
            evaluate(history, target_rows, best_decay, f"test_{season}")
        )

    print("\n=== TEST: 2024–2025 (NO DECAY RETUNING) ===")
    print(pl.DataFrame(test_results))


    print("\n=== POSITION-AVERAGE BENCHMARK ===")
    for season in (2023, 2024, 2025):
        history = df.filter(pl.col("season") < season)
        target_rows = df.filter(pl.col("season") == season)
        result = evaluate_position_baseline(
            history, target_rows, f"season_{season}"
        )
        print(result)



if __name__ == "__main__":
    main()
