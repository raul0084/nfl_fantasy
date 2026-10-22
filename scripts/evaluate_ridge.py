"""Evaluate Ridge projections using chronological NFL-season splits."""

from pathlib import Path

import polars as pl

from fantasy_ml.data.training import TARGET
from fantasy_ml.models.ridge import (
    DEFAULT_ALPHAS,
    fit_ridge,
    predict_ridge,
    regression_metrics,
)

DATA_PATH = Path("data/processed/training_player_games_2018_2025.parquet")
VALIDATION_SEASON = 2023
TEST_SEASONS = (2024, 2025)


def evaluate_season(
    history: pl.DataFrame,
    target_rows: pl.DataFrame,
    alpha: float,
    label: str,
) -> dict:
    """Fit on earlier seasons and evaluate one later season."""
    if history.is_empty():
        raise ValueError("Training history must not be empty")
    if target_rows.is_empty():
        raise ValueError(f"No target rows available for {label}")
    if history.get_column("season").max() >= target_rows.get_column("season").min():
        raise ValueError("Leakage detected: training includes target-season data")

    model = fit_ridge(history, alpha=alpha)
    predictions = predict_ridge(model, target_rows)
    actual = target_rows.get_column(TARGET).to_numpy()

    return {
        "period": label,
        "alpha": alpha,
        "rows": len(actual),
        **regression_metrics(actual, predictions),
        "actual_mean": float(actual.mean()),
        "prediction_mean": float(predictions.mean()),
    }


def main() -> None:
    df = (
        pl.read_parquet(DATA_PATH)
        .filter(pl.col("season_type") == "REG")
        .filter(pl.col("player_id").is_not_null())
        .filter(pl.col(TARGET).is_not_null())
        .sort(["season", "week", "game_id", "player_id"])
    )

    train_history = df.filter(pl.col("season") < VALIDATION_SEASON)
    validation = df.filter(pl.col("season") == VALIDATION_SEASON)

    # Select alpha using validation only; do not tune on test seasons.
    validation_results = [
        evaluate_season(
            train_history,
            validation,
            alpha,
            "validation_2023",
        )
        for alpha in DEFAULT_ALPHAS
    ]
    validation_table = pl.DataFrame(validation_results).sort("mae")

    print("\n=== RIDGE VALIDATION: 2023 ===")
    print(validation_table)

    best_alpha = validation_table.row(0, named=True)["alpha"]
    print(f"\nSelected alpha: {best_alpha}")

    # Each test season can use only earlier seasons.
    test_results = []
    for season in TEST_SEASONS:
        history = df.filter(pl.col("season") < season)
        target_rows = df.filter(pl.col("season") == season)
        test_results.append(
            evaluate_season(
                history,
                target_rows,
                best_alpha,
                f"test_{season}",
            )
        )

    print("\n=== RIDGE TEST: 2024–2025 ===")
    print(pl.DataFrame(test_results))


if __name__ == "__main__":
    main()
