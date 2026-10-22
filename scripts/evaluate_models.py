
"""Chronological comparison of rolling baseline, Ridge, and LightGBM."""

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fantasy_ml.data.training import TARGET
from fantasy_ml.models.ridge import (
    DEFAULT_ALPHAS,
    fit_ridge,
    predict_ridge,
)
from fantasy_ml.models.lightgbm_model import (
    fit_lightgbm,
    predict_lightgbm,
)

DATA_PATH = Path("data/processed/training_player_games_2018_2025.parquet")
VALIDATION_SEASON = 2023
TEST_SEASONS = (2024, 2025)
BASELINE_FEATURE = f"{TARGET}_avg_prev_3_games"

# Standard fantasy positions; retain DEF/DST if present in the data.
STANDARD_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF", "DST")

LIGHTGBM_CONFIGS = [
    {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15},
    {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
    {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 31},
    {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 15},
]


def score_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Compute regression and ranking metrics without constant-input warnings."""
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError("No valid predictions to score")

    if np.unique(actual).size < 2 or np.unique(predicted).size < 2:
        correlation = float("nan")
    else:
        correlation = float(spearmanr(actual, predicted).statistic)

    return {
        "rows": len(actual),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "spearman": correlation,
        "actual_mean": float(actual.mean()),
        "prediction_mean": float(predicted.mean()),
    }


def predict_all(
    history: pl.DataFrame,
    targets: pl.DataFrame,
    ridge_alpha: float,
    lgb_config: dict,
) -> dict[str, np.ndarray]:
    """Fit models on history and predict the same target rows."""
    if history.is_empty() or targets.is_empty():
        raise ValueError("History and target rows must not be empty")

    if history["season"].max() >= targets["season"].min():
        raise ValueError("Leakage detected: history overlaps target season")

    ridge = fit_ridge(history, alpha=ridge_alpha)
    lightgbm = fit_lightgbm(history, **lgb_config)

    return {
        "rolling_avg_prev_3": (
            targets[BASELINE_FEATURE].fill_null(0.0).to_numpy()
        ),
        "ridge": predict_ridge(ridge, targets),
        "lightgbm": predict_lightgbm(lightgbm, targets),
    }


def main() -> None:
    df = (
        pl.read_parquet(DATA_PATH)
        .filter(pl.col("season_type") == "REG")
        .filter(pl.col("player_id").is_not_null())
        .filter(pl.col(TARGET).is_not_null())
        .filter(pl.col("position").is_in(STANDARD_POSITIONS))
        .sort(["season", "week", "game_id", "player_id"])
    )

    if BASELINE_FEATURE not in df.columns:
        raise ValueError(
            f"Missing {BASELINE_FEATURE}; rebuild the training dataset first"
        )

    print("=== EVALUATION DATA ===")
    print(f"Rows: {df.height}")
    print(df.group_by("position").len().sort("position"))

    train = df.filter(pl.col("season") < VALIDATION_SEASON)
    validation = df.filter(pl.col("season") == VALIDATION_SEASON)

    if train.is_empty() or validation.is_empty():
        raise ValueError("Training or validation data is empty")

    # Select Ridge alpha on validation 2023 only.
    ridge_candidates = []
    for alpha in DEFAULT_ALPHAS:
        model = fit_ridge(train, alpha=alpha)
        pred = predict_ridge(model, validation)
        ridge_candidates.append({
            "alpha": alpha,
            **score_predictions(validation[TARGET].to_numpy(), pred),
        })

    ridge_table = pl.DataFrame(ridge_candidates).sort("mae")
    best_alpha = ridge_table.row(0, named=True)["alpha"]

    print("\n=== RIDGE VALIDATION SEARCH: 2023 ===")
    print(ridge_table)
    print(f"Selected Ridge alpha: {best_alpha}")

    # Select LightGBM configuration on validation 2023 only.
    lgb_candidates = []
    for config in LIGHTGBM_CONFIGS:
        model = fit_lightgbm(train, **config)
        pred = predict_lightgbm(model, validation)
        lgb_candidates.append({
            **config,
            **score_predictions(validation[TARGET].to_numpy(), pred),
        })

    lgb_table = pl.DataFrame(lgb_candidates).sort("mae")
    best_lgb_config = {
        key: lgb_table.row(0, named=True)[key]
        for key in ("n_estimators", "learning_rate", "num_leaves")
    }

    print("\n=== LIGHTGBM VALIDATION SEARCH: 2023 ===")
    print(lgb_table)
    print(f"Selected LightGBM config: {best_lgb_config}")

    # Validation report and sequential held-out seasons.
    periods = [("validation_2023", train, validation)]

    for season in TEST_SEASONS:
        history = df.filter(pl.col("season") < season)
        targets = df.filter(pl.col("season") == season)
        periods.append((f"test_{season}", history, targets))

    overall_results = []
    position_results = []

    for label, history, targets in periods:
        predictions = predict_all(
            history, targets, best_alpha, best_lgb_config
        )
        actual = targets[TARGET].to_numpy()
        positions = targets["position"].to_numpy()

        for model_name, pred in predictions.items():
            overall_results.append({
                "period": label,
                "model": model_name,
                "ridge_alpha": best_alpha if model_name == "ridge" else None,
                **score_predictions(actual, pred),
            })

            for position in sorted(set(positions)):
                mask = positions == position
                position_results.append({
                    "period": label,
                    "position": position,
                    "model": model_name,
                    **score_predictions(actual[mask], pred[mask]),
                })

    print("\n=== OVERALL MODEL COMPARISON ===")
    print(pl.DataFrame(overall_results).sort(["period", "model"]))

    print("\n=== POSITION-LEVEL MODEL COMPARISON ===")
    print(
        pl.DataFrame(position_results)
        .filter(pl.col("period").str.starts_with("test_"))
        .sort(["period", "position", "model"])
    )


if __name__ == "__main__":
    main()
