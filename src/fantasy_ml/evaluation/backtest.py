"""Chronological evaluation for weekly fantasy-point projections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fantasy_ml.data.training import TARGET
from fantasy_ml.models.lightgbm_model import (
    fit_lightgbm,
    predict_lightgbm,
)
from fantasy_ml.models.ridge import fit_ridge, predict_ridge


BASELINE_FEATURES = (
    "model_fantasy_points_avg_prev_3_games",
    "model_fantasy_points_avg_prev_5_games",
)


def chronological_split(
    dataset: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return train (<=2022), validation (2023), and test (>=2024)."""
    required = {"season", "position", TARGET}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    train = dataset.filter(pl.col("season") <= 2022)
    validation = dataset.filter(pl.col("season") == 2023)
    test = dataset.filter(pl.col("season").is_in([2024, 2025]))

    if train.is_empty() or validation.is_empty() or test.is_empty():
        raise ValueError(
            "Chronological split requires rows in train (<=2022), "
            "validation (2023), and test (2024-2025)."
        )

    return train, validation, test


def _baseline_predictions(
    training_rows: pl.DataFrame,
    target_rows: pl.DataFrame,
) -> np.ndarray:
    """Use prior 3-game average, then 5-game average, then train mean."""
    available = [
        column for column in BASELINE_FEATURES
        if column in target_rows.columns
    ]
    if not available:
        raise ValueError("Dataset has no prior-game baseline features")

    fallback = float(training_rows[TARGET].mean())
    prediction = np.full(target_rows.height, fallback, dtype=float)

    for column in reversed(available):
        values = target_rows[column].cast(pl.Float64).to_numpy()
        prediction = np.where(np.isfinite(values), values, prediction)

    return prediction


def _metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    rho = spearmanr(actual, predicted).statistic
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "spearman": float(rho) if np.isfinite(rho) else float("nan"),
        "n": int(len(actual)),
    }


def _evaluate_predictions(
    rows: pl.DataFrame,
    predictions: np.ndarray,
    model_name: str,
    split_name: str,
) -> list[dict]:
    actual = rows[TARGET].to_numpy()
    records = []

    for position in ["ALL", *sorted(rows["position"].drop_nulls().unique().to_list())]:
        mask = (
            np.ones(rows.height, dtype=bool)
            if position == "ALL"
            else (rows["position"].to_numpy() == position)
        )
        if mask.sum() == 0:
            continue

        records.append({
            "split": split_name,
            "model": model_name,
            "position": position,
            **_metrics(actual[mask], predictions[mask]),
        })

    return records



def run_backtest(
    dataset: pl.DataFrame,
    *,
    lightgbm_estimators: int = 400,
) -> pl.DataFrame:
    """Evaluate 2023 validation and 2024-2025 test, including per-season metrics."""
    train, validation, test = chronological_split(dataset)
    records: list[dict] = []

    # Validation: fit exclusively on 2018-2022.
    ridge = fit_ridge(train)
    lgbm = fit_lightgbm(train, n_estimators=lightgbm_estimators)

    records.extend(_evaluate_predictions(
        validation,
        _baseline_predictions(train, validation),
        "rolling_baseline",
        "validation",
    ))
    records.extend(_evaluate_predictions(
        validation,
        predict_ridge(ridge, validation),
        "ridge",
        "validation",
    ))
    records.extend(_evaluate_predictions(
        validation,
        predict_lightgbm(lgbm, validation),
        "lightgbm",
        "validation",
    ))

    # Test models use only data available through the 2023 season.
    final_train = pl.concat([train, validation], how="vertical")
    ridge_final = fit_ridge(final_train)
    lgbm_final = fit_lightgbm(
        final_train, n_estimators=lightgbm_estimators
    )

    # Preserve the combined 2024-2025 test metrics.
    test_predictions = {
        "rolling_baseline": _baseline_predictions(final_train, test),
        "ridge": predict_ridge(ridge_final, test),
        "lightgbm": predict_lightgbm(lgbm_final, test),
    }

    for model_name, predictions in test_predictions.items():
        records.extend(_evaluate_predictions(
            test, predictions, model_name, "test"
        ))

        # Additional diagnostics by individual test season.
        for season in sorted(test["season"].unique().to_list()):
            mask = (test["season"].to_numpy() == season)
            season_rows = test.filter(pl.col("season") == season)

            records.extend(_evaluate_predictions(
                season_rows,
                predictions[mask],
                model_name,
                f"test_{season}",
            ))

    return pl.DataFrame(records).sort(
        ["split", "position", "model"]
    )



def save_metrics(metrics: pl.DataFrame, path: str | Path) -> None:
    """Persist metrics as CSV, creating the parent directory if needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_csv(output)
