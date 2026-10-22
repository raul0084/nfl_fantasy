
"""LightGBM fantasy-point regression with training-only categories."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from fantasy_ml.data.training import split_features_target

CATEGORICAL_FEATURES = ("position", "team", "opponent_team")


@dataclass
class LightGBMModel:
    estimator: lgb.LGBMRegressor
    feature_columns: list[str]
    categorical_columns: list[str]
    categories: dict[str, pd.Index]


def fit_lightgbm(
    training_rows: pl.DataFrame,
    *,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
) -> LightGBMModel:
    """Fit LightGBM; categorical levels are learned from training rows only."""
    if training_rows.is_empty():
        raise ValueError("training_rows must not be empty")

    X, y, _ = split_features_target(training_rows)
    X_pd = X.to_pandas()

    categorical_columns = [
        col for col in CATEGORICAL_FEATURES if col in X_pd.columns
    ]
    categories: dict[str, pd.Index] = {}

    for col in categorical_columns:
        X_pd[col] = X_pd[col].astype("category")
        categories[col] = X_pd[col].cat.categories

    estimator = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    estimator.fit(X_pd, y.to_numpy())

    return LightGBMModel(
        estimator=estimator,
        feature_columns=list(X_pd.columns),
        categorical_columns=categorical_columns,
        categories=categories,
    )


def predict_lightgbm(
    model: LightGBMModel,
    target_rows: pl.DataFrame,
) -> np.ndarray:
    """Predict with the training model's exact feature and category schema."""
    X, _, _ = split_features_target(target_rows)
    X_pd = X.to_pandas()

    missing = set(model.feature_columns) - set(X_pd.columns)
    if missing:
        raise ValueError(f"Missing prediction features: {sorted(missing)}")

    X_pd = X_pd[model.feature_columns]

    for col in model.categorical_columns:
        X_pd[col] = pd.Categorical(
            X_pd[col],
            categories=model.categories[col],
        )

    return model.estimator.predict(X_pd)
