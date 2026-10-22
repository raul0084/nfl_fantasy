
"""Ridge regression for leakage-safe fantasy point projections."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fantasy_ml.data.training import TARGET, split_features_target

CATEGORICAL_FEATURES = ("position", "team", "opponent_team")
DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0)


def make_ridge_pipeline(
    dataset: pl.DataFrame,
    alpha: float = 1.0,
) -> Pipeline:
    """Create preprocessing and Ridge regression, fitted later on training data."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    X, _, _ = split_features_target(dataset)

    categorical = [
        column for column in CATEGORICAL_FEATURES if column in X.columns
    ]
    numeric = [column for column in X.columns if column not in categorical]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def fit_ridge(
    training_rows: pl.DataFrame,
    alpha: float = 1.0,
) -> Pipeline:
    """Fit Ridge using only the rows passed to this function."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if training_rows.is_empty():
        raise ValueError("training_rows must not be empty")

    X, y, _ = split_features_target(training_rows)
    pipeline = make_ridge_pipeline(training_rows, alpha=alpha)
    pipeline.fit(X.to_pandas(), y.to_numpy())
    return pipeline


def predict_ridge(
    model: Pipeline,
    target_rows: pl.DataFrame,
) -> np.ndarray:
    """Predict fantasy points for rows not used to fit the model."""
    X, _, _ = split_features_target(target_rows)
    return model.predict(X.to_pandas())


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate MAE and RMSE."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }
