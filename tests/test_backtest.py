import polars as pl
import pytest

from fantasy_ml.data.training import TARGET
from fantasy_ml.evaluation.backtest import chronological_split


def test_chronological_split_respects_seasons():
    df = pl.DataFrame({
        "season": [2018, 2022, 2023, 2024, 2025],
        "position": ["RB"] * 5,
        TARGET: [10.0, 12.0, 13.0, 14.0, 15.0],
    })

    train, validation, test = chronological_split(df)

    assert train["season"].to_list() == [2018, 2022]
    assert validation["season"].to_list() == [2023]
    assert test["season"].to_list() == [2024, 2025]


def test_chronological_split_requires_all_periods():
    df = pl.DataFrame({
        "season": [2020, 2021],
        "position": ["RB", "WR"],
        TARGET: [10.0, 12.0],
    })

    with pytest.raises(ValueError, match="requires rows"):
        chronological_split(df)
