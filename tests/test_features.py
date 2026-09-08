import numpy as np
import pandas as pd

from cse_stock_analyzer.features import FEATURE_COLUMNS, prepare_features
from cse_stock_analyzer.model import prepare_training_frame


def prices(rows=80):
    close = 100 + np.sin(np.arange(rows) / 3) + np.arange(rows) * 0.1
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=rows),
        "Open": close * 0.995,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": np.arange(rows) * 100 + 1_000,
    })


def test_features_have_expected_schema_and_finite_latest_row():
    result = prepare_features(prices())
    assert set(FEATURE_COLUMNS).issubset(result.columns)
    assert np.isfinite(result.iloc[-1][FEATURE_COLUMNS].astype(float)).all()


def test_target_is_the_next_return():
    result = prepare_training_frame(prices())
    first = result.iloc[0]
    source = prepare_features(prices())
    source_row = source.loc[first.name]
    expected = source.loc[first.name + 1, "Current_Return"]
    assert first["Target_Return"] == expected
    assert source_row["Current_Return"] != first["Target_Return"]


def test_zero_volume_rows_are_removed():
    frame = prices()
    frame.loc[10, "Volume"] = 0
    assert len(prepare_features(frame)) == len(frame) - 1
