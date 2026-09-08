import numpy as np
import pandas as pd
import xgboost as xgb

from cse_stock_analyzer.features import FEATURE_COLUMNS, prepare_features
from cse_stock_analyzer.model import predict_latest


def test_prediction_record_shape():
    rows = 80
    close = 100 + np.sin(np.arange(rows) / 4) + np.arange(rows) * 0.05
    frame = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=rows),
        "Open": close * 0.995, "High": close * 1.01,
        "Low": close * 0.99, "Close": close, "Volume": 10_000 + np.arange(rows),
    })
    featured = prepare_features(frame).dropna(subset=FEATURE_COLUMNS)
    model = xgb.XGBRegressor(n_estimators=2, max_depth=1, random_state=42)
    model.fit(featured[FEATURE_COLUMNS], np.zeros(len(featured)))
    result = predict_latest(model, frame, "JKH.N0000")
    assert result["symbol"] == "JKH.N0000"
    assert result["direction"] in {"UP", "DOWN"}
    assert isinstance(result["predicted_next_day_percent"], float)
