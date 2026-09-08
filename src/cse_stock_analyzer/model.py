"""Training, persistence, and inference for the cross-sectional regressor."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import load_history
from .features import FEATURE_COLUMNS, prepare_features

MAX_ABS_DAILY_RETURN = 0.40
TRAIN_SPLIT = 0.80


def prepare_training_frame(frame: pd.DataFrame, train_end: str | None = None) -> pd.DataFrame:
    """Create features and next-session targets with contaminated jumps removed."""
    data = frame.copy()
    if train_end:
        if "Date" not in data:
            raise ValueError("Training data must include a Date column")
        data = data[pd.to_datetime(data["Date"]) <= pd.Timestamp(train_end)].copy()
    data = prepare_features(data)
    data["Target_Return"] = data["Current_Return"].shift(-1)

    contaminated = data["Current_Return"].abs() > MAX_ABS_DAILY_RETURN
    for lag in ("Return_Lag1", "Return_Lag2"):
        contaminated |= data[lag].abs() > MAX_ABS_DAILY_RETURN
    contaminated |= contaminated.shift(-1, fill_value=False)
    data = data.loc[~contaminated]
    return data.dropna(subset=FEATURE_COLUMNS + ["Target_Return"])


def train(
    csv_paths: list[Path],
    model_path: Path,
    metadata_path: Path,
    train_end: str | None = None,
) -> dict[str, Any]:
    """Train using an independent chronological split for each security."""
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    symbols: list[str] = []

    for path in sorted(csv_paths):
        frame = prepare_training_frame(load_history(path), train_end)
        if len(frame) < 50:
            continue
        split = int(len(frame) * TRAIN_SPLIT)
        train_frames.append(frame.iloc[:split])
        test_frames.append(frame.iloc[split:])
        symbols.append(path.stem)

    if not train_frames:
        raise ValueError("No CSV contains enough usable observations for training")

    training = pd.concat(train_frames, ignore_index=True)
    testing = pd.concat(test_frames, ignore_index=True)
    model = xgb.XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="mae", n_jobs=-1,
    )
    model.fit(training[FEATURE_COLUMNS], training["Target_Return"])
    predictions = model.predict(testing[FEATURE_COLUMNS])
    actual = testing["Target_Return"]
    metrics = {
        "mae": float(mean_absolute_error(actual, predictions)),
        "mse": float(mean_squared_error(actual, predictions)),
        "r2": float(r2_score(actual, predictions)),
        "directional_accuracy": float(((predictions >= 0) == (actual.to_numpy() >= 0)).mean()),
        "zero_return_baseline_mae": float(actual.abs().mean()),
    }
    metadata: dict[str, Any] = {
        "model_type": "XGBoost next-day return regressor",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_end": train_end,
        "features": FEATURE_COLUMNS,
        "training_rows": len(training),
        "test_rows": len(testing),
        "assets": symbols,
        "metrics": metrics,
        "hyperparameters": model.get_params(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata


def load_model(model_path: Path) -> xgb.XGBRegressor:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run the train command first.")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    saved_features = model.get_booster().feature_names
    if saved_features and saved_features != FEATURE_COLUMNS:
        raise ValueError("Model feature schema does not match the application feature schema")
    return model


def predict_latest(model: xgb.XGBRegressor, frame: pd.DataFrame, symbol: str) -> dict[str, Any]:
    featured = prepare_features(frame).dropna(subset=FEATURE_COLUMNS)
    if featured.empty:
        raise ValueError(f"Not enough usable history to calculate features for {symbol}")
    latest = featured.iloc[-1]
    predicted_return = float(model.predict(featured.iloc[[-1]][FEATURE_COLUMNS])[0])
    date = latest.get("Date", featured.index[-1])
    return {
        "symbol": symbol,
        "observation_date": pd.Timestamp(date).date().isoformat(),
        "predicted_next_day_return": predicted_return,
        "predicted_next_day_percent": predicted_return * 100,
        "direction": "UP" if predicted_return >= 0 else "DOWN",
    }
