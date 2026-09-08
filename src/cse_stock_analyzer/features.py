"""Shared feature engineering for model training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "Open_Pct", "High_Pct", "Low_Pct", "Rel_Volume", "RSI_3",
    "Return_Lag1", "Return_Lag2", "SMA_3_Pct", "EMA_3_Pct",
    "MACD_Pct", "MACD_Hist_Pct",
]
REQUIRED_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def validate_ohlcv(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_OHLCV if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Price data is empty")


def drop_non_trading_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Yahoo's zero-volume, forward-filled rows."""
    validate_ohlcv(df)
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    return df.loc[volume > 0].copy()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact momentum and trend indicators without future information."""
    validate_ohlcv(df)
    result = df.copy()
    close = pd.to_numeric(result["Close"], errors="coerce")
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(3, min_periods=3).mean()
    losses = -delta.clip(upper=0).rolling(3, min_periods=3).mean()
    rs = gains / losses.replace(0, np.nan)
    result["RSI_3"] = (100 - (100 / (1 + rs))).where(losses.ne(0), 100.0)
    result.loc[gains.eq(0) & losses.eq(0), "RSI_3"] = 50.0
    result["SMA_3"] = close.rolling(3, min_periods=3).mean()
    result["EMA_3"] = close.ewm(span=3, adjust=False, min_periods=3).mean()
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["MACD"] = ema_12 - ema_26
    result["MACD_Signal"] = result["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    result["MACD_Hist"] = result["MACD"] - result["MACD_Signal"]
    return result


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the exact scale-independent feature vector expected by the model."""
    result = df.copy()
    close = pd.to_numeric(result["Close"], errors="coerce").replace(0, np.nan)
    result["Open_Pct"] = result["Open"] / close - 1
    result["High_Pct"] = result["High"] / close - 1
    result["Low_Pct"] = result["Low"] / close - 1
    volume_mean = result["Volume"].rolling(20, min_periods=5).mean().replace(0, np.nan)
    result["Rel_Volume"] = result["Volume"] / volume_mean
    result["Current_Return"] = close.pct_change(fill_method=None)
    result["Return_Lag1"] = result["Current_Return"].shift(1)
    result["Return_Lag2"] = result["Current_Return"].shift(2)
    result["SMA_3_Pct"] = result["SMA_3"] / close - 1
    result["EMA_3_Pct"] = result["EMA_3"] / close - 1
    result["MACD_Pct"] = result["MACD"] / close
    result["MACD_Hist_Pct"] = result["MACD_Hist"] / close
    return result


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the complete, shared preprocessing pipeline."""
    clean = drop_non_trading_rows(df).reset_index(drop=True)
    return add_model_features(add_indicators(clean))
