"""Feature-table construction shared by every strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import historical_volatility, macd, moving_average, rsi, volume_ratio
from .market_data import normalize_ohlcv


FEATURE_COLUMNS = (
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "ma50",
    "ma200",
    "distance_ma50",
    "distance_ma200",
    "volume_ratio",
    "historical_volatility",
    "return_1d",
    "return_5d",
)


def build_feature_table(data: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """Compute a reusable, point-in-time feature table from OHLCV data."""
    features = normalize_ohlcv(data)
    features["rsi"] = rsi(features["Close"])
    features = features.join(macd(features["Close"]))
    features["ma50"] = moving_average(features["Close"], 50)
    features["ma200"] = moving_average(features["Close"], 200)
    features["distance_ma50"] = features["Close"] / features["ma50"] - 1.0
    features["distance_ma200"] = features["Close"] / features["ma200"] - 1.0
    features["volume_ratio"] = volume_ratio(features["Volume"])
    features["historical_volatility"] = historical_volatility(features["Close"], window=30)
    features["return_1d"] = features["Close"].pct_change()
    features["return_5d"] = features["Close"].pct_change(5)
    features = features.replace([np.inf, -np.inf], np.nan)
    if drop_warmup:
        features = features.dropna(subset=list(FEATURE_COLUMNS))
    if features.empty:
        raise ValueError(
            "Not enough history to compute features. At least 200 daily candles are required."
        )
    return features
