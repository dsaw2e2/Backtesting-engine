"""Pure numerical indicator functions with no trading decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR


def moving_average(values: pd.Series, window: int) -> pd.Series:
    """Return a simple moving average."""
    return values.astype(float).rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Return Wilder's Relative Strength Index on a 0-100 scale."""
    delta = close.astype(float).diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + relative_strength))
    result = result.where(average_loss != 0.0, 100.0)
    return result.where(average_gain != 0.0, 0.0)


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Return MACD, signal line, and histogram."""
    values = close.astype(float)
    fast = values.ewm(span=fast_period, adjust=False).mean()
    slow = values.ewm(span=slow_period, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=signal_period, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "macd_signal": signal, "macd_hist": line - signal},
        index=close.index,
    )


def historical_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Return annualized rolling volatility of log returns."""
    log_returns = np.log(close.astype(float) / close.astype(float).shift(1))
    return log_returns.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Return current volume divided by its trailing simple average."""
    baseline = volume.astype(float).rolling(window=window, min_periods=window).mean()
    return volume.astype(float) / baseline.replace(0.0, np.nan)

