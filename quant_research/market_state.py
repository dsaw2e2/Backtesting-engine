"""Immutable strategy input constructed from one historical candle."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketState:
    """All market information visible to a strategy at one timestamp."""

    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    ma50: float
    ma200: float
    distance_ma50: float
    distance_ma200: float
    volume_ratio: float
    historical_volatility: float
    return_1d: float
    return_5d: float

    @classmethod
    def from_row(cls, date: pd.Timestamp, row: pd.Series) -> "MarketState":
        """Build state from one row of the feature table."""
        return cls(
            date=pd.Timestamp(date),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
            rsi=float(row["rsi"]),
            macd=float(row["macd"]),
            macd_signal=float(row["macd_signal"]),
            macd_hist=float(row["macd_hist"]),
            ma50=float(row["ma50"]),
            ma200=float(row["ma200"]),
            distance_ma50=float(row["distance_ma50"]),
            distance_ma200=float(row["distance_ma200"]),
            volume_ratio=float(row["volume_ratio"]),
            historical_volatility=float(row["historical_volatility"]),
            return_1d=float(row["return_1d"]),
            return_5d=float(row["return_5d"]),
        )
