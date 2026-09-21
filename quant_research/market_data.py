"""Market-data loading and OHLCV schema normalization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a data frame to a sorted, unique OHLCV time series."""
    if data.empty:
        raise ValueError("Market data is empty.")

    normalized = data.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    aliases = {str(column).strip().lower(): column for column in normalized.columns}
    rename = {}
    for expected in REQUIRED_COLUMNS:
        source = aliases.get(expected.lower())
        if source is None:
            raise ValueError(f"Market data is missing required column: {expected}")
        rename[source] = expected

    normalized = normalized.rename(columns=rename).loc[:, list(REQUIRED_COLUMNS)]
    normalized.index = pd.to_datetime(normalized.index, utc=True).tz_convert(None)
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    normalized = normalized.apply(pd.to_numeric, errors="coerce").dropna()
    if normalized.empty:
        raise ValueError("Market data contains no valid OHLCV rows after normalization.")
    if (normalized[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")
    return normalized


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from CSV; the first column is treated as the date index."""
    path = Path(path)
    first_line = path.read_text(encoding="utf-8-sig").splitlines()[0]
    if first_line.startswith("Price,Adj Close,"):
        data = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
        ratio = data["Adj Close"] / data["Close"]
        for column in ("Open", "High", "Low", "Close"):
            data[column] = data[column] * ratio
    else:
        data = pd.read_csv(path, index_col=0, parse_dates=True)
    return normalize_ohlcv(data)


def load_market_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str | None = "10y",
) -> pd.DataFrame:
    """Download adjusted daily OHLCV data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for downloads. Run: pip install -r requirements.txt"
        ) from exc

    cache_directory = Path(tempfile.gettempdir()) / "quant_research_yfinance"
    cache_directory.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_directory))

    kwargs: dict[str, object] = {
        "tickers": ticker,
        "auto_adjust": True,
        "progress": False,
        "threads": False,
    }
    if start or end:
        kwargs.update(start=start, end=end)
    else:
        kwargs["period"] = period or "10y"
    data = yf.download(**kwargs)
    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker!r}.")
    return normalize_ohlcv(data)
