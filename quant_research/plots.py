"""Publication-ready static charts for each backtest experiment."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quant_research_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR


def _setup() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 10})


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_charts(history: pd.DataFrame, trades: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate all required research charts and return their paths."""
    _setup()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = history.copy().set_index(pd.to_datetime(history["date"]))
    values = frame["portfolio_value"]
    returns = values.pct_change()
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    (values / values.iloc[0]).plot(ax=ax, color="#176B87", linewidth=1.7)
    ax.set(title="Equity Curve", ylabel="Growth of $1", xlabel="")
    paths.append(output_dir / "equity_curve.png")
    _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(10, 5))
    values.plot(ax=ax, color="#0B3C5D", linewidth=1.7)
    ax.set(title="Portfolio Value", ylabel="Portfolio value", xlabel="")
    paths.append(output_dir / "portfolio_value.png")
    _save(fig, paths[-1])

    drawdown = values / values.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#C0392B", alpha=0.6)
    ax.set(title="Drawdown", ylabel="Drawdown", xlabel="")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    paths.append(output_dir / "drawdown.png")
    _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(10, 5))
    frame["price"].plot(ax=ax, color="#333333", linewidth=1.2, label="Close")
    if not trades.empty:
        trade_frame = trades.copy()
        trade_frame["date"] = pd.to_datetime(trade_frame["date"])
        buys = trade_frame[trade_frame["action"] == "BUY"]
        sells = trade_frame[trade_frame["action"] == "SELL"]
        buy_dates = mdates.date2num(buys["date"].dt.to_pydatetime())
        sell_dates = mdates.date2num(sells["date"].dt.to_pydatetime())
        ax.scatter(buy_dates, buys["price"], marker="^", color="#148F77", s=38, label="Buy")
        ax.scatter(sell_dates, sells["price"], marker="v", color="#C0392B", s=38, label="Sell")
    ax.set(title="Price and Executions", ylabel="Price", xlabel="")
    ax.legend()
    paths.append(output_dir / "price_and_trades.png")
    _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(10, 4))
    frame["position_allocation"].clip(0, 1).plot(ax=ax, color="#7D3C98", linewidth=1.4)
    ax.set(title="Portfolio Allocation", ylabel="Invested", xlabel="", ylim=(-0.03, 1.03))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    paths.append(output_dir / "portfolio_allocation.png")
    _save(fig, paths[-1])

    rolling_returns = (1.0 + returns).rolling(21).apply(np.prod, raw=True) - 1.0
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling_returns.plot(ax=ax, color="#D35400", linewidth=1.2)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set(title="Rolling 21-Day Return", ylabel="Return", xlabel="")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    paths.append(output_dir / "rolling_returns.png")
    _save(fig, paths[-1])

    rolling_sharpe = (
        returns.rolling(63).mean() / returns.rolling(63).std(ddof=1)
    ) * np.sqrt(TRADING_DAYS_PER_YEAR)
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling_sharpe.plot(ax=ax, color="#2874A6", linewidth=1.2)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set(title="Rolling 63-Day Sharpe Ratio", ylabel="Sharpe ratio", xlabel="")
    paths.append(output_dir / "rolling_sharpe.png")
    _save(fig, paths[-1])
    return paths
