"""Performance and trade statistics for comparable experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR


ROUND_TRIP_COLUMNS = (
    "entry_date",
    "exit_date",
    "holding_days",
    "buy_executions",
    "sell_executions",
    "quantity",
    "gross_buy_cost",
    "net_sell_proceeds",
    "fees",
    "pnl",
    "return",
)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def build_round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """Combine partial rebalances into completed flat-to-flat position episodes."""
    if trades.empty:
        return pd.DataFrame(columns=ROUND_TRIP_COLUMNS)

    required = {"date", "price", "action", "quantity", "fees"}
    missing = sorted(required.difference(trades.columns))
    if missing:
        raise ValueError(f"Trade log is missing columns: {', '.join(missing)}")

    completed: list[dict[str, Any]] = []
    position = 0.0
    episode: dict[str, Any] | None = None
    tolerance = 1e-8

    for row in trades.sort_values("date").itertuples(index=False):
        quantity = float(row.quantity)
        price = float(row.price)
        fees = float(row.fees)
        date = pd.Timestamp(row.date)

        if row.action == "BUY":
            if position <= tolerance:
                episode = {
                    "entry_date": date,
                    "buy_executions": 0,
                    "sell_executions": 0,
                    "quantity": 0.0,
                    "gross_buy_cost": 0.0,
                    "net_sell_proceeds": 0.0,
                    "fees": 0.0,
                }
            if episode is None:
                raise ValueError("BUY execution could not be assigned to a position episode.")
            position += quantity
            episode["buy_executions"] += 1
            episode["quantity"] += quantity
            episode["gross_buy_cost"] += quantity * price + fees
            episode["fees"] += fees
            continue

        if row.action != "SELL":
            raise ValueError(f"Unknown trade action: {row.action!r}")
        if episode is None or position <= tolerance:
            raise ValueError("SELL execution appears before an open long position.")

        position -= quantity
        episode["sell_executions"] += 1
        episode["net_sell_proceeds"] += quantity * price - fees
        episode["fees"] += fees
        if position > tolerance:
            continue

        pnl = episode["net_sell_proceeds"] - episode["gross_buy_cost"]
        cost = episode["gross_buy_cost"]
        completed.append(
            {
                "entry_date": episode["entry_date"],
                "exit_date": date,
                "holding_days": float((date - episode["entry_date"]).days),
                "buy_executions": int(episode["buy_executions"]),
                "sell_executions": int(episode["sell_executions"]),
                "quantity": float(episode["quantity"]),
                "gross_buy_cost": float(cost),
                "net_sell_proceeds": float(episode["net_sell_proceeds"]),
                "fees": float(episode["fees"]),
                "pnl": float(pnl),
                "return": float(pnl / cost) if cost else np.nan,
            }
        )
        position = 0.0
        episode = None

    return pd.DataFrame(completed, columns=ROUND_TRIP_COLUMNS)


def calculate_metrics(
    history: pd.DataFrame,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Calculate portfolio metrics and completed flat-to-flat trade statistics."""
    if history.empty:
        raise ValueError("Portfolio history is empty.")

    values = history["portfolio_value"].astype(float)
    returns = values.pct_change().dropna()
    total_return = values.iloc[-1] / values.iloc[0] - 1.0
    years = max(len(returns) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0
    volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    excess_annual = returns.mean() * TRADING_DAYS_PER_YEAR - risk_free_rate
    sharpe = excess_annual / volatility if volatility and not np.isnan(volatility) else np.nan

    downside = returns[returns < 0.0]
    downside_deviation = downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = (
        excess_annual / downside_deviation
        if downside_deviation and not np.isnan(downside_deviation)
        else np.nan
    )
    running_max = values.cummax()
    drawdown = values / running_max - 1.0

    round_trips = build_round_trips(trades)
    pnl = (
        round_trips["pnl"].astype(float)
        if not round_trips.empty
        else pd.Series(dtype=float)
    )
    gross_profit = pnl[pnl > 0.0].sum()
    gross_loss = -pnl[pnl < 0.0].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    win_rate = float((pnl > 0.0).mean()) if not pnl.empty else np.nan
    average_holding = (
        round_trips["holding_days"].astype(float).mean()
        if not round_trips.empty
        else np.nan
    )
    total_fees = float(trades["fees"].sum()) if not trades.empty else 0.0
    traded_notional = (
        float((trades["quantity"].astype(float) * trades["price"].astype(float)).sum())
        if not trades.empty
        else 0.0
    )
    average_value = float(values.mean())
    years = max(len(returns) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)
    annual_turnover = traded_notional / average_value / years if average_value else np.nan

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": _finite_or_none(float(volatility)),
        "sharpe_ratio": _finite_or_none(float(sharpe)),
        "sortino_ratio": _finite_or_none(float(sortino)),
        "maximum_drawdown": float(drawdown.min()),
        "profit_factor": _finite_or_none(float(profit_factor)),
        "win_rate": _finite_or_none(float(win_rate)),
        "average_trade_pnl": _finite_or_none(float(pnl.mean())) if not pnl.empty else None,
        "average_holding_days": _finite_or_none(float(average_holding)),
        "trade_count": int(len(round_trips)),
        "execution_count": int(len(trades)),
        "total_fees": total_fees,
        "annual_turnover": _finite_or_none(float(annual_turnover)),
        "ending_portfolio_value": float(values.iloc[-1]),
    }
