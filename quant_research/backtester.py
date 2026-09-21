"""Point-in-time historical event loop with next-open execution."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DEFAULT_FEE_RATE, DEFAULT_INITIAL_CAPITAL, DEFAULT_SLIPPAGE_BPS
from .features import FEATURE_COLUMNS
from .market_state import MarketState
from .portfolio import Portfolio
from .signals import Action
from .strategies.base_strategy import BaseStrategy


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    fee_rate: float = DEFAULT_FEE_RATE
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    feature_data: pd.DataFrame
    trades: pd.DataFrame
    history: pd.DataFrame
    config: BacktestConfig


class Backtester:
    """Replay complete feature rows and execute each decision one candle later."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(self, feature_data: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        required = {"Open", "Close", *FEATURE_COLUMNS}
        missing = sorted(required.difference(feature_data.columns))
        if missing:
            raise ValueError(f"Feature data is missing columns: {', '.join(missing)}")

        data = feature_data.dropna(subset=list(required)).sort_index()
        if len(data) < 2:
            raise ValueError("A backtest needs at least two complete feature rows.")

        portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            fee_rate=self.config.fee_rate,
            slippage_bps=self.config.slippage_bps,
        )
        portfolio.record(data.index[0], float(data.iloc[0]["Close"]))
        active_target = 0.0

        for index in range(len(data) - 1):
            signal_date = data.index[index]
            next_date = data.index[index + 1]
            state = MarketState.from_row(signal_date, data.iloc[index])
            decision = strategy.predict(state)
            target_changed = abs(decision.position_size - active_target) > 1e-12
            if decision.action != Action.HOLD and target_changed:
                portfolio.rebalance(next_date, float(data.iloc[index + 1]["Open"]), decision)
                active_target = decision.position_size
            portfolio.record(next_date, float(data.iloc[index + 1]["Close"]))

        return BacktestResult(
            strategy_name=strategy.name,
            feature_data=data,
            trades=portfolio.trades_frame(),
            history=portfolio.history_frame(),
            config=self.config,
        )
