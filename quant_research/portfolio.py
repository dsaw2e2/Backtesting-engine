"""Long-only portfolio accounting and target-allocation execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .signals import Action, StrategyDecision


@dataclass
class Trade:
    date: pd.Timestamp
    price: float
    action: str
    quantity: float
    cash: float
    portfolio_value: float
    reason: str
    confidence: float
    realized_pnl: float
    fees: float
    holding_days: float | None


@dataclass
class PortfolioSnapshot:
    date: pd.Timestamp
    price: float
    cash: float
    position: float
    average_entry_price: float
    portfolio_value: float
    realized_pnl: float
    unrealized_pnl: float
    position_allocation: float


class Portfolio:
    """Track cash, long inventory, cost basis, fees, and mark-to-market value."""

    def __init__(self, initial_capital: float, fee_rate: float, slippage_bps: float):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if fee_rate < 0 or slippage_bps < 0:
            raise ValueError("fees and slippage cannot be negative.")
        self.cash = float(initial_capital)
        self.position = 0.0
        self.average_entry_price = 0.0
        self.realized_pnl = 0.0
        self.fee_rate = float(fee_rate)
        self.slippage_bps = float(slippage_bps)
        self.position_open_date: pd.Timestamp | None = None
        self.trades: list[Trade] = []
        self.history: list[PortfolioSnapshot] = []

    def value(self, market_price: float) -> float:
        return self.cash + self.position * market_price

    def rebalance(
        self,
        date: pd.Timestamp,
        market_price: float,
        decision: StrategyDecision,
    ) -> Trade | None:
        """Rebalance to the decision's target percentage at the supplied price."""
        if market_price <= 0:
            raise ValueError("market_price must be positive.")
        if decision.action == Action.HOLD:
            return None

        target = decision.position_size
        portfolio_value = self.value(market_price)
        current_value = self.position * market_price
        desired_value = portfolio_value * target
        value_delta = desired_value - current_value
        tolerance = max(portfolio_value * 1e-8, 1e-8)
        if abs(value_delta) <= tolerance:
            return None

        timestamp = pd.Timestamp(date)
        if value_delta > 0:
            execution_price = market_price * (1.0 + self.slippage_bps / 10_000.0)
            quantity = value_delta / execution_price
            quantity = min(quantity, self.cash / (execution_price * (1.0 + self.fee_rate)))
            if quantity <= 1e-12:
                return None
            gross = quantity * execution_price
            fees = gross * self.fee_rate
            old_cost = self.position * self.average_entry_price
            self.cash -= gross + fees
            self.position += quantity
            self.average_entry_price = (old_cost + gross + fees) / self.position
            if self.position_open_date is None:
                self.position_open_date = timestamp
            action = Action.BUY.value
            realized = 0.0
            holding_days = None
        else:
            execution_price = market_price * (1.0 - self.slippage_bps / 10_000.0)
            desired_quantity = desired_value / market_price
            quantity = min(self.position, self.position - desired_quantity)
            if quantity <= 1e-12:
                return None
            gross = quantity * execution_price
            fees = gross * self.fee_rate
            realized = gross - fees - quantity * self.average_entry_price
            self.cash += gross - fees
            self.position -= quantity
            self.realized_pnl += realized
            holding_days = (
                float((timestamp - self.position_open_date).days)
                if self.position_open_date is not None
                else None
            )
            if self.position <= 1e-10:
                self.position = 0.0
                self.average_entry_price = 0.0
                self.position_open_date = None
            action = Action.SELL.value

        trade = Trade(
            date=timestamp,
            price=float(execution_price),
            action=action,
            quantity=float(quantity),
            cash=float(self.cash),
            portfolio_value=float(self.value(market_price)),
            reason=decision.reason,
            confidence=float(decision.confidence),
            realized_pnl=float(realized),
            fees=float(fees),
            holding_days=holding_days,
        )
        self.trades.append(trade)
        return trade

    def record(self, date: pd.Timestamp, market_price: float) -> None:
        """Record end-of-candle portfolio state at the current close."""
        portfolio_value = self.value(market_price)
        unrealized = self.position * (market_price - self.average_entry_price)
        allocation = self.position * market_price / portfolio_value if portfolio_value else 0.0
        self.history.append(
            PortfolioSnapshot(
                date=pd.Timestamp(date),
                price=float(market_price),
                cash=float(self.cash),
                position=float(self.position),
                average_entry_price=float(self.average_entry_price),
                portfolio_value=float(portfolio_value),
                realized_pnl=float(self.realized_pnl),
                unrealized_pnl=float(unrealized),
                position_allocation=float(allocation),
            )
        )

    def trades_frame(self) -> pd.DataFrame:
        columns = list(Trade.__dataclass_fields__)
        return pd.DataFrame([asdict(item) for item in self.trades], columns=columns)

    def history_frame(self) -> pd.DataFrame:
        columns = list(PortfolioSnapshot.__dataclass_fields__)
        return pd.DataFrame([asdict(item) for item in self.history], columns=columns)
