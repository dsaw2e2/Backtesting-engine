"""Transparent first model combining trend, momentum, RSI, and volume."""

from __future__ import annotations

from ..market_state import MarketState
from ..signals import Action, StrategyDecision
from .base_strategy import BaseStrategy


class IndicatorStrategy(BaseStrategy):
    """Score independent indicators and map the score to a target allocation."""

    name = "indicator"

    def predict(self, state: MarketState) -> StrategyDecision:
        score = 0.0
        evidence: list[str] = []

        if state.ma50 > state.ma200 and state.close > state.ma50:
            score += 2.0
            evidence.append("positive long-term trend")
        elif state.ma50 < state.ma200 and state.close < state.ma50:
            score -= 2.0
            evidence.append("negative long-term trend")

        if state.macd_hist > 0.0 and state.macd > state.macd_signal:
            score += 1.0
            evidence.append("positive MACD momentum")
        elif state.macd_hist < 0.0 and state.macd < state.macd_signal:
            score -= 1.0
            evidence.append("negative MACD momentum")

        if state.rsi < 30.0:
            score += 1.0
            evidence.append("RSI oversold")
        elif state.rsi > 70.0:
            score -= 1.0
            evidence.append("RSI overbought")

        if state.volume_ratio > 1.2:
            volume_direction = 0.5 if state.macd_hist > 0.0 else -0.5
            score += volume_direction
            evidence.append("above-average volume confirms momentum")

        confidence = min(abs(score) / 4.0, 1.0)
        reason = "; ".join(evidence) or "indicators are mixed"

        if score >= 2.5:
            target = 1.0
            action = Action.BUY
        elif score >= 1.0:
            target = 0.5
            action = Action.BUY
        elif score <= -1.0:
            target = 0.0
            action = Action.SELL
        else:
            target = 0.0
            action = Action.HOLD

        if state.historical_volatility > 0.45 and target > 0.5:
            target = 0.5
            reason += "; exposure capped by high volatility"

        return StrategyDecision(action, target, confidence, reason)
