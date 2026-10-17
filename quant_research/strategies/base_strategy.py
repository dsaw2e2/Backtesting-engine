"""Stable strategy contract used by the historical backtester."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..market_state import MarketState
from ..signals import StrategyDecision


class BaseStrategy(ABC):
    """Interface implemented by rule-based strategies and future ML models."""

    name = "base"

    @abstractmethod
    def predict(self, state: MarketState) -> StrategyDecision:
        """Return a decision using only the supplied point-in-time state."""
        raise NotImplementedError

