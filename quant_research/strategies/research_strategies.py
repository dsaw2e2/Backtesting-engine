"""Strategies used by walk-forward research and executable baselines."""

from __future__ import annotations

import pandas as pd

from ..market_state import MarketState
from ..signals import Action, StrategyDecision
from .base_strategy import BaseStrategy
from .probability_strategy import (
    PRIMARY_EXPOSURE_THRESHOLDS,
    probability_decision,
)


class PrecomputedProbabilityStrategy(BaseStrategy):
    """Replay strictly out-of-sample probabilities through the normal backtester."""

    def __init__(
        self,
        name: str,
        probabilities: pd.Series,
        thresholds: tuple[float, float, float] = PRIMARY_EXPOSURE_THRESHOLDS,
    ):
        self.name = name
        self.display_name = name.replace("_", " ").title()
        self.thresholds = thresholds
        indexed = probabilities.copy()
        indexed.index = pd.to_datetime(indexed.index)
        self.probabilities = indexed.astype(float).sort_index()

    def predict(self, state: MarketState) -> StrategyDecision:
        try:
            probability = float(self.probabilities.loc[state.date])
        except KeyError as error:
            raise ValueError(f"No out-of-sample probability for {state.date}.") from error
        return probability_decision(probability, self.display_name, self.thresholds)


class FixedExposureStrategy(BaseStrategy):
    """Enter one fixed long-only exposure and hold it for the full experiment."""

    def __init__(self, exposure: float, name: str | None = None):
        if not 0.0 <= exposure <= 1.0:
            raise ValueError("exposure must be between 0 and 1.")
        self.exposure = float(exposure)
        self.name = name or f"fixed_{int(exposure * 100)}"

    def predict(self, state: MarketState) -> StrategyDecision:
        action = Action.BUY if self.exposure > 0.0 else Action.SELL
        return StrategyDecision(
            action,
            self.exposure,
            1.0,
            f"fixed target exposure {self.exposure:.0%}",
        )


class VolatilityTargetStrategy(BaseStrategy):
    """Scale long exposure to a fixed annualized volatility target."""

    name = "volatility_target_15"

    def __init__(self, annual_target: float = 0.15):
        if annual_target <= 0.0:
            raise ValueError("annual_target must be positive.")
        self.annual_target = float(annual_target)

    def predict(self, state: MarketState) -> StrategyDecision:
        target = min(max(self.annual_target / state.historical_volatility, 0.0), 1.0)
        return StrategyDecision(
            Action.BUY if target > 0.0 else Action.SELL,
            target,
            1.0,
            f"{self.annual_target:.0%} volatility target; exposure {target:.2%}",
        )
