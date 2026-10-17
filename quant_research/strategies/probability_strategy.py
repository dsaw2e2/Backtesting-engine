"""Shared probability-to-exposure behavior for supervised ML strategies."""

from __future__ import annotations

import pandas as pd

from ..market_state import MarketState
from ..ml_model import ModelBundle
from ..signals import Action, StrategyDecision
from .base_strategy import BaseStrategy


PRIMARY_EXPOSURE_THRESHOLDS = (0.45, 0.55, 0.60)


def probability_to_exposure(
    probability: float,
    thresholds: tuple[float, float, float] = PRIMARY_EXPOSURE_THRESHOLDS,
) -> float:
    """Map positive-return probability to long-only target exposure."""
    zero_cutoff, half_cutoff, full_cutoff = thresholds
    if not 0.0 <= zero_cutoff < half_cutoff < full_cutoff <= 1.0:
        raise ValueError("Exposure thresholds must be strictly increasing within [0, 1].")
    if probability >= full_cutoff:
        return 1.0
    if probability >= half_cutoff:
        return 0.5
    if probability > zero_cutoff:
        return 0.25
    return 0.0


def probability_decision(
    probability: float,
    display_name: str,
    thresholds: tuple[float, float, float] = PRIMARY_EXPOSURE_THRESHOLDS,
) -> StrategyDecision:
    """Build the standard strategy decision for one model probability."""
    target = probability_to_exposure(probability, thresholds)
    action = Action.BUY if target > 0.0 else Action.SELL
    confidence = min(abs(probability - 0.5) * 2.0, 1.0)
    reason = (
        f"{display_name} probability of positive 5-day return is "
        f"{probability:.2%}; target exposure {target:.0%}"
    )
    return StrategyDecision(action, target, confidence, reason)


class ProbabilityStrategy(BaseStrategy):
    """Map a fitted binary classifier's probability to long-only exposure."""

    name = "probability_model"
    display_name = "Probability model"

    def __init__(self, model: ModelBundle):
        if model.model_name != self.name:
            raise ValueError(
                f"{self.__class__.__name__} requires model {self.name!r}, "
                f"received {model.model_name!r}."
            )
        self.model = model

    def predict(self, state: MarketState) -> StrategyDecision:
        row = pd.DataFrame(
            [
                {
                    "rsi": state.rsi,
                    "macd": state.macd,
                    "macd_hist": state.macd_hist,
                    "distance_ma50": state.distance_ma50,
                    "distance_ma200": state.distance_ma200,
                    "volume_ratio": state.volume_ratio,
                    "historical_volatility": state.historical_volatility,
                    "return_1d": state.return_1d,
                    "return_5d": state.return_5d,
                }
            ],
            columns=list(self.model.feature_columns),
        )
        probability = float(self.model.pipeline.predict_proba(row)[0, 1])
        return probability_decision(probability, self.display_name)
