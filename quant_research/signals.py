"""Strategy output types shared by rule-based and future ML models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class StrategyDecision:
    """A strategy decision; position_size is the desired allocation for BUY/SELL."""

    action: Action
    position_size: float
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.position_size <= 1.0:
            raise ValueError("position_size must be between 0 and 1.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if self.action == Action.SELL and self.position_size != 0.0:
            raise ValueError("A long-only SELL decision must target a 0% position.")
        if self.action == Action.HOLD and self.position_size != 0.0:
            raise ValueError("HOLD must use 0%; the portfolio preserves its current position.")
