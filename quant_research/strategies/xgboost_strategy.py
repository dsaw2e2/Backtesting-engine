"""XGBoost strategy using the standard strategy interface."""

from .probability_strategy import ProbabilityStrategy


class XGBoostStrategy(ProbabilityStrategy):
    """Map XGBoost probabilities to target long-only exposure."""

    name = "xgboost"
    display_name = "XGBoost"
