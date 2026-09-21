"""Logistic-regression strategy using the standard strategy interface."""

from .probability_strategy import ProbabilityStrategy


class LogisticRegressionStrategy(ProbabilityStrategy):
    """Map positive-return probability to a target long-only exposure."""

    name = "logistic_regression"
    display_name = "Logistic regression"
