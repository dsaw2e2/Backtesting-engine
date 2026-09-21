"""Strategy implementations and strategy factory."""

from .base_strategy import BaseStrategy
from .indicator_strategy import IndicatorStrategy


def create_strategy(name: str, **kwargs: object) -> BaseStrategy:
    """Create a strategy by CLI name without coupling it to the backtester."""
    normalized = name.lower()
    if normalized == "indicator":
        return IndicatorStrategy(**kwargs)
    if normalized == "logistic_regression":
        from .logistic_regression_strategy import LogisticRegressionStrategy

        return LogisticRegressionStrategy(**kwargs)
    if normalized == "random_forest":
        from .random_forest_strategy import RandomForestStrategy

        return RandomForestStrategy(**kwargs)
    if normalized == "xgboost":
        from .xgboost_strategy import XGBoostStrategy

        return XGBoostStrategy(**kwargs)
    available = "indicator, logistic_regression, random_forest, xgboost"
    raise ValueError(f"Unknown strategy {name!r}. Available: {available}")


__all__ = ["BaseStrategy", "IndicatorStrategy", "create_strategy"]
