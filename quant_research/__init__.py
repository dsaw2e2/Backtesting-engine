"""Modular quantitative research and historical backtesting toolkit."""

from .backtester import BacktestConfig, BacktestResult, Backtester
from .features import build_feature_table
from .market_data import load_market_data
from .ml_dataset import build_ml_dataset

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    "build_feature_table",
    "build_ml_dataset",
    "load_market_data",
]
