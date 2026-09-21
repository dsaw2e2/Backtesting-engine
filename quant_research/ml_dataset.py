"""Supervised learning dataset generation from point-in-time features."""

from __future__ import annotations

import pandas as pd


ML_FEATURE_COLUMNS = (
    "rsi",
    "macd",
    "macd_hist",
    "distance_ma50",
    "distance_ma200",
    "volume_ratio",
    "historical_volatility",
    "return_1d",
    "return_5d",
)

ML_DATASET_COLUMNS = (
    "date",
    *ML_FEATURE_COLUMNS,
    "future_5d_return",
    "target",
)


def build_ml_dataset(feature_data: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    """Create one row per day with features at t and future return over t+h.

    The target is computed as Close[t+h] / Close[t] - 1. Features remain
    point-in-time, while the final horizon rows are dropped because their
    future return is not observable inside the sample.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")

    required = {"Close", *ML_FEATURE_COLUMNS}
    missing = sorted(required.difference(feature_data.columns))
    if missing:
        raise ValueError(f"Feature data is missing columns: {', '.join(missing)}")

    data = feature_data.sort_index().copy()
    data["future_5d_return"] = data["Close"].shift(-horizon_days) / data["Close"] - 1.0
    data["target"] = (data["future_5d_return"] > 0.0).astype(int)

    dataset = data.loc[:, [*ML_FEATURE_COLUMNS, "future_5d_return", "target"]]
    dataset = dataset.dropna().reset_index().rename(columns={data.index.name or "index": "date"})
    return dataset.loc[:, list(ML_DATASET_COLUMNS)]


def chronological_train_test_split(
    dataset: pd.DataFrame,
    train_fraction: float = 0.70,
    horizon_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically and purge target windows that touch the test set."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if dataset.empty:
        raise ValueError("dataset is empty.")

    ordered = dataset.sort_values("date").reset_index(drop=True)
    split_index = int(len(ordered) * train_fraction)
    train_end = split_index - horizon_days
    if train_end <= 0 or split_index >= len(ordered):
        raise ValueError("dataset is too small for a non-empty train/test split.")
    train = ordered.iloc[:train_end].copy()
    test = ordered.iloc[split_index:].copy()
    return train, test
