"""Expanding-window, purged walk-forward evaluation for supervised models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .ml_model import (
    feature_importance_table,
    predict_model_dataset,
    train_model,
)


@dataclass(frozen=True)
class WalkForwardFold:
    """One immutable chronological training, purge, and test partition."""

    fold: int
    train: pd.DataFrame
    purge: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class WalkForwardResult:
    """Combined strictly out-of-sample output from all walk-forward folds."""

    model_name: str
    predictions: pd.DataFrame
    folds: pd.DataFrame
    feature_importance: pd.DataFrame


def expanding_walk_forward_folds(
    dataset: pd.DataFrame,
    min_train_rows: int = 756,
    test_rows: int = 63,
    purge_rows: int = 5,
) -> list[WalkForwardFold]:
    """Create expanding train windows followed by purge and contiguous test folds."""
    if min_train_rows <= 0 or test_rows <= 0 or purge_rows <= 0:
        raise ValueError("min_train_rows, test_rows, and purge_rows must be positive.")
    if "date" not in dataset.columns:
        raise ValueError("dataset must contain a date column.")

    ordered = dataset.sort_values("date").reset_index(drop=True)
    first_test_start = min_train_rows + purge_rows
    if first_test_start >= len(ordered):
        raise ValueError("dataset is too small for the requested walk-forward protocol.")

    folds: list[WalkForwardFold] = []
    test_start = first_test_start
    fold_number = 1
    while test_start < len(ordered):
        train_end = test_start - purge_rows
        test_end = min(test_start + test_rows, len(ordered))
        train = ordered.iloc[:train_end].copy()
        purge = ordered.iloc[train_end:test_start].copy()
        test = ordered.iloc[test_start:test_end].copy()
        folds.append(WalkForwardFold(fold_number, train, purge, test))
        fold_number += 1
        test_start = test_end
    return folds


def run_walk_forward(
    dataset: pd.DataFrame,
    model_name: str,
    min_train_rows: int = 756,
    test_rows: int = 63,
    purge_rows: int = 5,
    models_directory: str | Path | None = None,
) -> WalkForwardResult:
    """Train once per fold and concatenate predictions made only out of sample."""
    folds = expanding_walk_forward_folds(
        dataset,
        min_train_rows=min_train_rows,
        test_rows=test_rows,
        purge_rows=purge_rows,
    )
    model_path = Path(models_directory) if models_directory is not None else None
    if model_path is not None:
        model_path.mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for fold in folds:
        model = train_model(fold.train, model_name)
        predictions = predict_model_dataset(model, fold.test)
        predictions.insert(
            predictions.columns.get_loc("target") + 1,
            "future_5d_return",
            fold.test["future_5d_return"].to_numpy(),
        )
        predictions.insert(1, "fold", fold.fold)
        prediction_frames.append(predictions)

        importance = feature_importance_table(model)
        importance.insert(0, "fold", fold.fold)
        importance_frames.append(importance)

        if model_path is not None:
            joblib.dump(model, model_path / f"fold_{fold.fold:02d}.joblib")

        fold_rows.append(
            {
                "fold": fold.fold,
                "train_start": pd.Timestamp(fold.train.iloc[0]["date"]),
                "train_end": pd.Timestamp(fold.train.iloc[-1]["date"]),
                "train_rows": len(fold.train),
                "purge_start": pd.Timestamp(fold.purge.iloc[0]["date"]),
                "purge_end": pd.Timestamp(fold.purge.iloc[-1]["date"]),
                "purge_rows": len(fold.purge),
                "test_start": pd.Timestamp(fold.test.iloc[0]["date"]),
                "test_end": pd.Timestamp(fold.test.iloc[-1]["date"]),
                "test_rows": len(fold.test),
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions["date"].duplicated().any():
        raise RuntimeError("Walk-forward predictions contain duplicate dates.")
    if not predictions["date"].is_monotonic_increasing:
        raise RuntimeError("Walk-forward predictions are not chronological.")

    return WalkForwardResult(
        model_name=model_name,
        predictions=predictions,
        folds=pd.DataFrame(fold_rows),
        feature_importance=pd.concat(importance_frames, ignore_index=True),
    )
