"""Leakage-safe classifier training, evaluation, and prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .ml_dataset import ML_FEATURE_COLUMNS


@dataclass(frozen=True)
class ModelBundle:
    """A fitted classifier plus the feature ordering it expects."""

    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    model_name: str


LogisticModelBundle = ModelBundle


def build_feature_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    """Select the fixed point-in-time feature allowlist in a stable order."""
    missing = sorted(set(ML_FEATURE_COLUMNS).difference(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing feature columns: {', '.join(missing)}")
    return dataset.loc[:, list(ML_FEATURE_COLUMNS)].copy()


def _training_data(train_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "target" not in train_dataset.columns:
        raise ValueError("Training data is missing column: target")
    features = build_feature_matrix(train_dataset)
    target = train_dataset["target"].astype(int)
    if target.nunique() < 2:
        raise ValueError("Binary classification needs both target classes in the training set.")
    return features, target


def create_logistic_pipeline() -> Pipeline:
    """Create the required scikit-learn pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def create_random_forest_pipeline() -> Pipeline:
    """Create the restrained Random Forest baseline."""
    return Pipeline(
        steps=[
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        ]
    )


def create_xgboost_pipeline() -> Pipeline:
    """Create the restrained XGBoost baseline with a lazy dependency import."""
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "XGBoost strategy requires the 'xgboost' package. "
            "Install project dependencies with: python -m pip install -r requirements.txt"
        ) from error

    return Pipeline(
        steps=[
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=10,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        ]
    )


def _fit_model(
    train_dataset: pd.DataFrame,
    pipeline: Pipeline,
    model_name: str,
) -> ModelBundle:
    """Fit one model using only the explicitly supplied training rows."""
    features, target = _training_data(train_dataset)
    pipeline.fit(features, target)
    return ModelBundle(
        pipeline=pipeline,
        feature_columns=ML_FEATURE_COLUMNS,
        model_name=model_name,
    )


def train_logistic_model(train_dataset: pd.DataFrame) -> ModelBundle:
    """Fit preprocessing and logistic regression using training rows only."""
    return _fit_model(train_dataset, create_logistic_pipeline(), "logistic_regression")


def train_random_forest_model(train_dataset: pd.DataFrame) -> ModelBundle:
    """Fit Random Forest using the same leakage-safe training matrix."""
    return _fit_model(train_dataset, create_random_forest_pipeline(), "random_forest")


def train_xgboost_model(train_dataset: pd.DataFrame) -> ModelBundle:
    """Fit XGBoost using the same leakage-safe training matrix."""
    return _fit_model(train_dataset, create_xgboost_pipeline(), "xgboost")


def train_model(train_dataset: pd.DataFrame, model_name: str) -> ModelBundle:
    """Train a supported classifier without exposing model details to callers."""
    trainers = {
        "logistic_regression": train_logistic_model,
        "random_forest": train_random_forest_model,
        "xgboost": train_xgboost_model,
    }
    try:
        trainer = trainers[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown ML model: {model_name!r}") from error
    return trainer(train_dataset)


def predict_model_dataset(
    model: ModelBundle,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Return model predictions with targets, probabilities, and features."""
    missing = sorted({"date", "target"}.difference(dataset.columns))
    if missing:
        raise ValueError(f"Prediction data is missing columns: {', '.join(missing)}")
    if model.feature_columns != ML_FEATURE_COLUMNS:
        raise ValueError("Model feature columns must match the approved point-in-time feature list.")

    features = build_feature_matrix(dataset)
    probability = model.pipeline.predict_proba(features)[:, 1]
    predicted_class = (probability >= 0.5).astype(int)
    predictions = dataset.loc[:, ["date", "target", *model.feature_columns]].copy()
    predictions.insert(2, "probability", probability)
    predictions.insert(3, "predicted_class", predicted_class)
    return predictions


def predict_logistic_dataset(
    model: ModelBundle,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Backward-compatible wrapper for logistic-regression callers."""
    return predict_model_dataset(model, dataset)


def feature_importance_table(model: ModelBundle) -> pd.DataFrame:
    """Return model coefficients or tree importances in descending magnitude."""
    estimator = model.pipeline.named_steps[model.model_name]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
        importance_type = "coefficient"
    else:
        raise ValueError(f"{model.model_name} does not expose feature importance.")

    table = pd.DataFrame(
        {
            "feature": model.feature_columns,
            importance_type: values,
        }
    )
    table["absolute_importance"] = table[importance_type].abs()
    return table.sort_values("absolute_importance", ascending=False).reset_index(drop=True)


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Calculate classification metrics for held-out predictions."""
    target = predictions["target"].astype(int)
    probability = predictions["probability"].astype(float)
    predicted = predictions["predicted_class"].astype(int)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(target, probability))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics
