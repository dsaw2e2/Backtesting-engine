"""Statistical, calibration, exposure, and regime diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .config import TRADING_DAYS_PER_YEAR
from .ml_model import evaluate_predictions
from .strategies.probability_strategy import probability_to_exposure


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def classification_diagnostics(predictions: pd.DataFrame) -> dict[str, Any]:
    """Extend ordinary classification metrics with calibration and return association."""
    required = {"target", "probability", "predicted_class", "future_5d_return"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {', '.join(missing)}")

    metrics = evaluate_predictions(predictions)
    target = predictions["target"].astype(int)
    probability = predictions["probability"].astype(float)
    future_return = predictions["future_5d_return"].astype(float)
    exposure = probability.map(probability_to_exposure)
    metrics.update(
        {
            "positive_class_rate": float(target.mean()),
            "predicted_positive_rate": float((probability >= 0.5).mean()),
            "always_positive_accuracy": float(target.mean()),
            "average_precision": float(average_precision_score(target, probability)),
            "brier_score": float(brier_score_loss(target, probability)),
            "pearson_return_ic": _finite(float(probability.corr(future_return))),
            "spearman_return_ic": _finite(
                float(probability.corr(future_return, method="spearman"))
            ),
            "average_target_exposure": float(exposure.mean()),
        }
    )
    return metrics


def probability_bucket_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize calibration and realized returns in fixed probability deciles."""
    data = predictions.loc[:, ["probability", "target", "future_5d_return"]].copy()
    data["bucket"] = pd.cut(
        data["probability"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
    )
    grouped = data.groupby("bucket", observed=False)
    result = grouped.agg(
        observations=("target", "size"),
        mean_probability=("probability", "mean"),
        observed_positive_rate=("target", "mean"),
        mean_future_5d_return=("future_5d_return", "mean"),
        median_future_5d_return=("future_5d_return", "median"),
    )
    result = result[result["observations"] > 0].reset_index()
    result["bucket"] = result["bucket"].astype(str)
    return result


def _moving_block_sample(
    length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = math.ceil(length / block_length)
    starts = rng.integers(0, length, size=blocks)
    indices = np.concatenate(
        [(start + np.arange(block_length)) % length for start in starts]
    )
    return indices[:length]


def prediction_significance(
    predictions: pd.DataFrame,
    resamples: int = 1_000,
    block_length: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Estimate block-bootstrap intervals and block-permutation p-values."""
    if resamples <= 0 or block_length <= 0:
        raise ValueError("resamples and block_length must be positive.")
    target = predictions["target"].astype(int).to_numpy()
    probability = predictions["probability"].astype(float).to_numpy()
    future_return = predictions["future_5d_return"].astype(float).to_numpy()
    if len(np.unique(target)) < 2:
        raise ValueError("Prediction significance requires both target classes.")

    observed_auc = float(roc_auc_score(target, probability))
    observed_ic = float(pd.Series(probability).corr(pd.Series(future_return), method="spearman"))
    rng = np.random.default_rng(random_state)
    bootstrap_auc: list[float] = []
    bootstrap_ic: list[float] = []
    permutation_auc: list[float] = []
    permutation_ic: list[float] = []

    label_blocks = [
        np.arange(start, min(start + block_length, len(target)))
        for start in range(0, len(target), block_length)
    ]
    for _ in range(resamples):
        sample = _moving_block_sample(len(target), block_length, rng)
        sampled_target = target[sample]
        if len(np.unique(sampled_target)) == 2:
            bootstrap_auc.append(
                float(roc_auc_score(sampled_target, probability[sample]))
            )
        bootstrap_ic.append(
            float(
                pd.Series(probability[sample]).corr(
                    pd.Series(future_return[sample]),
                    method="spearman",
                )
            )
        )

        order = rng.permutation(len(label_blocks))
        permutation = np.concatenate([label_blocks[index] for index in order])
        permuted_target = target[permutation]
        permuted_return = future_return[permutation]
        permutation_auc.append(float(roc_auc_score(permuted_target, probability)))
        permutation_ic.append(
            float(
                pd.Series(probability).corr(
                    pd.Series(permuted_return),
                    method="spearman",
                )
            )
        )

    auc_extreme = np.abs(np.asarray(permutation_auc) - 0.5) >= abs(observed_auc - 0.5)
    ic_extreme = np.abs(np.asarray(permutation_ic)) >= abs(observed_ic)
    return {
        "roc_auc": observed_auc,
        "roc_auc_ci_low": float(np.percentile(bootstrap_auc, 2.5)),
        "roc_auc_ci_high": float(np.percentile(bootstrap_auc, 97.5)),
        "roc_auc_block_permutation_p_value": float(
            (1 + auc_extreme.sum()) / (resamples + 1)
        ),
        "spearman_return_ic": observed_ic,
        "spearman_ic_ci_low": float(np.nanpercentile(bootstrap_ic, 2.5)),
        "spearman_ic_ci_high": float(np.nanpercentile(bootstrap_ic, 97.5)),
        "spearman_ic_block_permutation_p_value": float(
            (1 + ic_extreme.sum()) / (resamples + 1)
        ),
        "resamples": int(resamples),
        "block_length": int(block_length),
    }


def trading_significance(
    history: pd.DataFrame,
    resamples: int = 1_000,
    block_length: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Block-bootstrap annual return and Sharpe uncertainty from daily portfolio returns."""
    returns = history["portfolio_value"].astype(float).pct_change().dropna().to_numpy()
    if len(returns) < block_length:
        raise ValueError("Portfolio history is too short for the requested block length.")
    rng = np.random.default_rng(random_state)
    annual_returns: list[float] = []
    sharpes: list[float] = []
    for _ in range(resamples):
        sample = returns[_moving_block_sample(len(returns), block_length, rng)]
        annual_return = float(np.prod(1.0 + sample) ** (TRADING_DAYS_PER_YEAR / len(sample)) - 1.0)
        volatility = float(np.std(sample, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = (
            float(np.mean(sample) * TRADING_DAYS_PER_YEAR / volatility)
            if volatility > 0.0
            else np.nan
        )
        annual_returns.append(annual_return)
        sharpes.append(sharpe)
    return {
        "annual_return_ci_low": float(np.nanpercentile(annual_returns, 2.5)),
        "annual_return_ci_high": float(np.nanpercentile(annual_returns, 97.5)),
        "sharpe_ci_low": float(np.nanpercentile(sharpes, 2.5)),
        "sharpe_ci_high": float(np.nanpercentile(sharpes, 97.5)),
        "probability_sharpe_above_zero": float(np.nanmean(np.asarray(sharpes) > 0.0)),
        "resamples": int(resamples),
        "block_length": int(block_length),
    }


def paired_performance_significance(
    strategy_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
    resamples: int = 1_000,
    block_length: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Test paired annual-return differences with blocks kept aligned by date."""
    strategy = strategy_history.set_index(pd.to_datetime(strategy_history["date"]))[
        "portfolio_value"
    ].astype(float).pct_change()
    benchmark = benchmark_history.set_index(pd.to_datetime(benchmark_history["date"]))[
        "portfolio_value"
    ].astype(float).pct_change()
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned) < block_length:
        raise ValueError("Histories are too short for the requested block length.")

    strategy_returns = aligned["strategy"].to_numpy()
    benchmark_returns = aligned["benchmark"].to_numpy()
    difference = strategy_returns - benchmark_returns
    observed = float(difference.mean() * TRADING_DAYS_PER_YEAR)
    rng = np.random.default_rng(random_state)
    bootstrap_differences: list[float] = []
    null_differences: list[float] = []
    blocks = [
        np.arange(start, min(start + block_length, len(difference)))
        for start in range(0, len(difference), block_length)
    ]
    for _ in range(resamples):
        sample = _moving_block_sample(len(difference), block_length, rng)
        strategy_annual = (
            np.prod(1.0 + strategy_returns[sample])
            ** (TRADING_DAYS_PER_YEAR / len(sample))
            - 1.0
        )
        benchmark_annual = (
            np.prod(1.0 + benchmark_returns[sample])
            ** (TRADING_DAYS_PER_YEAR / len(sample))
            - 1.0
        )
        bootstrap_differences.append(float(strategy_annual - benchmark_annual))

        signs = rng.choice((-1.0, 1.0), size=len(blocks))
        null = np.concatenate(
            [difference[block] * signs[index] for index, block in enumerate(blocks)]
        )
        null_differences.append(float(null.mean() * TRADING_DAYS_PER_YEAR))

    extreme = np.abs(np.asarray(null_differences)) >= abs(observed)
    return {
        "observed_annualized_mean_return_difference": observed,
        "annual_return_difference_ci_low": float(
            np.percentile(bootstrap_differences, 2.5)
        ),
        "annual_return_difference_ci_high": float(
            np.percentile(bootstrap_differences, 97.5)
        ),
        "probability_annual_return_difference_above_zero": float(
            np.mean(np.asarray(bootstrap_differences) > 0.0)
        ),
        "block_sign_permutation_p_value": float(
            (1 + extreme.sum()) / (resamples + 1)
        ),
        "resamples": int(resamples),
        "block_length": int(block_length),
    }


def exposure_diagnostics(
    history: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, Any]:
    """Measure market dependence, exposure, turnover, and realized costs."""
    asset_returns = history["price"].astype(float).pct_change()
    portfolio_returns = history["portfolio_value"].astype(float).pct_change()
    aligned = pd.concat(
        [asset_returns.rename("asset"), portfolio_returns.rename("portfolio")],
        axis=1,
    ).dropna()
    variance = float(aligned["asset"].var(ddof=1))
    portfolio_variance = float(aligned["portfolio"].var(ddof=1))
    beta = (
        float(aligned["portfolio"].cov(aligned["asset"]) / variance)
        if variance > 0.0
        else np.nan
    )
    positive = aligned["asset"] > 0.0
    negative = aligned["asset"] < 0.0
    upside_capture = (
        float(aligned.loc[positive, "portfolio"].mean() / aligned.loc[positive, "asset"].mean())
        if positive.any()
        else np.nan
    )
    downside_capture = (
        float(aligned.loc[negative, "portfolio"].mean() / aligned.loc[negative, "asset"].mean())
        if negative.any()
        else np.nan
    )
    years = max(len(aligned) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)
    traded_notional = (
        float((trades["quantity"].astype(float) * trades["price"].astype(float)).sum())
        if not trades.empty
        else 0.0
    )
    average_value = float(history["portfolio_value"].mean())
    return {
        "average_allocation": float(history["position_allocation"].mean()),
        "median_allocation": float(history["position_allocation"].median()),
        "percent_days_invested": float((history["position_allocation"] > 1e-8).mean()),
        "market_correlation": (
            _finite(float(aligned["portfolio"].corr(aligned["asset"])))
            if variance > 0.0 and portfolio_variance > 0.0
            else None
        ),
        "market_beta": _finite(beta),
        "upside_capture": _finite(upside_capture),
        "downside_capture": _finite(downside_capture),
        "total_fees": float(trades["fees"].sum()) if not trades.empty else 0.0,
        "annual_turnover": (
            float(traded_notional / average_value / years)
            if average_value > 0.0
            else None
        ),
    }


def regime_performance(
    history: pd.DataFrame,
    feature_data: pd.DataFrame,
) -> pd.DataFrame:
    """Report strategy returns in ex-post trend and volatility regimes."""
    indexed = history.set_index(pd.to_datetime(history["date"]))
    features = feature_data.loc[indexed.index]
    portfolio_return = indexed["portfolio_value"].astype(float).pct_change()
    volatility_median = float(features["historical_volatility"].median())
    labels = pd.DataFrame(index=indexed.index)
    labels["trend"] = np.where(features["distance_ma200"] >= 0.0, "above_ma200", "below_ma200")
    labels["volatility"] = np.where(
        features["historical_volatility"] >= volatility_median,
        "high_volatility",
        "low_volatility",
    )
    rows: list[dict[str, Any]] = []
    for dimension in ("trend", "volatility"):
        for regime, dates in labels.groupby(dimension).groups.items():
            returns = portfolio_return.loc[dates].dropna()
            rows.append(
                {
                    "dimension": dimension,
                    "regime": regime,
                    "observations": len(returns),
                    "cumulative_return": float((1.0 + returns).prod() - 1.0),
                    "annualized_mean_return": float(returns.mean() * TRADING_DAYS_PER_YEAR),
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Apply Holm's family-wise multiple-testing correction."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    tests = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        corrected = min((tests - rank) * value, 1.0)
        running_max = max(running_max, corrected)
        adjusted[name] = running_max
    return adjusted
