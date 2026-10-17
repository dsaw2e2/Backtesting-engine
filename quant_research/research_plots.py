"""Cross-strategy publication charts for the paper research suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quant_research_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .backtester import BacktestResult


DISPLAY_NAMES = {
    "cash": "Cash",
    "fixed_25": "Initial 25% Buy & Hold",
    "fixed_50": "Initial 50% Buy & Hold",
    "fixed_75": "Initial 75% Buy & Hold",
    "executable_buy_hold": "Executable Buy & Hold",
    "volatility_target_15": "15% Volatility Target",
    "indicator": "Indicator",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

COLORS = {
    "executable_buy_hold": "#222222",
    "indicator": "#D55E00",
    "logistic_regression": "#0072B2",
    "random_forest": "#009E73",
    "xgboost": "#CC79A7",
}


def _setup() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_research_charts(
    results: dict[str, BacktestResult],
    metrics: pd.DataFrame,
    probability_buckets: dict[str, pd.DataFrame],
    feature_importance: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Generate concise cross-model charts used by the paper report."""
    _setup()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    primary = [
        name
        for name in (
            "executable_buy_hold",
            "indicator",
            "logistic_regression",
            "random_forest",
            "xgboost",
        )
        if name in results
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in primary:
        history = results[name].history
        values = history.set_index(pd.to_datetime(history["date"]))["portfolio_value"]
        ax.plot(
            values.index,
            values / values.iloc[0],
            label=DISPLAY_NAMES[name],
            color=COLORS[name],
            linewidth=1.4,
        )
    ax.set(title="Out-of-Sample Equity Curves", ylabel="Growth of $1", xlabel="")
    ax.legend(ncol=2)
    paths.append(_save(fig, output / "equity_comparison.png"))

    ordered = metrics.sort_values("total_return")
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [DISPLAY_NAMES.get(name, name) for name in ordered["strategy"]]
    ax.barh(labels, ordered["total_return"], color="#2874A6")
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(title="Out-of-Sample Total Return", xlabel="Total return", ylabel="")
    paths.append(_save(fig, output / "return_comparison.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in metrics.itertuples(index=False):
        name = row.strategy
        ax.scatter(
            row.annual_volatility,
            row.annual_return,
            s=45,
            color=COLORS.get(name, "#777777"),
        )
        ax.annotate(
            DISPLAY_NAMES.get(name, name),
            (row.annual_volatility, row.annual_return),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(title="Annualized Risk and Return", xlabel="Volatility", ylabel="Annual return")
    paths.append(_save(fig, output / "risk_return.png"))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in primary:
        history = results[name].history
        values = history.set_index(pd.to_datetime(history["date"]))["portfolio_value"]
        drawdown = values / values.cummax() - 1.0
        ax.plot(
            drawdown.index,
            drawdown,
            label=DISPLAY_NAMES[name],
            color=COLORS[name],
            linewidth=1.1,
        )
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(title="Out-of-Sample Drawdowns", ylabel="Drawdown", xlabel="")
    ax.legend(ncol=2)
    paths.append(_save(fig, output / "drawdown_comparison.png"))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.0, label="Perfect")
    for name, buckets in probability_buckets.items():
        ax.plot(
            buckets["mean_probability"],
            buckets["observed_positive_rate"],
            marker="o",
            linewidth=1.2,
            label=DISPLAY_NAMES.get(name, name),
            color=COLORS.get(name),
        )
    ax.set(
        title="Out-of-Sample Probability Calibration",
        xlabel="Mean predicted probability",
        ylabel="Observed positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    paths.append(_save(fig, output / "probability_calibration.png"))

    exposure = metrics.sort_values("average_allocation")
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [DISPLAY_NAMES.get(name, name) for name in exposure["strategy"]]
    ax.barh(labels, exposure["average_allocation"], color="#009E73")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(title="Average Out-of-Sample Market Exposure", xlabel="Average allocation", ylabel="")
    paths.append(_save(fig, output / "average_exposure.png"))

    models = list(probability_buckets)
    fig, axes = plt.subplots(1, len(models), figsize=(12, 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    feature_order = (
        feature_importance.groupby("feature")["mean_absolute_importance"]
        .mean()
        .sort_values()
        .index
    )
    for axis, model_name in zip(axes, models):
        model_importance = feature_importance[
            feature_importance["model"] == model_name
        ].set_index("feature").reindex(feature_order)
        axis.barh(
            model_importance.index,
            model_importance["mean_absolute_importance"],
            color=COLORS.get(model_name, "#2874A6"),
        )
        axis.set_title(DISPLAY_NAMES.get(model_name, model_name))
        axis.set_xlabel("Mean absolute importance")
    paths.append(_save(fig, output / "feature_importance.png"))
    return paths
