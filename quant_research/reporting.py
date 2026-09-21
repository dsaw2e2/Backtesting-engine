"""Export reproducible experiment data, charts, metrics, and Markdown summary."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .backtester import BacktestResult
from .metrics import build_round_trips
from .ml_dataset import build_ml_dataset
from .plots import generate_charts


def _display_metric(name: str, value: Any) -> str:
    if value is None:
        return "N/A"
    if name in {"total_return", "annual_return", "annual_volatility", "maximum_drawdown", "win_rate"}:
        return f"{value:.2%}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def export_results(
    result: BacktestResult,
    metrics: dict[str, Any],
    output_root: str | Path,
    ticker: str,
    ml_predictions: pd.DataFrame | None = None,
    ml_evaluation: dict[str, Any] | None = None,
    feature_importance: pd.DataFrame | None = None,
) -> Path:
    """Write one timestamped, self-contained experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = ticker.replace("=", "_").replace("^", "_")
    output_dir = Path(output_root) / f"{safe_ticker}_{result.strategy_name}_{timestamp}"
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=False)

    result.trades.to_csv(output_dir / "trades.csv", index=False)
    build_round_trips(result.trades).to_csv(output_dir / "round_trips.csv", index=False)
    result.history.to_csv(output_dir / "portfolio_history.csv", index=False)
    ml_dataset = build_ml_dataset(result.feature_data)
    ml_dataset.to_csv(output_dir / "ml_dataset.csv", index=False)
    if ml_predictions is not None:
        ml_predictions.to_csv(output_dir / "predictions.csv", index=False)
    if ml_evaluation is not None:
        with (output_dir / "ml_evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(ml_evaluation, handle, indent=2, allow_nan=False)
    if feature_importance is not None:
        feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    experiment_metadata = {
        "ticker": ticker,
        "strategy": result.strategy_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ml_dataset": "ml_dataset.csv",
        "ml_label": "future_5d_return",
        "ml_horizon_days": 5,
    }
    if ml_predictions is not None:
        experiment_metadata["predictions"] = "predictions.csv"
    if ml_evaluation is not None:
        experiment_metadata["ml_evaluation"] = "ml_evaluation.json"
    if feature_importance is not None:
        experiment_metadata["feature_importance"] = "feature_importance.csv"
    with (output_dir / "experiment.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_metadata, handle, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    chart_paths = generate_charts(result.history, result.trades, charts_dir)

    start = result.history.iloc[0]["date"]
    end = result.history.iloc[-1]["date"]
    metric_rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {_display_metric(name, value)} |"
        for name, value in metrics.items()
    )
    chart_links = "\n".join(f"![{path.stem}](charts/{path.name})" for path in chart_paths)
    ml_rows = ""
    if ml_evaluation is not None:
        ml_rows = "\n".join(
            f"| {name.replace('_', ' ').title()} | {_display_metric(name, value)} |"
            for name, value in ml_evaluation.items()
        )
        ml_rows = f"""
## ML Evaluation

| Metric | Value |
|---|---:|
{ml_rows}

Prediction file: `predictions.csv`
"""
    feature_importance_note = ""
    if feature_importance is not None:
        feature_importance_note = """
## Feature Importance

Model-specific feature importance is exported to `feature_importance.csv`.
"""
    report = f"""# Quantitative Backtest Report

## Experiment

| Field | Value |
|---|---|
| Ticker | {ticker} |
| Strategy | {result.strategy_name} |
| Period | {start} to {end} |
| Initial capital | {result.config.initial_capital:,.2f} |
| Transaction fee | {result.config.fee_rate:.4%} |
| Slippage | {result.config.slippage_bps:.2f} bps |
| Execution convention | Signal at close, execution at next open |
| ML dataset | ml_dataset.csv |
| ML label | Future 5-day close-to-close return |

## Metrics

| Metric | Value |
|---|---:|
{metric_rows}

## Charts

{chart_links}

{ml_rows}

{feature_importance_note}

## Methodology Note

Every strategy receives the same point-in-time `MarketState`. Indicators are computed before the event loop and contain no trading decisions. A decision based on candle *t* is executed at candle *t+1* open, so the strategy never trades using a close before that close is observable.

The ML dataset is exported for later modeling only. Each row contains features known at date *t* and the label `future_5d_return`, calculated from future close-to-close performance over the next five trading rows. The last five rows are excluded because the label is not available.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return output_dir
