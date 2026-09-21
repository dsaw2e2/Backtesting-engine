"""Paper-grade walk-forward research orchestration and reporting."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtester import BacktestConfig, BacktestResult, Backtester
from .config import DEFAULT_FEE_RATE, DEFAULT_INITIAL_CAPITAL, DEFAULT_SLIPPAGE_BPS
from .diagnostics import (
    classification_diagnostics,
    exposure_diagnostics,
    holm_adjust,
    paired_performance_significance,
    prediction_significance,
    probability_bucket_table,
    regime_performance,
    trading_significance,
)
from .features import build_feature_table
from .metrics import build_round_trips, calculate_metrics
from .ml_dataset import ML_FEATURE_COLUMNS, build_ml_dataset
from .plots import generate_charts
from .reproducibility import model_parameters, runtime_metadata, sha256_file
from .research_plots import DISPLAY_NAMES, generate_research_charts
from .strategies import create_strategy
from .strategies.probability_strategy import PRIMARY_EXPOSURE_THRESHOLDS
from .strategies.research_strategies import (
    FixedExposureStrategy,
    PrecomputedProbabilityStrategy,
    VolatilityTargetStrategy,
)
from .walk_forward import WalkForwardResult, run_walk_forward


ML_MODELS = ("logistic_regression", "random_forest", "xgboost")
PRIMARY_RESULTS = (
    "executable_buy_hold",
    "indicator",
    *ML_MODELS,
)


@dataclass(frozen=True)
class PaperResearchConfig:
    """Frozen protocol settings for one complete paper experiment."""

    min_train_rows: int = 756
    test_rows: int = 63
    purge_rows: int = 5
    statistical_resamples: int = 1_000
    statistical_block_rows: int = 20
    random_state: int = 42
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    fee_rate: float = DEFAULT_FEE_RATE
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    risk_free_rate: float = 0.0


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    return value


def _json_dump(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_compatible(value), handle, indent=2, allow_nan=False)


def _probability_series(predictions: pd.DataFrame) -> pd.Series:
    return pd.Series(
        predictions["probability"].to_numpy(),
        index=pd.to_datetime(predictions["date"]),
        dtype=float,
    )


def _run(
    feature_data: pd.DataFrame,
    strategy: Any,
    config: BacktestConfig,
    risk_free_rate: float,
) -> tuple[BacktestResult, dict[str, Any]]:
    result = Backtester(config).run(feature_data, strategy)
    metrics = calculate_metrics(result.history, result.trades, risk_free_rate)
    return result, metrics


def _save_strategy_result(
    output_directory: Path,
    result: BacktestResult,
    metrics: dict[str, Any],
    charts: bool,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(output_directory / "portfolio_history.csv", index=False)
    result.trades.to_csv(output_directory / "executions.csv", index=False)
    build_round_trips(result.trades).to_csv(output_directory / "round_trips.csv", index=False)
    _json_dump(output_directory / "metrics.json", metrics)
    if charts:
        generate_charts(result.history, result.trades, output_directory / "charts")


def _metrics_row(
    name: str,
    metrics: dict[str, Any],
    exposure: dict[str, Any],
) -> dict[str, Any]:
    return {"strategy": name, **metrics, **exposure}


def _aggregate_feature_importance(
    walk_forward_results: dict[str, WalkForwardResult],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model_name, result in walk_forward_results.items():
        aggregate = (
            result.feature_importance.groupby("feature", as_index=False)
            .agg(
                mean_absolute_importance=("absolute_importance", "mean"),
                std_absolute_importance=("absolute_importance", "std"),
                folds=("fold", "nunique"),
            )
        )
        aggregate.insert(0, "model", model_name)
        rows.append(aggregate)
    return pd.concat(rows, ignore_index=True)


def _cost_sensitivity(
    test_features: pd.DataFrame,
    walk_forward_results: dict[str, WalkForwardResult],
    config: PaperResearchConfig,
) -> pd.DataFrame:
    scenarios = {
        "zero_cost": BacktestConfig(config.initial_capital, 0.0, 0.0),
        "primary": BacktestConfig(
            config.initial_capital,
            config.fee_rate,
            config.slippage_bps,
        ),
        "double_cost": BacktestConfig(
            config.initial_capital,
            config.fee_rate * 2.0,
            config.slippage_bps * 2.0,
        ),
        "five_times_cost": BacktestConfig(
            config.initial_capital,
            config.fee_rate * 5.0,
            config.slippage_bps * 5.0,
        ),
    }
    rows: list[dict[str, Any]] = []
    for model_name, walk_forward in walk_forward_results.items():
        probabilities = _probability_series(walk_forward.predictions)
        for scenario, backtest_config in scenarios.items():
            strategy = PrecomputedProbabilityStrategy(model_name, probabilities)
            result, metrics = _run(
                test_features,
                strategy,
                backtest_config,
                config.risk_free_rate,
            )
            rows.append(
                {
                    "model": model_name,
                    "scenario": scenario,
                    "fee_rate": backtest_config.fee_rate,
                    "slippage_bps": backtest_config.slippage_bps,
                    "total_return": metrics["total_return"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "maximum_drawdown": metrics["maximum_drawdown"],
                    "executions": metrics["execution_count"],
                }
            )
    return pd.DataFrame(rows)


def _threshold_sensitivity(
    test_features: pd.DataFrame,
    walk_forward_results: dict[str, WalkForwardResult],
    config: PaperResearchConfig,
) -> pd.DataFrame:
    policies = {
        "aggressive": (0.40, 0.50, 0.55),
        "primary": PRIMARY_EXPOSURE_THRESHOLDS,
        "conservative": (0.50, 0.60, 0.65),
    }
    backtest_config = BacktestConfig(
        config.initial_capital,
        config.fee_rate,
        config.slippage_bps,
    )
    rows: list[dict[str, Any]] = []
    for model_name, walk_forward in walk_forward_results.items():
        probabilities = _probability_series(walk_forward.predictions)
        for policy_name, thresholds in policies.items():
            strategy = PrecomputedProbabilityStrategy(
                model_name,
                probabilities,
                thresholds=thresholds,
            )
            result, metrics = _run(
                test_features,
                strategy,
                backtest_config,
                config.risk_free_rate,
            )
            rows.append(
                {
                    "model": model_name,
                    "policy": policy_name,
                    "zero_cutoff": thresholds[0],
                    "half_cutoff": thresholds[1],
                    "full_cutoff": thresholds[2],
                    "total_return": metrics["total_return"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "maximum_drawdown": metrics["maximum_drawdown"],
                    "average_allocation": result.history["position_allocation"].mean(),
                }
            )
    return pd.DataFrame(rows)


def _format(value: Any, percentage: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if percentage:
        return f"{float(value):.2%}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[tuple[str, str, bool]],
) -> str:
    headers = [title for _, title, _ in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if index == 0 else "---:" for index in range(len(headers))) + "|",
    ]
    for _, values in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                _format(values[name], percentage)
                for name, _, percentage in columns
            )
            + " |"
        )
    return "\n".join(lines)


def _methodology_text(
    ticker: str,
    source_path: Path,
    config: PaperResearchConfig,
    dataset: pd.DataFrame,
    folds: pd.DataFrame,
) -> str:
    first_fold = folds.iloc[0]
    last_fold = folds.iloc[-1]
    return f"""# Methodology

## Research Design

This experiment compares transparent rule-based and supervised machine-learning allocation
strategies on {ticker}. It is an allocation study, not a claim that prices can be forecast
precisely. Every strategy is long-only and is evaluated through the same portfolio and
execution engine.

The exact source file is `{source_path}`. The supervised dataset contains {len(dataset):,}
labeled daily observations. The walk-forward experiment contains {len(folds)} expanding-window
folds. The first training window runs from {first_fold['train_start']} through
{first_fold['train_end']}; the first test fold begins {first_fold['test_start']}. The final
out-of-sample fold ends {last_fold['test_end']}.

## Point-in-Time Features And Label

The approved model features are: {", ".join(ML_FEATURE_COLUMNS)}. RSI, MACD, moving-average
distances, volume ratio, realized volatility, and trailing returns use observations available
at or before the close of date *t*. The label is one when adjusted close at *t+5* exceeds
adjusted close at *t*. `future_5d_return`, `target`, future prices, and every other unapproved
column are excluded by an explicit feature allowlist.

## Walk-Forward Protocol

The first fold requires {config.min_train_rows} labeled training rows. Training expands after
every fold; it never rolls backward or discards earlier observations. Each test fold contains
at most {config.test_rows} trading sessions. Exactly {config.purge_rows} rows are removed
between training and testing so a training label's five-day outcome cannot enter the test
period. Models are fitted independently inside each fold and predictions are generated only
for that fold. Test folds are concatenated into one strictly out-of-sample probability series.

The model hyperparameters, nine features, and primary exposure thresholds were frozen before
this walk-forward suite was run. No random shuffle, cross-fold threshold selection, or
hyperparameter search is performed.

## Models And Allocation

The models are Logistic Regression with training-only standardization, Random Forest, and
XGBoost. Their exact estimator parameters are recorded in `paper_manifest.json`.

Probabilities become target exposures using the frozen primary policy:

- probability at or below 0.45: 0% exposure;
- above 0.45 but below 0.55: 25%;
- 0.55 through below 0.60: 50%;
- 0.60 or greater: 100%.

The backtester rebalances only when the target bucket changes. It does not trade merely to
correct small allocation drift while the target bucket remains unchanged. The separate
volatility-target baseline uses a continuous target and can therefore rebalance more often.

## Execution And Costs

A probability or rule is calculated after the close of candle *t*. Any target change executes
at candle *t+1* open. The final signal has no execution without a following open. Initial
capital is {config.initial_capital:,.2f}, transaction fees are {config.fee_rate:.4%} of traded
notional, and adverse slippage is {config.slippage_bps:.2f} basis points per execution.
Fractional shares are permitted; therefore these results describe ETF/equity accounting rather
than futures contract sizing or margin.

## Baselines

    The executable baselines are cash, initial 25%, 50%, 75%, and 100% buy-and-hold
    allocations, a 15% annual volatility target, and the existing indicator strategy. These
    initial-allocation baselines do not rebalance allocation drift. Executable buy and hold is
    the initial 100% strategy entering at the next open with identical costs. A separate frictionless
close-to-close reference is reported for comparability with conventional market summaries.

## Metrics And Trade Accounting

Portfolio metrics are calculated from daily marked-to-market equity. Sharpe and Sortino use
252 trading sessions per year and the configured risk-free rate. Drawdown is measured from
the running portfolio peak. Turnover uses total executed notional divided by average portfolio
value and annualized by experiment length.

A trade means one completed flat-to-flat position episode. Partial increases and reductions
inside an open position are combined into that episode. `execution_count` remains the number
of individual buy or sell operations. Open positions at the sample end affect portfolio
return but are excluded from round-trip win rate and profit factor.

## Statistical Diagnostics

Classification analysis reports ROC-AUC, precision-recall average precision, Brier score,
class prevalence, calibration by probability decile, and Pearson/Spearman association with
the continuous future five-day return.

Uncertainty is estimated with {config.statistical_resamples:,} deterministic resamples and
blocks of {config.statistical_block_rows} consecutive observations. Moving-block bootstrap
intervals preserve local serial dependence better than independent-row resampling.
Block-permutation tests shuffle contiguous outcome blocks against fixed out-of-sample
probabilities. Holm correction controls family-wise error across the three ML models.
These procedures reduce, but cannot eliminate, uncertainty caused by overlapping five-day
labels and a finite historical sample.

## Robustness

Cost sensitivity reports zero, primary, doubled, and five-times-primary costs. Threshold
sensitivity reports aggressive, primary, and conservative policies. These are robustness
tables, not alternative specifications from which the best result is selected.
Performance is also decomposed by above/below-MA200 trend and high/low realized-volatility
regimes. Regimes are descriptive and are not supplied to the strategy.

## Reproducibility And Limitations

The manifest records the data SHA-256 hash, source path, dates, fold settings, feature list,
model parameters, random seed, package versions, runtime, and Git commit when available.
Every fold model is serialized under `models/`.

This remains a retrospective study. Earlier static results on these assets were inspected
before the walk-forward protocol was introduced, so the historical record is not equivalent
to a live untouched future trial. Yahoo adjusted data can be revised, asset selection can
create selection bias, and QQQ is an ETF proxy rather than a fully specified NQ futures
continuous-contract execution model. Statistical significance must be interpreted together
with economic exposure, costs, and baseline comparisons.
"""


def run_paper_research(
    market_data: pd.DataFrame,
    ticker: str,
    output_root: str | Path,
    source_path: str | Path,
    config: PaperResearchConfig | None = None,
) -> Path:
    """Run and export one complete frozen walk-forward research suite."""
    settings = config or PaperResearchConfig()
    source = Path(source_path).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = ticker.replace("=", "_").replace("^", "_")
    output = Path(output_root) / f"{safe_ticker}_paper_research_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)

    feature_data = build_feature_table(market_data)
    dataset = build_ml_dataset(feature_data)
    walk_forward_results: dict[str, WalkForwardResult] = {}
    for model_name in ML_MODELS:
        model_output = output / "models" / model_name
        result = run_walk_forward(
            dataset,
            model_name,
            min_train_rows=settings.min_train_rows,
            test_rows=settings.test_rows,
            purge_rows=settings.purge_rows,
            models_directory=model_output,
        )
        walk_forward_results[model_name] = result
        result.predictions.to_csv(output / f"predictions_{model_name}.csv", index=False)
        result.folds.to_csv(output / f"folds_{model_name}.csv", index=False)
        result.feature_importance.to_csv(
            output / f"feature_importance_by_fold_{model_name}.csv",
            index=False,
        )

    reference_dates = pd.to_datetime(
        walk_forward_results[ML_MODELS[0]].predictions["date"]
    ).reset_index(drop=True)
    for model_name in ML_MODELS[1:]:
        dates = pd.to_datetime(
            walk_forward_results[model_name].predictions["date"]
        ).reset_index(drop=True)
        if not dates.equals(reference_dates):
            raise RuntimeError(f"{model_name} does not use the common walk-forward dates.")
    test_features = feature_data.loc[reference_dates].copy()

    backtest_config = BacktestConfig(
        settings.initial_capital,
        settings.fee_rate,
        settings.slippage_bps,
    )
    strategy_objects: dict[str, Any] = {
        "cash": FixedExposureStrategy(0.0, "cash"),
        "fixed_25": FixedExposureStrategy(0.25, "fixed_25"),
        "fixed_50": FixedExposureStrategy(0.50, "fixed_50"),
        "fixed_75": FixedExposureStrategy(0.75, "fixed_75"),
        "executable_buy_hold": FixedExposureStrategy(1.0, "executable_buy_hold"),
        "volatility_target_15": VolatilityTargetStrategy(0.15),
        "indicator": create_strategy("indicator"),
    }
    for model_name, walk_forward in walk_forward_results.items():
        strategy_objects[model_name] = PrecomputedProbabilityStrategy(
            model_name,
            _probability_series(walk_forward.predictions),
        )

    results: dict[str, BacktestResult] = {}
    metric_rows: list[dict[str, Any]] = []
    regime_rows: list[pd.DataFrame] = []
    trading_statistics: list[dict[str, Any]] = []
    for name, strategy in strategy_objects.items():
        result, metrics = _run(
            test_features,
            strategy,
            backtest_config,
            settings.risk_free_rate,
        )
        results[name] = result
        exposure = exposure_diagnostics(result.history, result.trades)
        metric_rows.append(_metrics_row(name, metrics, exposure))
        regime = regime_performance(result.history, test_features)
        regime.insert(0, "strategy", name)
        regime_rows.append(regime)
        if name != "cash":
            significance = trading_significance(
                result.history,
                resamples=settings.statistical_resamples,
                block_length=settings.statistical_block_rows,
                random_state=settings.random_state,
            )
            trading_statistics.append({"strategy": name, **significance})
        _save_strategy_result(
            output / "strategies" / name,
            result,
            metrics,
            charts=name in PRIMARY_RESULTS,
        )

    trading_metrics = pd.DataFrame(metric_rows)
    trading_metrics.to_csv(output / "trading_metrics.csv", index=False)
    pd.concat(regime_rows, ignore_index=True).to_csv(
        output / "regime_performance.csv",
        index=False,
    )
    pd.DataFrame(trading_statistics).to_csv(
        output / "trading_significance.csv",
        index=False,
    )

    paired_rows: list[dict[str, Any]] = []
    comparisons = [
        (name, "executable_buy_hold")
        for name in ("indicator", *ML_MODELS)
    ] + [
        (name, "indicator")
        for name in ML_MODELS
    ]
    for strategy_name, benchmark_name in comparisons:
        paired = paired_performance_significance(
            results[strategy_name].history,
            results[benchmark_name].history,
            resamples=settings.statistical_resamples,
            block_length=settings.statistical_block_rows,
            random_state=settings.random_state,
        )
        paired_rows.append(
            {
                "strategy": strategy_name,
                "benchmark": benchmark_name,
                **paired,
            }
        )
    paired_significance = pd.DataFrame(paired_rows)
    for benchmark_name, group in paired_significance.groupby("benchmark"):
        adjusted_p = holm_adjust(
            dict(zip(group["strategy"], group["block_sign_permutation_p_value"]))
        )
        mask = paired_significance["benchmark"] == benchmark_name
        paired_significance.loc[mask, "holm_p_value"] = paired_significance.loc[
            mask, "strategy"
        ].map(adjusted_p)
    paired_significance.to_csv(
        output / "benchmark_significance.csv",
        index=False,
    )

    classification_rows: list[dict[str, Any]] = []
    prediction_statistics: list[dict[str, Any]] = []
    probability_buckets: dict[str, pd.DataFrame] = {}
    for model_name, walk_forward in walk_forward_results.items():
        predictions = walk_forward.predictions
        classification_rows.append(
            {"model": model_name, **classification_diagnostics(predictions)}
        )
        significance = prediction_significance(
            predictions,
            resamples=settings.statistical_resamples,
            block_length=settings.statistical_block_rows,
            random_state=settings.random_state,
        )
        prediction_statistics.append({"model": model_name, **significance})
        buckets = probability_bucket_table(predictions)
        probability_buckets[model_name] = buckets
        buckets.to_csv(output / f"probability_buckets_{model_name}.csv", index=False)

    classification = pd.DataFrame(classification_rows)
    prediction_stats = pd.DataFrame(prediction_statistics)
    adjusted = holm_adjust(
        dict(
            zip(
                prediction_stats["model"],
                prediction_stats["roc_auc_block_permutation_p_value"],
            )
        )
    )
    prediction_stats["roc_auc_holm_p_value"] = prediction_stats["model"].map(adjusted)
    ic_adjusted = holm_adjust(
        dict(
            zip(
                prediction_stats["model"],
                prediction_stats["spearman_ic_block_permutation_p_value"],
            )
        )
    )
    prediction_stats["spearman_ic_holm_p_value"] = prediction_stats["model"].map(
        ic_adjusted
    )
    classification.to_csv(output / "classification_metrics.csv", index=False)
    prediction_stats.to_csv(output / "prediction_significance.csv", index=False)

    feature_importance = _aggregate_feature_importance(walk_forward_results)
    feature_importance.to_csv(output / "feature_importance_summary.csv", index=False)
    cost_sensitivity = _cost_sensitivity(test_features, walk_forward_results, settings)
    cost_sensitivity.to_csv(output / "cost_sensitivity.csv", index=False)
    threshold_sensitivity = _threshold_sensitivity(
        test_features,
        walk_forward_results,
        settings,
    )
    threshold_sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)

    frictionless_return = float(
        test_features["Close"].iloc[-1] / test_features["Close"].iloc[0] - 1.0
    )
    benchmark_reference = {
        "start": reference_dates.iloc[0],
        "end": reference_dates.iloc[-1],
        "frictionless_close_to_close_return": frictionless_return,
        "executable_buy_hold_return": float(
            trading_metrics.loc[
                trading_metrics["strategy"] == "executable_buy_hold",
                "total_return",
            ].iloc[0]
        ),
    }
    _json_dump(output / "benchmark_reference.json", benchmark_reference)

    chart_paths = generate_research_charts(
        results,
        trading_metrics,
        probability_buckets,
        feature_importance,
        output / "charts",
    )

    common_folds = walk_forward_results[ML_MODELS[0]].folds
    manifest = {
        "experiment_type": "purged_expanding_walk_forward",
        "ticker": ticker,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "raw_rows": len(market_data),
        "raw_start": market_data.index.min(),
        "raw_end": market_data.index.max(),
        "feature_rows": len(feature_data),
        "labeled_rows": len(dataset),
        "out_of_sample_rows": len(reference_dates),
        "out_of_sample_start": reference_dates.iloc[0],
        "out_of_sample_end": reference_dates.iloc[-1],
        "folds": len(common_folds),
        "protocol": asdict(settings),
        "features": ML_FEATURE_COLUMNS,
        "label": "adjusted_close[t+5] / adjusted_close[t] - 1 > 0",
        "primary_exposure_thresholds": PRIMARY_EXPOSURE_THRESHOLDS,
        "models": model_parameters(),
        "runtime": runtime_metadata(Path.cwd()),
    }
    _json_dump(output / "paper_manifest.json", manifest)

    methodology = _methodology_text(
        ticker,
        source,
        settings,
        dataset,
        common_folds,
    )
    (output / "methodology.md").write_text(methodology, encoding="utf-8")

    performance_table = _markdown_table(
        trading_metrics[
            trading_metrics["strategy"].isin(
                ["executable_buy_hold", "indicator", *ML_MODELS]
            )
        ],
        [
            ("strategy", "Strategy", False),
            ("total_return", "Total Return", True),
            ("annual_return", "Annual Return", True),
            ("annual_volatility", "Volatility", True),
            ("sharpe_ratio", "Sharpe", False),
            ("maximum_drawdown", "Max Drawdown", True),
            ("average_allocation", "Avg Exposure", True),
            ("market_beta", "Beta", False),
        ],
    )
    classification_table = _markdown_table(
        classification,
        [
            ("model", "Model", False),
            ("roc_auc", "ROC-AUC", False),
            ("average_precision", "PR AUC", False),
            ("brier_score", "Brier", False),
            ("positive_class_rate", "Positive Rate", True),
            ("always_positive_accuracy", "Always-Up Accuracy", True),
            ("accuracy", "Model Accuracy", True),
            ("spearman_return_ic", "Return IC", False),
        ],
    )
    significance_table = _markdown_table(
        prediction_stats,
        [
            ("model", "Model", False),
            ("roc_auc", "ROC-AUC", False),
            ("roc_auc_ci_low", "CI Low", False),
            ("roc_auc_ci_high", "CI High", False),
            ("roc_auc_block_permutation_p_value", "Permutation p", False),
            ("roc_auc_holm_p_value", "Holm p", False),
        ],
    )
    chart_links = "\n".join(
        f"![{path.stem}](charts/{path.name})" for path in chart_paths
    )
    report = f"""# Paper Research Report: {ticker}

## Protocol

This report uses {len(common_folds)} purged expanding-window folds and
{len(reference_dates):,} strictly out-of-sample daily predictions from
{reference_dates.iloc[0].date()} through {reference_dates.iloc[-1].date()}.
No model is selected or tuned inside these test folds.

Frictionless close-to-close buy and hold returned {frictionless_return:.2%}.
The executable next-open buy-and-hold baseline, including identical costs, is shown below.

## Trading Results

{performance_table}

## Classification Results

{classification_table}

## Statistical Uncertainty

{significance_table}

ROC-AUC confidence intervals use moving-block bootstrap. Permutation p-values shuffle
contiguous outcome blocks; Holm p-values correct across the three ML models.

## Robustness Files

- `cost_sensitivity.csv`
- `threshold_sensitivity.csv`
- `regime_performance.csv`
- `trading_significance.csv`
- `benchmark_significance.csv`
- `probability_buckets_<model>.csv`
- `feature_importance_summary.csv`

## Charts

{chart_links}

## Interpretation Rule

High return alone is not evidence of forecasting skill. Conclusions must consider executable
baselines, average exposure, beta, probability calibration, confidence intervals, permutation
tests, turnover, and cost sensitivity together. Full assumptions and limitations are explained
in [methodology.md](methodology.md).
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return output


def export_paper_comparison(output_root: str | Path) -> tuple[Path, Path]:
    """Compare the latest paper suite for every asset on its own OOS period."""
    root = Path(output_root)
    latest: dict[str, tuple[Path, pd.DataFrame, dict[str, Any]]] = {}
    for directory in root.glob("*_paper_research_*"):
        manifest_path = directory / "paper_manifest.json"
        metrics_path = directory / "trading_metrics.csv"
        if not manifest_path.exists() or not metrics_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ticker = str(manifest["ticker"])
            metrics = pd.read_csv(metrics_path)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        previous = latest.get(ticker)
        if previous is None or directory.stat().st_mtime > previous[0].stat().st_mtime:
            latest[ticker] = (directory, metrics, manifest)

    rows: list[dict[str, Any]] = []
    for ticker, (_, metrics, manifest) in sorted(latest.items()):
        row: dict[str, Any] = {
            "Asset": ticker,
            "Start": str(manifest["out_of_sample_start"])[:10],
            "End": str(manifest["out_of_sample_end"])[:10],
        }
        for strategy in PRIMARY_RESULTS:
            match = metrics[metrics["strategy"] == strategy]
            row[DISPLAY_NAMES[strategy]] = (
                float(match.iloc[0]["total_return"]) if not match.empty else np.nan
            )
        rows.append(row)

    comparison = pd.DataFrame(rows)
    csv_path = root / "paper_research_comparison.csv"
    markdown_path = root / "paper_research_comparison.md"
    comparison.to_csv(csv_path, index=False)
    if comparison.empty:
        table = "No completed paper research suites found."
    else:
        columns = [(column, column, column not in {"Asset", "Start", "End"}) for column in comparison]
        table = _markdown_table(comparison, columns)
    markdown_path.write_text(
        "# Walk-Forward Research Comparison\n\n"
        "Each asset uses its own purged expanding-window out-of-sample period. "
        "Returns are executable next-open results with identical configured costs.\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    return csv_path, markdown_path
