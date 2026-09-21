"""Command-line entry point for reproducible backtest experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_research.backtester import BacktestConfig, Backtester
from quant_research.config import DEFAULT_FEE_RATE, DEFAULT_INITIAL_CAPITAL, DEFAULT_SLIPPAGE_BPS
from quant_research.comparison import export_comparison
from quant_research.features import build_feature_table
from quant_research.market_data import load_csv, load_market_data
from quant_research.metrics import calculate_metrics
from quant_research.reporting import export_results
from quant_research.strategies import create_strategy


ML_STRATEGIES = ("logistic_regression", "random_forest", "xgboost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a historical quantitative research experiment.")
    parser.add_argument("--ticker", default="QQQ", help="Yahoo symbol, e.g. QQQ or NQ=F")
    parser.add_argument(
        "--strategy",
        default="indicator",
        choices=["indicator", *ML_STRATEGIES],
    )
    parser.add_argument("--start", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", help="End date in YYYY-MM-DD format")
    parser.add_argument("--period", default="10y", help="Yahoo period when dates are omitted")
    parser.add_argument("--csv", type=Path, help="Use a local OHLCV CSV instead of downloading")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def run_standard_backtest(args: argparse.Namespace, features) -> None:
    strategy = create_strategy(args.strategy)
    config = BacktestConfig(args.initial_capital, args.fee_rate, args.slippage_bps)
    result = Backtester(config).run(features, strategy)
    metrics = calculate_metrics(result.history, result.trades, args.risk_free_rate)
    output_dir = export_results(result, metrics, args.output_dir, args.ticker)
    _, comparison_report = export_comparison(args.output_dir)

    print(f"Backtest complete: {args.ticker} / {strategy.name}")
    print(f"Period: {result.history.iloc[0]['date']} to {result.history.iloc[-1]['date']}")
    print(f"Total return: {metrics['total_return']:.2%}")
    print(f"Sharpe ratio: {metrics['sharpe_ratio'] if metrics['sharpe_ratio'] is not None else 'N/A'}")
    print(f"Maximum drawdown: {metrics['maximum_drawdown']:.2%}")
    print(f"Results: {output_dir.resolve()}")
    print(f"Model comparison: {comparison_report.resolve()}")


def run_ml_research_backtest(args: argparse.Namespace, features: pd.DataFrame) -> None:
    from quant_research.ml_dataset import build_ml_dataset, chronological_train_test_split
    from quant_research.ml_model import (
        evaluate_predictions,
        feature_importance_table,
        predict_model_dataset,
        train_model,
    )

    dataset = build_ml_dataset(features)
    train_dataset, test_dataset = chronological_train_test_split(dataset, train_fraction=0.70)
    model = train_model(train_dataset, args.strategy)
    predictions = predict_model_dataset(model, test_dataset)
    ml_evaluation = evaluate_predictions(predictions)
    feature_importance = feature_importance_table(model)

    test_dates = pd.to_datetime(test_dataset["date"])
    test_features = features.loc[test_dates].copy()
    config = BacktestConfig(args.initial_capital, args.fee_rate, args.slippage_bps)

    indicator_strategy = create_strategy("indicator")
    indicator_result = Backtester(config).run(test_features, indicator_strategy)
    indicator_metrics = calculate_metrics(
        indicator_result.history, indicator_result.trades, args.risk_free_rate
    )
    indicator_output = export_results(
        indicator_result, indicator_metrics, args.output_dir, args.ticker
    )

    ml_strategy = create_strategy(args.strategy, model=model)
    ml_result = Backtester(config).run(test_features, ml_strategy)
    ml_metrics = calculate_metrics(
        ml_result.history, ml_result.trades, args.risk_free_rate
    )
    ml_output = export_results(
        ml_result,
        ml_metrics,
        args.output_dir,
        args.ticker,
        ml_predictions=predictions,
        ml_evaluation=ml_evaluation,
        feature_importance=feature_importance,
    )
    _, comparison_report = export_comparison(args.output_dir)

    print(f"ML research run complete: {args.ticker} / {args.strategy}")
    print(f"Train rows: {len(train_dataset)}")
    print(f"Test rows: {len(test_dataset)}")
    print(f"Test period: {test_features.index[0]} to {test_features.index[-1]}")
    print(f"ML accuracy: {ml_evaluation['accuracy']:.4f}")
    print(f"ML F1: {ml_evaluation['f1']:.4f}")
    roc_auc = ml_evaluation["roc_auc"]
    print(f"ML ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "ML ROC AUC: N/A")
    print(f"Indicator test results: {indicator_output.resolve()}")
    print(f"{args.strategy} test results: {ml_output.resolve()}")
    print(f"Model comparison: {comparison_report.resolve()}")


def main() -> None:
    args = parse_args()
    market_data = load_csv(args.csv) if args.csv else load_market_data(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        period=args.period,
    )
    features = build_feature_table(market_data)
    if args.strategy in ML_STRATEGIES:
        run_ml_research_backtest(args, features)
    else:
        run_standard_backtest(args, features)


if __name__ == "__main__":
    main()
