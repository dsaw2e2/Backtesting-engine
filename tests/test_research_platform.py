from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from quant_research.backtester import BacktestConfig, Backtester
from quant_research.comparison import ExperimentSummary, build_return_comparison
from quant_research.features import build_feature_table
from quant_research.market_data import load_csv
from quant_research.market_state import MarketState
from quant_research.diagnostics import prediction_significance
from quant_research.metrics import build_round_trips, calculate_metrics
from quant_research.ml_dataset import (
    ML_DATASET_COLUMNS,
    ML_FEATURE_COLUMNS,
    build_ml_dataset,
    chronological_train_test_split,
)
from quant_research.portfolio import Portfolio
from quant_research.reporting import export_results
from quant_research.signals import Action, StrategyDecision
from quant_research.strategies import create_strategy
from quant_research.strategies.base_strategy import BaseStrategy
from quant_research.strategies.probability_strategy import probability_to_exposure
from quant_research.strategies.research_strategies import PrecomputedProbabilityStrategy
from quant_research.walk_forward import expanding_walk_forward_folds, run_walk_forward

try:
    from quant_research.ml_model import (
        build_feature_matrix,
        evaluate_predictions,
        feature_importance_table,
        predict_model_dataset,
        train_logistic_model,
        train_random_forest_model,
        train_xgboost_model,
    )

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    build_feature_matrix = None
    evaluate_predictions = None
    feature_importance_table = None
    predict_model_dataset = None
    train_logistic_model = None
    train_random_forest_model = None
    train_xgboost_model = None
    SKLEARN_AVAILABLE = False

try:
    import xgboost  # noqa: F401

    XGBOOST_AVAILABLE = True
except ModuleNotFoundError:
    XGBOOST_AVAILABLE = False


def sample_market_data(rows: int = 280) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=rows)
    trend = np.linspace(100.0, 150.0, rows)
    cycle = np.sin(np.arange(rows) / 7.0) * 2.0
    close = trend + cycle
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000 + (np.arange(rows) % 20) * 10_000,
        },
        index=index,
    )


def sample_ml_market_data(rows: int = 360) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=rows)
    step = np.arange(rows)
    close = 100.0 + np.sin(step / 4.0) * 8.0 + np.sin(step / 17.0) * 3.0
    return pd.DataFrame(
        {
            "Open": close * (1.0 + np.sin(step / 11.0) * 0.001),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000 + ((step * 37) % 25) * 20_000,
        },
        index=index,
    )


class AlwaysInvestedStrategy(BaseStrategy):
    name = "always_invested"

    def predict(self, state: MarketState) -> StrategyDecision:
        return StrategyDecision(Action.BUY, 1.0, 1.0, f"signal from {state.date.date()}")


class ResearchPlatformTests(unittest.TestCase):
    def test_feature_table_has_complete_reusable_features(self) -> None:
        features = build_feature_table(sample_market_data())
        self.assertGreater(len(features), 50)
        self.assertFalse(features.isna().any().any())
        self.assertTrue((features["rsi"] >= 0).all())
        self.assertTrue((features["rsi"] <= 100).all())

    def test_yahoo_csv_export_uses_adjusted_ohlc_prices(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "QQQ.csv"
            path.write_text(
                "Price,Adj Close,Close,High,Low,Open,Volume\n"
                "Ticker,QQQ,QQQ,QQQ,QQQ,QQQ,QQQ\n"
                "Date,,,,,,\n"
                "2024-01-02,50,100,110,90,95,1000\n",
                encoding="utf-8",
            )
            data = load_csv(path)

        self.assertAlmostEqual(float(data.iloc[0]["Close"]), 50.0)
        self.assertAlmostEqual(float(data.iloc[0]["Open"]), 47.5)
        self.assertAlmostEqual(float(data.iloc[0]["High"]), 55.0)
        self.assertAlmostEqual(float(data.iloc[0]["Low"]), 45.0)
        self.assertEqual(int(data.iloc[0]["Volume"]), 1000)

    def test_backtester_executes_signal_at_next_open(self) -> None:
        features = build_feature_table(sample_market_data())
        result = Backtester(BacktestConfig(fee_rate=0.0, slippage_bps=0.0)).run(
            features, AlwaysInvestedStrategy()
        )
        first_trade = result.trades.iloc[0]
        self.assertEqual(pd.Timestamp(first_trade["date"]), features.index[1])
        self.assertAlmostEqual(float(first_trade["price"]), float(features.iloc[1]["Open"]))
        self.assertIn(str(features.index[0].date()), first_trade["reason"])
        self.assertEqual(len(result.trades), 1)

    def test_ml_dataset_uses_point_in_time_features_and_future_label(self) -> None:
        features = build_feature_table(sample_market_data())
        dataset = build_ml_dataset(features, horizon_days=5)

        self.assertEqual(list(dataset.columns), list(ML_DATASET_COLUMNS))
        self.assertEqual(len(dataset), len(features) - 5)
        self.assertEqual(pd.Timestamp(dataset.iloc[0]["date"]), features.index[0])
        self.assertAlmostEqual(float(dataset.iloc[0]["rsi"]), float(features.iloc[0]["rsi"]))

        expected_return = float(features.iloc[5]["Close"] / features.iloc[0]["Close"] - 1.0)
        self.assertAlmostEqual(float(dataset.iloc[0]["future_5d_return"]), expected_return)
        self.assertEqual(int(dataset.iloc[0]["target"]), int(expected_return > 0.0))

    def test_ml_dataset_excludes_final_five_unlabeled_rows(self) -> None:
        features = build_feature_table(sample_ml_market_data())
        dataset = build_ml_dataset(features, horizon_days=5)

        self.assertEqual(len(dataset), len(features) - 5)
        self.assertEqual(pd.Timestamp(dataset.iloc[-1]["date"]), features.index[-6])

    def test_ml_training_set_has_no_future_rows_from_test_period(self) -> None:
        features = build_feature_table(sample_ml_market_data())
        dataset = build_ml_dataset(features, horizon_days=5)
        train, test = chronological_train_test_split(dataset, train_fraction=0.70, horizon_days=5)
        ordered_dates = pd.Index(pd.to_datetime(dataset["date"]))
        last_train_position = ordered_dates.get_loc(pd.Timestamp(train.iloc[-1]["date"]))
        first_test_position = ordered_dates.get_loc(pd.Timestamp(test.iloc[0]["date"]))

        self.assertLess(last_train_position + 5, first_test_position)
        self.assertLess(pd.Timestamp(train["date"].max()), pd.Timestamp(test["date"].min()))

    def test_walk_forward_folds_are_purged_contiguous_and_non_overlapping(self) -> None:
        features = build_feature_table(sample_ml_market_data(rows=700))
        dataset = build_ml_dataset(features, horizon_days=5)
        folds = expanding_walk_forward_folds(
            dataset,
            min_train_rows=250,
            test_rows=63,
            purge_rows=5,
        )
        ordered_dates = pd.Index(pd.to_datetime(dataset["date"]))
        test_dates: list[pd.Timestamp] = []

        for fold in folds:
            last_train = ordered_dates.get_loc(pd.Timestamp(fold.train.iloc[-1]["date"]))
            first_test = ordered_dates.get_loc(pd.Timestamp(fold.test.iloc[0]["date"]))
            self.assertEqual(len(fold.purge), 5)
            self.assertLess(last_train + 5, first_test)
            self.assertLess(fold.train["date"].max(), fold.purge["date"].min())
            self.assertLess(fold.purge["date"].max(), fold.test["date"].min())
            test_dates.extend(pd.to_datetime(fold.test["date"]).tolist())

        self.assertEqual(len(test_dates), len(set(test_dates)))
        expected = ordered_dates[255:]
        self.assertTrue(pd.Index(test_dates).equals(expected))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required for walk-forward tests")
    def test_walk_forward_predictions_are_only_fold_test_rows_and_models_are_saved(self) -> None:
        features = build_feature_table(sample_ml_market_data(rows=650))
        dataset = build_ml_dataset(features, horizon_days=5)
        with TemporaryDirectory() as temporary_directory:
            result = run_walk_forward(
                dataset,
                "logistic_regression",
                min_train_rows=250,
                test_rows=63,
                purge_rows=5,
                models_directory=temporary_directory,
            )
            model_files = list(Path(temporary_directory).glob("fold_*.joblib"))

        expected_dates = pd.to_datetime(dataset.iloc[255:]["date"]).reset_index(drop=True)
        actual_dates = pd.to_datetime(result.predictions["date"]).reset_index(drop=True)
        self.assertTrue(actual_dates.equals(expected_dates))
        self.assertEqual(len(model_files), len(result.folds))
        self.assertFalse(result.predictions["date"].duplicated().any())

    def test_feature_row_does_not_change_when_future_prices_change(self) -> None:
        market_data = sample_ml_market_data()
        baseline = build_feature_table(market_data)

        changed_future = market_data.copy()
        cutoff = baseline.index[20]
        changed_future.loc[changed_future.index > cutoff, "Close"] *= 3.0
        changed = build_feature_table(changed_future)

        columns = ["rsi", "macd", "macd_hist", "ma50", "ma200", "return_1d", "return_5d"]
        pd.testing.assert_series_equal(
            baseline.loc[cutoff, columns],
            changed.loc[cutoff, columns],
            check_names=False,
        )

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required for ML model tests")
    def test_model_matrix_uses_only_approved_point_in_time_features(self) -> None:
        features = build_feature_table(sample_ml_market_data())
        dataset = build_ml_dataset(features, horizon_days=5)
        dataset["future_price"] = dataset["future_5d_return"] + 1.0
        dataset["close_shift_minus_5"] = dataset["future_5d_return"]
        dataset["leaked_label_copy"] = dataset["target"]

        assert build_feature_matrix is not None
        matrix = build_feature_matrix(dataset)

        self.assertEqual(tuple(matrix.columns), ML_FEATURE_COLUMNS)
        self.assertFalse(
            {"future_5d_return", "future_price", "target", "close_shift_minus_5"}
            .intersection(matrix.columns)
        )

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required for logistic model tests")
    def test_logistic_scaler_is_fitted_only_on_training_data(self) -> None:
        features = build_feature_table(sample_ml_market_data())
        dataset = build_ml_dataset(features, horizon_days=5)
        train, _ = chronological_train_test_split(dataset, train_fraction=0.70, horizon_days=5)
        assert train_logistic_model is not None
        model = train_logistic_model(train)

        scaler = model.pipeline.named_steps["scaler"]
        expected_mean = train.loc[:, list(ML_FEATURE_COLUMNS)].mean().to_numpy()
        full_sample_mean = dataset.loc[:, list(ML_FEATURE_COLUMNS)].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected_mean)
        self.assertGreater(float(np.abs(scaler.mean_ - full_sample_mean).sum()), 1e-9)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required for logistic model tests")
    def test_indicator_and_logistic_use_identical_test_dates_and_costs(self) -> None:
        features = build_feature_table(sample_ml_market_data())
        dataset = build_ml_dataset(features, horizon_days=5)
        train, test = chronological_train_test_split(dataset, train_fraction=0.70, horizon_days=5)
        assert train_logistic_model is not None
        model = train_logistic_model(train)
        test_features = features.loc[pd.to_datetime(test["date"])].copy()
        config = BacktestConfig(fee_rate=0.001, slippage_bps=5.0)

        indicator = Backtester(config).run(test_features, create_strategy("indicator"))
        logistic = Backtester(config).run(
            test_features, create_strategy("logistic_regression", model=model)
        )

        self.assertTrue(indicator.history["date"].equals(logistic.history["date"]))
        self.assertEqual(indicator.config, logistic.config)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required for ML model tests")
    def test_random_forest_uses_identical_test_dates_and_costs(self) -> None:
        features = build_feature_table(sample_ml_market_data())
        dataset = build_ml_dataset(features, horizon_days=5)
        train, test = chronological_train_test_split(dataset, train_fraction=0.70, horizon_days=5)
        assert train_random_forest_model is not None
        model = train_random_forest_model(train)
        test_features = features.loc[pd.to_datetime(test["date"])].copy()
        config = BacktestConfig(fee_rate=0.001, slippage_bps=5.0)

        indicator = Backtester(config).run(test_features, create_strategy("indicator"))
        random_forest = Backtester(config).run(
            test_features, create_strategy("random_forest", model=model)
        )

        self.assertTrue(indicator.history["date"].equals(random_forest.history["date"]))
        self.assertEqual(indicator.config, random_forest.config)

    @unittest.skipUnless(XGBOOST_AVAILABLE, "xgboost is required for XGBoost model tests")
    def test_xgboost_uses_identical_test_dates_and_costs(self) -> None:
        features = build_feature_table(sample_ml_market_data())
        dataset = build_ml_dataset(features, horizon_days=5)
        train, test = chronological_train_test_split(dataset, train_fraction=0.70, horizon_days=5)
        assert train_xgboost_model is not None
        model = train_xgboost_model(train)
        test_features = features.loc[pd.to_datetime(test["date"])].copy()
        config = BacktestConfig(fee_rate=0.001, slippage_bps=5.0)

        indicator = Backtester(config).run(test_features, create_strategy("indicator"))
        xgboost_result = Backtester(config).run(
            test_features, create_strategy("xgboost", model=model)
        )

        self.assertTrue(indicator.history["date"].equals(xgboost_result.history["date"]))
        self.assertEqual(indicator.config, xgboost_result.config)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required for ML model tests")
    def test_random_forest_exports_feature_importance(self) -> None:
        features = build_feature_table(sample_ml_market_data())
        dataset = build_ml_dataset(features, horizon_days=5)
        train, _ = chronological_train_test_split(dataset, train_fraction=0.70, horizon_days=5)
        assert train_random_forest_model is not None
        assert feature_importance_table is not None

        table = feature_importance_table(train_random_forest_model(train))

        self.assertEqual(set(table["feature"]), set(ML_FEATURE_COLUMNS))
        self.assertIn("feature_importance", table.columns)
        self.assertAlmostEqual(float(table["feature_importance"].sum()), 1.0)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required for ML model tests")
    def test_shuffled_training_labels_have_near_random_held_out_auc(self) -> None:
        rng = np.random.default_rng(42)
        rows = 1_200
        feature_values = rng.normal(size=(rows, len(ML_FEATURE_COLUMNS)))
        target = (feature_values[:, 0] + rng.normal(scale=0.35, size=rows) > 0.0).astype(int)
        dataset = pd.DataFrame(feature_values, columns=ML_FEATURE_COLUMNS)
        dataset.insert(0, "date", pd.bdate_range("2018-01-01", periods=rows))
        dataset["future_5d_return"] = np.where(target == 1, 0.01, -0.01)
        dataset["target"] = target
        train, test = chronological_train_test_split(dataset, train_fraction=0.70, horizon_days=5)

        assert train_random_forest_model is not None
        assert predict_model_dataset is not None
        assert evaluate_predictions is not None
        shuffled_aucs = []
        for seed in range(10):
            shuffled = train.copy()
            shuffled["target"] = np.random.default_rng(seed).permutation(
                shuffled["target"].to_numpy()
            )
            model = train_random_forest_model(shuffled)
            evaluation = evaluate_predictions(predict_model_dataset(model, test))
            self.assertIsNotNone(evaluation["roc_auc"])
            shuffled_aucs.append(float(evaluation["roc_auc"]))

        self.assertAlmostEqual(float(np.mean(shuffled_aucs)), 0.5, delta=0.08)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required for logistic model tests")
    def test_final_test_prediction_is_not_executed_without_next_open(self) -> None:
        features = build_feature_table(sample_ml_market_data())
        dataset = build_ml_dataset(features, horizon_days=5)
        train, test = chronological_train_test_split(dataset, train_fraction=0.70, horizon_days=5)
        assert train_logistic_model is not None
        model = train_logistic_model(train)
        test_features = features.loc[pd.to_datetime(test["date"])].copy()

        result = Backtester(BacktestConfig(fee_rate=0.0, slippage_bps=0.0)).run(
            test_features, create_strategy("logistic_regression", model=model)
        )

        if not result.trades.empty:
            self.assertLess(pd.Timestamp(result.trades["date"].max()), test_features.index[-1])

    def test_probability_exposure_thresholds_are_exact(self) -> None:
        self.assertEqual(probability_to_exposure(0.45), 0.0)
        self.assertEqual(probability_to_exposure(0.450001), 0.25)
        self.assertEqual(probability_to_exposure(0.55), 0.5)
        self.assertEqual(probability_to_exposure(0.60), 1.0)

    def test_precomputed_probability_signal_executes_next_open(self) -> None:
        features = build_feature_table(sample_market_data())
        probabilities = pd.Series(0.70, index=features.index)
        strategy = PrecomputedProbabilityStrategy("test_model", probabilities)
        result = Backtester(BacktestConfig(fee_rate=0.0, slippage_bps=0.0)).run(
            features,
            strategy,
        )
        self.assertEqual(pd.Timestamp(result.trades.iloc[0]["date"]), features.index[1])
        self.assertAlmostEqual(float(result.trades.iloc[0]["price"]), float(features.iloc[1]["Open"]))

    def test_portfolio_round_trip_tracks_realized_pnl(self) -> None:
        portfolio = Portfolio(10_000.0, fee_rate=0.0, slippage_bps=0.0)
        portfolio.rebalance(
            pd.Timestamp("2024-01-02"), 100.0, StrategyDecision(Action.BUY, 1.0, 1.0, "enter")
        )
        portfolio.rebalance(
            pd.Timestamp("2024-01-10"), 110.0, StrategyDecision(Action.SELL, 0.0, 1.0, "exit")
        )
        self.assertAlmostEqual(portfolio.position, 0.0)
        self.assertAlmostEqual(portfolio.cash, 11_000.0)
        self.assertAlmostEqual(portfolio.realized_pnl, 1_000.0)

    def test_partial_sells_form_one_completed_round_trip(self) -> None:
        portfolio = Portfolio(10_000.0, fee_rate=0.0, slippage_bps=0.0)
        portfolio.rebalance(
            pd.Timestamp("2024-01-02"),
            100.0,
            StrategyDecision(Action.BUY, 1.0, 1.0, "enter"),
        )
        portfolio.rebalance(
            pd.Timestamp("2024-01-05"),
            110.0,
            StrategyDecision(Action.BUY, 0.5, 1.0, "reduce"),
        )
        portfolio.rebalance(
            pd.Timestamp("2024-01-10"),
            120.0,
            StrategyDecision(Action.SELL, 0.0, 1.0, "exit"),
        )
        round_trips = build_round_trips(portfolio.trades_frame())

        self.assertEqual(len(round_trips), 1)
        self.assertEqual(int(round_trips.iloc[0]["sell_executions"]), 2)
        self.assertAlmostEqual(float(round_trips.iloc[0]["pnl"]), 1500.0)

    def test_prediction_significance_is_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        rows = 240
        probability = rng.uniform(0.2, 0.8, rows)
        target = rng.binomial(1, probability)
        predictions = pd.DataFrame(
            {
                "target": target,
                "probability": probability,
                "future_5d_return": np.where(target == 1, 0.01, -0.01),
            }
        )
        first = prediction_significance(predictions, resamples=50, block_length=10)
        second = prediction_significance(predictions, resamples=50, block_length=10)
        self.assertEqual(first, second)
        self.assertLessEqual(first["roc_auc_ci_low"], first["roc_auc"])
        self.assertGreaterEqual(first["roc_auc_ci_high"], first["roc_auc"])

    def test_hold_does_not_change_the_portfolio(self) -> None:
        portfolio = Portfolio(10_000.0, fee_rate=0.0, slippage_bps=0.0)
        portfolio.rebalance(
            pd.Timestamp("2024-01-02"), 100.0, StrategyDecision(Action.BUY, 0.5, 1.0, "enter")
        )
        cash_before = portfolio.cash
        position_before = portfolio.position
        trade = portfolio.rebalance(
            pd.Timestamp("2024-01-03"), 101.0, StrategyDecision(Action.HOLD, 0.0, 0.0, "wait")
        )
        self.assertIsNone(trade)
        self.assertEqual(portfolio.cash, cash_before)
        self.assertEqual(portfolio.position, position_before)

    def test_metrics_return_required_fields(self) -> None:
        features = build_feature_table(sample_market_data())
        result = Backtester(BacktestConfig(fee_rate=0.0, slippage_bps=0.0)).run(
            features, AlwaysInvestedStrategy()
        )
        metrics = calculate_metrics(result.history, result.trades)
        required = {
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "maximum_drawdown",
            "profit_factor",
            "win_rate",
            "average_trade_pnl",
            "average_holding_days",
            "trade_count",
        }
        self.assertTrue(required.issubset(metrics))

    def test_complete_experiment_export(self) -> None:
        features = build_feature_table(sample_market_data())
        result = Backtester(BacktestConfig(fee_rate=0.0, slippage_bps=0.0)).run(
            features, AlwaysInvestedStrategy()
        )
        metrics = calculate_metrics(result.history, result.trades)
        with TemporaryDirectory() as temporary_directory:
            output = export_results(result, metrics, temporary_directory, "TEST")
            self.assertTrue((output / "trades.csv").exists())
            self.assertTrue((output / "portfolio_history.csv").exists())
            self.assertTrue((output / "ml_dataset.csv").exists())
            self.assertTrue((output / "metrics.json").exists())
            self.assertTrue((output / "report.md").exists())
            self.assertEqual(len(list((output / "charts").glob("*.png"))), 7)

    def test_comparison_aligns_model_and_buy_hold_dates(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=4)
        first = pd.DataFrame(
            {"date": dates, "price": [100, 110, 120, 130], "portfolio_value": [100, 105, 115, 125]}
        )
        second = pd.DataFrame(
            {"date": dates[1:], "price": [110, 120, 130], "portfolio_value": [100, 108, 120]}
        )
        comparison = build_return_comparison(
            [
                ExperimentSummary("TEST", "indicator", Path("first"), first),
                ExperimentSummary("TEST", "hybrid", Path("second"), second),
            ]
        )
        self.assertAlmostEqual(comparison.iloc[0]["Buy & Hold"], 130 / 110 - 1)
        self.assertAlmostEqual(comparison.iloc[0]["Indicator"], 125 / 105 - 1)
        self.assertAlmostEqual(comparison.iloc[0]["Hybrid"], 0.20)


if __name__ == "__main__":
    unittest.main()
