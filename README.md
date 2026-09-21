# A Comparative Study of Machine Learning and Rule-Based Allocation Strategies Under a Purged Expanding-Window Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-26%20Passing-brightgreen.svg)]()
[![Validation](https://img.shields.io/badge/Protocol-Purged%20Walk--Forward-orange.svg)]()
[![Safeguards](https://img.shields.io/badge/Point--in--Time%20Safeguards-Tested-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modular quantitative research platform for evaluating and comparing systematic allocation strategies on US equities. 

This repository implements an empirical framework to examine whether supervised machine learning models (Logistic Regression, Random Forest, and XGBoost) exhibit directional predictive ability versus dynamic market exposure when compared to technical indicator heuristics and passive benchmarks.

---

## Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Empirical Results Summary](#empirical-results-summary)
- [Methodological Design & Safeguards](#methodological-design--safeguards)
- [Software Architecture](#software-architecture)
- [Installation & Setup](#installation--setup)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Automated Verification & Unit Tests](#automated-verification--unit-tests)
- [Research Limitations](#research-limitations)
- [Citation](#citation)
- [License](#license)

---

## Overview

In systematic trading research, models frequently report strong historical performance that fails out of sample due to target overlap leakage, in-sample preprocessing contamination, or unmodeled execution friction.

This project implements an end-to-end backtesting and walk-forward evaluation platform adhering to the partitioning principles outlined by Marcos López de Prado (*Advances in Financial Machine Learning*):
1. **Purged Expanding-Window Walk-Forward Protocol**: 23 strictly out-of-sample non-overlapping evaluation folds spanning October 2020 through July 2026 (1,427 trading sessions).
2. **Purge Gaps**: A 5-session purge gap preceding each test fold to prevent overlap between the 5-day forward-return target horizon and test-period prices.
3. **Execution Modeling**: Signals generated after candle $t$ close execute at candle $t+1$ open, incorporating fixed 5 bps transaction fees and 2 bps adverse execution slippage.
4. **Statistical Diagnostics**: 1,000 moving-block bootstrap resamples, 1,000 block-permutation tests, Holm step-down multiple testing adjustments, Brier scores, and probability calibration curves.

---

## Research Questions

The framework evaluates four primary empirical questions:

1. **Do Machine Learning Models Outperform Rule-Based Baselines?**  
   Across the tested assets, the stronger ML strategies (particularly XGBoost and Logistic Regression) produced higher total returns than the fixed technical baseline, although performance varied substantially by model and asset (for example, Random Forest underperformed the indicator baseline on `SPY`).
2. **Can Active Dynamic Allocation Outperform Passive Buy & Hold in a Sustained Bull Market?**  
   No. Across all three assets, simulated executable Buy & Hold produced higher total return over the 2020–2026 out-of-sample period due to the opportunity cost of holding reduced allocations (0%–50% exposure) during secular upward trends.
3. **Are Observed Trading Returns Driven by Directional Prediction or Dynamic Exposure?**  
   The empirical results are consistent with the hypothesis that observed profitability is primarily associated with dynamic market exposure rather than strong directional predictive edge. Under the implemented block-permutation null, out-of-sample ROC-AUC values did not produce a statistically significant departure from 0.50 after Holm adjustment.
4. **How Does Single-Asset Growth Behavior Contrast with Broad Index Proxies?**  
   NVIDIA (`NVDA`) experienced exceptional appreciation over the evaluation period (+1,369% executable Buy & Hold return out-of-sample), providing an extreme single-asset momentum test case. In contrast, `SPY` and `QQQ` serve as broad US equity-market benchmarks with differing sector concentrations.

---

## Empirical Results Summary

### 23-Fold Out-of-Sample Walk-Forward Results (2020-10-26 to 2026-07-02)

All strategies evaluated under identical initial capital ($100,000), next-open execution, and fixed costs (5 bps fee, 2 bps slippage):

| Asset | Executable Buy & Hold | Technical Indicator | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|---:|---:|
| **NVDA** | **1,369.54%** | 136.94% | 310.46% | 277.64% | 946.42% |
| **QQQ** | **161.76%** | 62.10% | 79.51% | 59.98% | 95.16% |
| **SPY** | **136.92%** | 29.50% | 67.10% | 11.19% | 48.07% |

*Note: Executable Buy & Hold simulates passive holding entered at next-open prices with transaction costs and slippage applied to the split- and dividend-adjusted series.*

### Key Diagnostic Observations

- **ROC-AUC & Ranking Ability**: Out-of-sample ROC-AUC ranged between 0.48 and 0.54 across folds and assets. Under block permutation of labels, departures from 0.50 were not statistically significant after Holm correction.
- **Drawdown Behavior**: While trailing Buy & Hold on total return, the indicator strategy and Random Forest reduced maximum drawdowns by 30% to 50% relative to 100% equity exposure by de-allocating capital during volatile periods.
- **Trade Accounting**: Trades are aggregated into flat-to-flat position episodes to avoid counting individual rebalancing executions as separate round trips.

---

## Methodological Design & Safeguards

The platform enforces point-in-time isolation at each stage of the research pipeline:

- **Feature Table & Model Inputs**: `features.py` computes an underlying feature set from historical OHLCV data. The ML models consume an approved 9-feature subset:
  - `rsi`, `macd`, `macd_hist`
  - `distance_ma50`, `distance_ma200`
  - `volume_ratio`, `historical_volatility` (30-day realized)
  - `return_1d`, `return_5d` (trailing close returns)
  Intermediate levels (such as moving average values or MACD signal lines) remain in the feature table but are excluded from the model matrix by an explicit allowlist.
- **Target Formulation**: $Y_t = \mathbb{I}\left(\frac{\text{Close}_{t+5}}{\text{Close}_t} - 1 > 0\right)$. The final 5 rows of any input series are discarded because their forward outcomes are unobservable.
- **Preprocessing Isolation**: All transformers (such as `StandardScaler` for Logistic Regression) are fitted strictly on training data within each fold.
- **Execution Timing**: Signals computed using data through candle $t$ close are executed at candle $t+1$ open. Signals generated on the final evaluation date are excluded because their next-open execution falls outside the defined evaluation window.
- **Exposure Mapping**: Predicted probabilities $p = P(Y=1)$ map monotonically to discrete target portfolio allocations:
  - $p \ge 0.60 \implies 100\%$
  - $0.55 \le p < 0.60 \implies 50\%$
  - $0.45 < p < 0.55 \implies 25\%$
  - $p \le 0.45 \implies 0\%$
  The portfolio rebalances only when the target exposure bucket changes, avoiding unnecessary turnover from normal price drift.

---

## Software Architecture

The platform separates data ingestion, feature generation, strategy logic, and execution accounting into dedicated modules:

```
quant_research/
├── market_data.py          # CSV loading and OHLCV schema validation
├── indicators.py           # Pure numerical indicator functions
├── features.py             # Point-in-time feature calculations
├── market_state.py         # Immutable state dataclass passed to predict(state)
├── signals.py              # StrategyDecision and Action primitives (BUY, SELL, HOLD)
├── portfolio.py            # Long-only position accounting, cash, fees, slippage, PnL
├── backtester.py           # Candle-by-candle replay engine with next-open execution
├── metrics.py              # Performance analytics and flat-to-flat trade statistics
├── ml_dataset.py           # Supervised dataset construction with purge gap
├── ml_model.py             # Estimator pipelines (Logistic Regression, Random Forest, XGBoost)
├── walk_forward.py         # Purged expanding-window cross-validation generator
├── diagnostics.py          # ROC-AUC, Brier score, bootstrap and permutation tests
├── reproducibility.py      # Environment hashing, parameter logging, and manifest export
├── research_plots.py       # Visualization and figure generation
└── strategies/
    ├── base_strategy.py            # Abstract BaseStrategy interface
    ├── indicator_strategy.py       # Rule-based baseline (MA trend + RSI + MACD)
    └── probability_strategy.py     # Probability-to-exposure mapper for ML models
```

---

## Installation & Setup

### 1. Clone Repository & Setup Virtual Environment

```powershell
git clone https://github.com/dsaw2e2/Backtesting-engine.git
cd Backtesting-engine

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

For standard development:
```powershell
python -m pip install -r requirements.txt
```

For exact replication of the numerical environment:
```powershell
python -m pip install -r requirements-lock.txt
```

---

## Reproducing the Experiments

The repository includes date-bounded historical CSV files in `data/` for `QQQ`, `SPY`, and `NVDA` (January 2017 – July 2026).

### Run Single Strategy Backtests

```powershell
# Run rule-based indicator strategy on QQQ
python main.py --csv data/QQQ.csv --ticker QQQ --strategy indicator

# Run XGBoost strategy on QQQ (70% train / 30% test)
python main.py --csv data/QQQ.csv --ticker QQQ --strategy xgboost

# Run on NVDA or SPY
python main.py --csv data/NVDA.csv --ticker NVDA --strategy random_forest
python main.py --csv data/SPY.csv --ticker SPY --strategy logistic_regression
```

Each run exports:
- `portfolio_history.csv` and `trades.csv`
- `metrics.json` and `ml_evaluation.json`
- Performance and trade distribution charts
- A Markdown summary report (`report.md`)
- Auto-updated comparison tables (`outputs/model_comparison.md`)

### Reproduce the 23-Fold Walk-Forward Protocol

Run the purged expanding-window protocol with 1,000 block-bootstrap and permutation resamples:

```powershell
# Run full walk-forward research suite on QQQ
python paper_research.py --csv data/QQQ.csv --ticker QQQ

# Run on SPY and NVDA
python paper_research.py --csv data/SPY.csv --ticker SPY
python paper_research.py --csv data/NVDA.csv --ticker NVDA
```

This exports:
- 69 serialized fold models (`.joblib` bundles per asset)
- Statistical significance tables with Holm-adjusted $p$-values
- Probability calibration diagrams and feature importance figures
- A self-contained `methodology.md` chapter and summary report

---

## Automated Verification & Unit Tests

The test suite contains 26 deterministic unit tests covering portfolio accounting, point-in-time leakage safeguards, walk-forward partitioning, and ML preprocessing pipelines:

```powershell
python -m unittest discover -s tests -v
```

### Key Tested Behaviors
- `test_backtester_executes_signal_at_next_open`: Verifies order execution occurs at candle $t+1$ open.
- `test_feature_row_does_not_change_when_future_prices_change`: Asserts backward-looking feature calculations.
- `test_logistic_scaler_is_fitted_only_on_training_data`: Prevents test-period statistics from affecting feature scaling.
- `test_ml_training_set_has_no_future_rows_from_test_period`: Verifies the 5-session purge gap before test splits.
- `test_shuffled_training_labels_have_near_random_held_out_auc`: Verifies that breaking label relationships yields near-random test AUC.
- `test_walk_forward_folds_are_purged_contiguous_and_non_overlapping`: Asserts out-of-sample partition structure.

---

## Research Limitations

1. **Simulated Execution vs. Live Order Books**: Execution is modeled at next-open prices with fixed transaction fees (5 bps) and adverse slippage (2 bps). Live multi-month forward paper trading in live order books was not performed due to academic timeline constraints.
2. **Universe & Survivorship Considerations**: The tested universe consists of highly liquid, historically successful US instruments (`QQQ`, `SPY`, `NVDA`). Results on less liquid or distressed equities may differ.
3. **Daily Resolution**: The study assumes daily market-on-open rebalancing and 252 trading sessions per year. Intraday execution dynamics and overnight gap risk are not modeled beyond open-price fills.
4. **Data Provenance**: To avoid external API dependency and retroactive data adjustments, empirical results are anchored to static, date-bounded historical CSV archives.

---

## Citation

```bibtex
@article{quant_research_framework_2026,
  title   = {A Comparative Study of Machine Learning and Rule-Based Allocation Strategies Under a Purged Expanding-Window Framework},
  author  = {Miras},
  year    = {2026},
  journal = {Quantitative Research Working Paper},
  url     = {https://github.com/dsaw2e2/Backtesting-engine}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
