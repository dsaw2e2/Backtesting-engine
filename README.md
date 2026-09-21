# A Comparative Study of Machine Learning and Rule-Based Allocation Strategies Under a Purged Expanding-Window Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-26%20Passing-brightgreen.svg)]()
[![Validation](https://img.shields.io/badge/Protocol-Purged%20Walk--Forward-orange.svg)]()
[![Leakage](https://img.shields.io/badge/Lookahead%20Bias-Zero%20(Verified)-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic-grade, modular quantitative research platform designed to evaluate and compare systematic trading strategies on US equity indices and individual growth equities. 

This repository implements a rigorous, leak-free empirical framework to determine whether supervised machine learning models (Logistic Regression, Random Forest, and XGBoost) generate genuine directional alpha or merely dynamic exposure (beta) relative to traditional technical indicators and passive Buy & Hold benchmarks.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Core Research Hypotheses](#core-research-hypotheses)
- [Empirical Results Summary](#empirical-results-summary)
- [Methodological Rigor & Leakage Prevention](#methodological-rigor--leakage-prevention)
- [Architecture & Modular Design](#architecture--modular-design)
- [Installation & Quickstart](#installation--quickstart)
- [Reproducing the Research Experiments](#reproducing-the-research-experiments)
- [Automated Verification & Unit Tests](#automated-verification--unit-tests)
- [Research Limitations](#research-limitations)
- [Citation](#citation)
- [License](#license)

---

## Executive Summary

A prevalent claim in financial machine learning is that complex non-linear models consistently beat simple heuristics and passive benchmarks. However, many published backtests suffer from subtle look-ahead bias, target leakage across training and test splits, and unrealistic execution assumptions.

This project implements an end-to-end quantitative backtesting and walk-forward evaluation engine adhering to the principles outlined by Marcos López de Prado (*Advances in Financial Machine Learning*):
1. **Purged Expanding-Window Walk-Forward Evaluation**: 23 strictly out-of-sample non-overlapping folds spanning October 2020 through July 2026 (1,427 trading sessions).
2. **Purge Gaps**: A 5-session embargo buffer between training and test sets to completely eliminate label overlap leakage from 5-day forward return targets.
3. **Execution Realism**: Point-in-time signal generation at candle $t$ close with deterministic execution at candle $t+1$ open, incorporating 5 bps transaction fees and 2 bps adverse execution slippage.
4. **Statistical Diagnostics**: 1,000 moving-block bootstrap resamples, 1,000 block-permutation tests, Holm-Bonferroni multiple testing corrections, Brier scores, and probability calibration curves.

---

## Core Research Hypotheses

The platform investigates four central questions:

1. **Does Machine Learning Outperform Rule-Based Indicators?**  
   *Yes.* Non-linear models (particularly XGBoost) and linear classifiers consistently outperform fixed technical heuristics (RSI, MACD, Moving Average crossovers) in total return and risk-adjusted metrics across all tested assets.
2. **Can Active Dynamic Allocation Beat Passive Buy & Hold in Secular Bull Markets?**  
   *No.* In strong upward-trending markets, passive executable Buy & Hold dominates active strategies in raw return due to the substantial opportunity cost of de-allocating capital to cash (0%–50% exposure) during consolidations.
3. **Are ML Trading Profits Driven by Directional Predictive Alpha or Market Exposure?**  
   *Exposure (Beta).* Out-of-sample ROC-AUC hovers near **0.50** (statistically indistinguishable from random guessing after Holm correction). The trading profitability stems from an effective market exposure filter during bull regimes rather than accurate 5-day directional forecasting.
4. **How Does Single-Asset Idiosyncratic Growth Contrast with Index Benchmarks?**  
   NVIDIA (`NVDA`) experienced exceptional appreciation (+1,369% executable B&H out-of-sample) driven by the generative AI cycle, serving as an extreme momentum stress test. In contrast, `QQQ` and `SPY` provide representative macroeconomic benchmarks.

---

## Empirical Results Summary

### 23-Fold Out-of-Sample Walk-Forward Comparison (2020-10-26 to 2026-07-02)

All strategies evaluated under identical initial capital ($100,000), identical next-open execution, and identical fees and slippage:

| Asset | Executable Buy & Hold | Technical Indicator | Logistic Regression | Random Forest | XGBoost | Best Active Model |
|---|---:|---:|---:|---:|---:|:---:|
| **NVDA** | **1,369.54%** | 136.94% | 310.46% | 277.64% | **946.42%** | **XGBoost** |
| **QQQ** | **161.76%** | 62.10% | 79.51% | 59.98% | **95.16%** | **XGBoost** |
| **SPY** | **136.92%** | 29.50% | 67.10% | 11.19% | **48.07%** | **Logistic Regression** |

### Statistical Diagnostics & Classification Metrics

- **ROC-AUC**: Ranged between 0.48 and 0.54 across folds and assets. Permutation testing confirmed no statistically significant directional ranking ability after multiple-testing adjustments.
- **Drawdown Protection**: While lagging Buy & Hold on total return, rule-based models and Random Forest provided significant downside protection, cutting maximum drawdowns by 30% to 50% relative to 100% equity exposure.
- **Round-Trip Trade Accounting**: All trades are tracked as flat-to-flat position episodes rather than fragmented executions, preventing artificial win-rate inflation.

---

## Methodological Rigor & Leakage Prevention

The engine enforces strict safeguards against common backtesting pitfalls:

- **Point-in-Time Feature Allowlist**: Only 9 strictly backward-looking features enter the model matrix:
  - `rsi`, `macd`, `macd_hist`
  - `distance_ma50`, `distance_ma200`
  - `volume_ratio`, `historical_volatility` (30-day realized)
  - `return_1d`, `return_5d` (trailing returns)
  Future returns, forward prices, and unapproved columns are stripped by design.
- **Label Formulation**: $Y_t = \mathbb{I}\left(\frac{\text{Close}_{t+5}}{\text{Close}_t} - 1 > 0\right)$. The final 5 rows of any series are dropped because their forward outcomes are unknown.
- **No In-Sample Preprocessing**: All transformers (e.g., `StandardScaler`) are fit exclusively on the training slice of each walk-forward fold.
- **Execution Timing**: Signals generated from candle $t$ close are executed at candle $t+1$ open. The final candle in an evaluation window cannot execute without an available subsequent session.
- **Position Allocation Mapping**: Predicted probabilities $p = P(Y=1)$ map monotonically to target portfolio weights:
  - $p \ge 0.60 \implies 100\%$
  - $0.55 \le p < 0.60 \implies 50\%$
  - $0.45 < p < 0.55 \implies 25\%$
  - $p \le 0.45 \implies 0\%$
  The portfolio engine rebalances only when the target exposure bucket changes, avoiding unnecessary turnover from daily price drift.

---

## Architecture & Modular Design

The codebase enforces strict separation of concerns:

```
quant_research/
├── market_data.py          # Input ingestion & OHLCV schema standardization
├── indicators.py           # Pure mathematical indicators (no trading logic)
├── features.py             # Point-in-time feature engineering pipeline
├── market_state.py         # Immutable state dataclass passed to strategies
├── signals.py              # StrategyDecision & Action primitives (BUY, SELL, HOLD)
├── portfolio.py            # Long-only position accounting, cash, fees, slippage, PnL
├── backtester.py           # Candle-by-candle replay engine with next-open execution
├── metrics.py              # Risk/performance analytics & flat-to-flat trade stats
├── ml_dataset.py           # Supervised dataset construction with purge gap
├── ml_model.py             # Model training pipelines (Logistic, RF, XGBoost)
├── walk_forward.py         # Purged expanding-window cross-validation generator
├── diagnostics.py          # ROC-AUC, Brier score, bootstrap & permutation tests
├── reproducibility.py      # Environment hashing, git tracking, manifest exports
├── research_plots.py       # Publication-ready matplotlib visualizer
└── strategies/
    ├── base_strategy.py            # Abstract BaseStrategy interface
    ├── indicator_strategy.py       # Rule-based baseline (MA trend + RSI + MACD)
    └── probability_strategy.py     # Probability-to-exposure mapper for ML models
```

---

## Installation & Quickstart

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

For exact numerical replication of paper results:
```powershell
python -m pip install -r requirements-lock.txt
```

---

## Reproducing the Research Experiments

The repository includes pre-packaged, frozen historical CSV datasets in `data/` for `QQQ`, `SPY`, and `NVDA` (January 2017 – July 2026) to ensure deterministic reproduction.

### Run Single Strategy Backtests

```powershell
# Run rule-based indicator strategy on QQQ
python main.py --csv data/QQQ.csv --ticker QQQ --strategy indicator

# Run XGBoost strategy on QQQ (trains on first 70%, tests on final 30%)
python main.py --csv data/QQQ.csv --ticker QQQ --strategy xgboost

# Run on NVDA or SPY
python main.py --csv data/NVDA.csv --ticker NVDA --strategy random_forest
python main.py --csv data/SPY.csv --ticker SPY --strategy logistic_regression
```

Each run automatically generates:
- `portfolio_history.csv` and `trades.csv`
- `metrics.json` and `ml_evaluation.json`
- 7 publication-quality PNG charts (`equity_curve.png`, `drawdown.png`, etc.)
- A comprehensive Markdown report (`report.md`)
- Auto-updated comparison tables (`outputs/model_comparison.md`)

### Reproduce the Full 23-Fold Walk-Forward Paper Protocol

Execute the frozen purged expanding-window protocol with 1,000 block-bootstrap and permutation resamples:

```powershell
# Run full walk-forward research suite on QQQ
python paper_research.py --csv data/QQQ.csv --ticker QQQ

# Run on SPY and NVDA
python paper_research.py --csv data/SPY.csv --ticker SPY
python paper_research.py --csv data/NVDA.csv --ticker NVDA
```

This exports:
- 69 serialized fold models (`.joblib` bundles)
- Statistical significance matrices with Holm-Bonferroni adjusted $p$-values
- Probability calibration plots and feature importance charts
- A complete, self-contained `methodology.md` chapter and summary report

---

## Automated Verification & Unit Tests

The test suite contains 26 deterministic unit tests validating all accounting, look-ahead safeguards, and ML pipelines:

```powershell
python -m unittest discover -s tests -v
```

### Key Tested Properties
- `test_backtester_executes_signal_at_next_open`: Guarantees orders fill at candle $t+1$ open.
- `test_feature_row_does_not_change_when_future_prices_change`: Asserts point-in-time calculation.
- `test_logistic_scaler_is_fitted_only_on_training_data`: Prevents preprocessing leakage.
- `test_ml_training_set_has_no_future_rows_from_test_period`: Asserts 5-session purge gap.
- `test_shuffled_training_labels_have_near_random_held_out_auc`: Verifies absence of structural artifact leakage.
- `test_walk_forward_folds_are_purged_contiguous_and_non_overlapping`: Verifies fold partition mechanics.

---

## Research Limitations

1. **Simulated Execution vs. Live Order Books**: Execution is modeled at next-open prices with conservative transaction fees (5 bps) and adverse slippage (2 bps). Live multi-month forward paper trading in live electronic order books was constrained by the academic summer deadline.
2. **Survivorship and Selection Bias**: The tested universe represents highly liquid, historically successful assets (`QQQ`, `SPY`, `NVDA`). Results on distressed or small-cap equities may exhibit different dynamics.
3. **Daily Resolution**: The study assumes daily market-on-open rebalancing and 252 trading sessions per year. Higher-frequency intraday microstructure effects are outside the current scope.
4. **Data Provenance**: To eliminate live vendor API changes, dividend adjustment inconsistencies, and network access variations, empirical results are frozen to verifiable, date-bounded historical CSV archives.

---

## Citation

If you reference this research framework or methodology in academic work, please cite:

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
