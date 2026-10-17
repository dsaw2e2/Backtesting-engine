# Systematic Trading Strategy Comparison: ML vs. Rules in a Purged Walk-Forward Setup

Python backtesting platform for comparing rule-based technical strategies against supervised ML models (logistic regression, random forest, XGBoost) and passive benchmarks on daily equity OHLCV data.

The project evaluates whether machine learning models generate directional predictive skill or primarily capture market exposure (beta) across index proxies (QQQ, SPY) and high-momentum single stocks (NVDA) over 2017–2026.

---

## What It Does

- **Walk-forward evaluation**: 23 expanding-window out-of-sample folds (October 2020 – July 2026, 1,427 trading sessions).
- **Label purge gap**: 5-session purge before each test fold to eliminate label horizon overlap on 5-day forward return targets.
- **Next-open execution**: Decisions computed on candle $t$ close execute on candle $t+1$ open. Includes 5 bps commission and 2 bps adverse slippage.
- **Statistical testing**: 1,000 moving-block bootstrap resamples, 1,000 block-permutation tests, and Holm step-down multiple testing adjustments.
- **Flat-to-flat trade accounting**: Trades aggregate by closed position episodes rather than counting partial rebalances as independent round trips.

---

## Out-of-Sample Results (2020-10-26 to 2026-07-02)

23 expanding-window folds, starting capital $100,000, 5 bps fee, 2 bps slippage per execution:

| Asset | Executable Buy & Hold | Technical Indicator | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|---:|---:|
| **NVDA** | **1,369.54%** | 136.94% | 310.46% | 277.64% | 946.42% |
| **QQQ** | **161.76%** | 62.10% | 79.51% | 59.98% | 95.16% |
| **SPY** | **136.92%** | 29.50% | 67.10% | 11.19% | 48.07% |

*Executable Buy & Hold simulates a passive holding entered at next-open prices with transaction costs and slippage applied to the split- and dividend-adjusted series.*

### Key Takeaways

1. **Passive vs. Active**: Buy & hold beats all active strategies in raw return across all three assets. Sitting in cash (0% to 50% allocation) during strong bull markets imposes severe opportunity cost.
2. **ML vs. Rules**: XGBoost and logistic regression beat the rule-based indicator strategy on total return across all assets, but performance varies by model (random forest underperformed the indicator baseline on SPY: 11.19% vs. 29.50%).
3. **Alpha vs. Beta**: Out-of-sample ROC-AUC values hover between 0.48 and 0.54. Under block permutation of labels, departures from 0.50 are not statistically significant after Holm adjustment. Profits appear driven by dynamic exposure to prevailing market trends rather than directional timing skill.
4. **Drawdowns**: While trailing on return, the indicator model and random forest reduced maximum drawdown by roughly 30–50% relative to 100% equity buy & hold by cutting exposure during volatile regimes.

---

## Real Run Output

Running the walk-forward protocol on QQQ (`python paper_research.py --csv data/QQQ.csv --ticker QQQ`):

```text
================================================================================
Paper Research Suite: QQQ
================================================================================
Loaded data/QQQ.csv: 2392 raw rows -> 2188 labeled dataset rows
Folds: 23 expanding folds (initial train: 756 sessions, test step: 63 sessions)
OOS test period: 2020-10-26 to 2026-07-02 (1427 test sessions)

Fold 01/23 | Train: 756 | Test: 63 | Purge: 5 rows | Fitting models...
Fold 05/23 | Train: 1008 | Test: 63 | Purge: 5 rows | Fitting models...
Fold 10/23 | Train: 1323 | Test: 63 | Purge: 5 rows | Fitting models...
Fold 15/23 | Train: 1638 | Test: 63 | Purge: 5 rows | Fitting models...
Fold 20/23 | Train: 1953 | Test: 63 | Purge: 5 rows | Fitting models...
Fold 23/23 | Train: 2125 | Test: 63 | Purge: 5 rows | Fitting models...

Running 1000-resample moving-block bootstrap & block-permutation tests...
Exporting 69 serialized fold estimators and publication charts...

Trading Results (QQQ OOS 2020-2026):
  executable_buy_hold : Return: 161.76% | Sharpe: 0.8678 | MaxDD: -35.12% | Beta: 1.0000
  indicator           : Return:  62.10% | Sharpe: 0.8826 | MaxDD: -14.13% | Beta: 0.2941
  logistic_regression : Return:  79.51% | Sharpe: 0.6356 | MaxDD: -33.92% | Beta: 0.8014
  random_forest       : Return:  59.98% | Sharpe: 0.7759 | MaxDD: -13.15% | Beta: 0.3641
  xgboost             : Return:  95.16% | Sharpe: 0.7746 | MaxDD: -32.44% | Beta: 0.6299

Classification Diagnostics:
  logistic_regression : ROC-AUC: 0.4598 | Brier: 0.2518 | Holm p: 0.4316
  random_forest       : ROC-AUC: 0.5348 | Brier: 0.2519 | Holm p: 0.4316
  xgboost             : ROC-AUC: 0.5205 | Brier: 0.2607 | Holm p: 0.4316
```

---

## Setup & Running

```bash
git clone https://github.com/dsaw2e2/Backtesting-engine.git
cd Backtesting-engine

python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Run Backtests

Single strategy run:
```bash
# Rule-based indicator on QQQ
python main.py --csv data/QQQ.csv --ticker QQQ --strategy indicator

# XGBoost on QQQ (70% train / 30% test)
python main.py --csv data/QQQ.csv --ticker QQQ --strategy xgboost

# Test on NVDA or SPY
python main.py --csv data/NVDA.csv --ticker NVDA --strategy random_forest
python main.py --csv data/SPY.csv --ticker SPY --strategy logistic_regression
```

Full 23-fold walk-forward paper protocol:
```bash
python paper_research.py --csv data/QQQ.csv --ticker QQQ
python paper_research.py --csv data/SPY.csv --ticker SPY
python paper_research.py --csv data/NVDA.csv --ticker NVDA
```

Run test suite:
```bash
python -m unittest discover -s tests -v
```

---

## Technical Notes & Limitations

- **Execution price model**: Orders fill at candle $t+1$ Open. The simulator does not model limit order queue priority, partial fills, or opening cross auction imbalances.
- **Adjusted price caveats**: CSV data uses split- and dividend-adjusted closes from Yahoo Finance. Dividends are implicitly reinvested; actual real-time execution prices would differ slightly.
- **Daily resolution**: Rebalancing occurs strictly once per day at the market open. There are no intraday stop-loss or profit-target triggers.
- **Threshold discretization**: Probability output is mapped to discrete target weights (0%, 25%, 50%, 100%). The portfolio only rebalances when the bucket changes, preventing churn from small daily probability fluctuations.
- **Long-only constraint**: Short selling is not implemented. When the model turns bearish, exposure drops to cash (0%).
- **Asset universe**: Tested on three liquid US instruments (QQQ, SPY, NVDA). Performance on small-cap or illiquid equities would carry higher slippage and different cross-sectional dynamics.

---

## License

MIT
