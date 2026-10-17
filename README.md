# Systematic Trading Strategy Comparison: Machine Learning vs. Rule-Based Allocation Under a Purged Expanding-Window Framework

An empirical research platform in Python designed to evaluate whether machine learning models or simple technical indicators generate profitable out-of-sample trading signals after transaction costs, and how their performance compares to passive buy-and-hold benchmarks.

---

## 1. Research Question & Core Findings

### Primary Research Question
**Can simple technical indicators or supervised machine learning models generate useful out-of-sample trading signals after realistic execution costs, and how do their results compare with a passive buy-and-hold benchmark?**

### Key Findings
1. **Passive Benchmarks Dominate Raw Return**: Over the 2020–2026 out-of-sample window, executable buy-and-hold outperformed all active strategies across all three assets (`QQQ`, `SPY`, `NVDA`). In sustained bull regimes, holding partial or zero equity exposure imposes severe opportunity costs.
2. **Machine Learning Outperformed Technical Heuristics on Cumulative Return, but Not Uniformly**: XGBoost and logistic regression generated higher total returns than the fixed indicator heuristic across all assets, but performance was model- and asset-dependent (e.g., random forest underperformed the indicator baseline on `SPY`: 11.19% vs. 29.50%).
3. **Statistical vs. Economic Significance**: Outperformance of ML over the indicator heuristic was statistically significant after Holm multiple testing correction only for XGBoost on `NVDA` ($p = 0.048$). On `QQQ` ($p = 1.000$) and `SPY` ($p = 0.665$), return differences vs. the indicator strategy were statistically indistinguishable from zero under block permutation testing.
4. **No Measurable Directional Edge (Alpha)**: Out-of-sample ROC-AUC values ranged between 0.45 and 0.53 across assets. Departures from 0.50 were not statistically significant after Holm correction. Active profitability was driven by varying exposure to prevailing market trends (beta management), not directional prediction over 5-day horizons.
5. **Drawdown Reduction**: While lagging on total return, the indicator strategy and random forest reduced maximum drawdown by approximately 30% to 50% relative to buy-and-hold by de-allocating capital during high-volatility regimes.

---

## 2. Engine Implementation & Execution Model

The platform executes historical simulations under explicit point-in-time constraints:

- **Next-Open Execution**: Decisions calculated after candle $t$ close are executed at candle $t+1$ open. Signals on the final evaluation date are excluded because their next-open execution falls outside the sample window.
- **Transaction Costs & Slippage**:
  - Commission: fixed 5 bps (0.0500%) per trade notional.
  - Adverse slippage: fixed 2 bps (0.0200%) added to buy prices and subtracted from sell prices.
- **Position Allocation**: Long-only target weights mapped from model output probabilities $p = P(Y=1)$:
  - $p \ge 0.60 \implies 100\%$ target equity allocation
  - $0.55 \le p < 0.60 \implies 50\%$ target equity allocation
  - $0.45 < p < 0.55 \implies 25\%$ target equity allocation
  - $p \le 0.45 \implies 0\%$ (100% cash)
- **Turnover Filter**: Portfolio rebalances only when the target exposure bucket changes, avoiding transaction drag from daily price drift.
- **Flat-to-Flat Trade Accounting**: Win rate and trade statistics are calculated on closed position episodes (cash-to-cash round trips) rather than individual rebalancing executions.

---

## 3. Data & Validation Protocol

### Datasets
Experiments evaluate three liquid US instruments from January 3, 2017 to July 10, 2026 (2,392 raw daily sessions), archived as static CSV files in `data/`:
- `QQQ`: Invesco QQQ Trust (large-cap growth & technology equity benchmark)
- `SPY`: SPDR S&P 500 ETF Trust (broad US large-cap equity benchmark)
- `NVDA`: NVIDIA Corporation (single-asset high-momentum growth outlier)

### Walk-Forward Protocol
The study implements a purged expanding-window cross-validation protocol adhering to Marcos López de Prado (*Advances in Financial Machine Learning*):
- **Expanding Training Window**: Initial training window of 756 labeled trading days (~3 years), expanding by 63 days per fold.
- **Test Folds**: 23 contiguous, non-overlapping out-of-sample folds of 63 trading days (~1 calendar quarter) each.
- **Purge Gap**: Exactly 5 trading sessions dropped between the end of each training window and the start of the test fold to eliminate label overlap from the 5-day forward target.
- **Out-of-Sample Evaluation Window**: October 26, 2020 through July 2, 2026 (**1,427 aggregate out-of-sample trading sessions**).

### Feature Allowlist & Target Definition
- **Target Variable**: $Y_t = \mathbb{I}\left(\frac{\text{Close}_{t+5}}{\text{Close}_t} - 1 > 0\right)$. The final 5 rows of each series are discarded due to unobservable forward prices.
- **9 Backward-Looking Features**:
  - `rsi`: 14-day Wilder Relative Strength Index
  - `macd`, `macd_hist`: (12, 26, 9) MACD line and histogram
  - `distance_ma50`, `distance_ma200`: percentage distance of close from 50-day and 200-day simple moving averages
  - `volume_ratio`: current volume divided by 20-day trailing mean
  - `historical_volatility`: 30-day annualized realized volatility of log returns
  - `return_1d`, `return_5d`: 1-day and 5-day trailing close returns
- **Isolation**: All transformers (e.g., `StandardScaler` for logistic regression) are fitted strictly on the training subset of each fold.

---

## 4. Out-of-Sample Performance Results

### Trading Performance (2020-10-26 to 2026-07-02, 1,427 OOS Sessions)
Starting capital $100,000, 5 bps fee, 2 bps adverse slippage:

| Asset | Strategy | Total Return | Ann. Return | Volatility | Sharpe (Rf=0) | Max Drawdown | Avg Exposure | Beta vs Asset |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **QQQ** | Executable Buy & Hold | 161.76% | 18.54% | 22.52% | 0.8678 | -35.12% | 99.93% | 0.9999 |
| | Technical Indicator | 62.10% | 8.91% | 10.27% | 0.8826 | -14.13% | 49.34% | 0.2941 |
| | Logistic Regression | 79.51% | 10.89% | 19.14% | 0.6356 | -33.92% | 71.28% | 0.8014 |
| | Random Forest | 59.98% | 8.66% | 11.55% | 0.7759 | -13.15% | 30.11% | 0.3641 |
| | XGBoost | 95.16% | 12.54% | 17.15% | 0.7746 | -32.44% | 62.98% | 0.6299 |
| **SPY** | Executable Buy & Hold | 136.92% | 16.47% | 16.83% | 0.9901 | -24.50% | 99.93% | 1.0000 |
| | Technical Indicator | 29.50% | 4.67% | 7.51% | 0.6458 | -9.29% | 51.26% | 0.2883 |
| | Logistic Regression | 67.10% | 9.50% | 15.06% | 0.6776 | -24.95% | 78.45% | 0.8466 |
| | Random Forest | 11.19% | 1.89% | 6.50% | 0.3209 | -11.69% | 28.49% | 0.2853 |
| | XGBoost | 48.07% | 7.18% | 13.11% | 0.5943 | -24.61% | 65.78% | 0.6641 |
| **NVDA** | Executable Buy & Hold | 1369.54% | 60.79% | 50.74% | 1.1880 | -66.34% | 99.93% | 0.9998 |
| | Technical Indicator | 136.94% | 16.47% | 26.33% | 0.7090 | -32.67% | 43.54% | 0.3785 |
| | Logistic Regression | 310.46% | 28.34% | 35.36% | 0.8806 | -32.64% | 56.15% | 0.5766 |
| | Random Forest | 277.64% | 26.47% | 19.01% | 1.3296 | -13.74% | 29.33% | 0.2534 |
| | XGBoost | 946.42% | 51.43% | 34.78% | 1.3639 | -34.12% | 53.28% | 0.5323 |

*Frictionless close-to-close buy-and-hold returns over the same window were: QQQ: 162.91%, SPY: 137.34%, NVDA: 1389.01%.*

### Classification Diagnostics Across 1,427 OOS Predictions

| Asset | Model | ROC-AUC | PR-AUC | Brier Score | Positive Base Rate | Model Accuracy | Always-Up Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| **QQQ** | Logistic Regression | 0.4598 | 0.5571 | 0.2518 | 58.86% | 59.22% | 58.86% |
| | Random Forest | 0.5348 | 0.6229 | 0.2519 | 58.86% | 51.02% | 58.86% |
| | XGBoost | 0.5205 | 0.6025 | 0.2607 | 58.86% | 56.27% | 58.86% |
| **SPY** | Logistic Regression | 0.4546 | 0.5763 | 0.2473 | 60.90% | 59.78% | 60.90% |
| | Random Forest | 0.4853 | 0.5922 | 0.2610 | 60.90% | 48.91% | 60.90% |
| | XGBoost | 0.5142 | 0.6239 | 0.2554 | 60.90% | 55.71% | 60.90% |
| **NVDA** | Logistic Regression | 0.4669 | 0.5452 | 0.2576 | 57.25% | 53.75% | 57.25% |
| | Random Forest | 0.4899 | 0.5697 | 0.2586 | 57.25% | 48.70% | 57.25% |
| | XGBoost | 0.4949 | 0.5771 | 0.2678 | 57.25% | 53.54% | 57.25% |

---

## 5. Statistical Uncertainty & Hypothesis Testing

All statistical tests utilize 1,000 resamples with a block length of 20 trading sessions to preserve autocorrelation and overlapping return structures. Multiple comparisons are corrected across models using Holm's step-down procedure:

### Classification Uncertainty (ROC-AUC vs. 0.50 Null)
- **QQQ**: Logistic AUC = 0.4598 [0.4078, 0.5100], Holm $p = 0.4316$; RF AUC = 0.5348 [0.4908, 0.5798], Holm $p = 0.4316$; XGB AUC = 0.5205 [0.4804, 0.5644], Holm $p = 0.4316$.
- **SPY**: Logistic AUC = 0.4546 [0.4077, 0.5033], Holm $p = 0.2757$; RF AUC = 0.4853 [0.4384, 0.5315], Holm $p = 1.0000$; XGB AUC = 0.5142 [0.4661, 0.5623], Holm $p = 1.0000$.
- **NVDA**: Logistic AUC = 0.4669 [0.4147, 0.5225], Holm $p = 0.7912$; RF AUC = 0.4899 [0.4420, 0.5394], Holm $p = 1.0000$; XGB AUC = 0.4949 [0.4490, 0.5446], Holm $p = 1.0000$.

*Conclusion: None of the machine learning models demonstrated statistically significant directional ranking discrimination over a random baseline after multiple-testing correction.*

### Return Differences vs. Benchmarks (Paired Block Sign-Permutation Test)
- **Active Models vs. Executable Buy & Hold**:
  - On `SPY`, all active models significantly underperformed buy-and-hold (Holm $p = 0.028$ for all models).
  - On `NVDA`, indicator ($p = 0.008$) and logistic ($p = 0.039$) significantly underperformed buy-and-hold; XGBoost underperformed but was not statistically significant ($p = 0.349$).
  - On `QQQ`, active models underperformed buy-and-hold with Holm $p$-values between $0.084$ and $0.254$.
- **Machine Learning vs. Indicator Baseline**:
  - `NVDA`: XGBoost vs. Indicator: mean return difference +28.76%, Holm $p = 0.048$ (statistically significant outperformance). Logistic ($p = 0.346$) and Random Forest ($p = 0.541$) were not significant.
  - `QQQ`: Logistic vs. Indicator (Holm $p = 1.000$), Random Forest vs. Indicator (Holm $p = 1.000$), XGBoost vs. Indicator (Holm $p = 1.000$).
  - `SPY`: Logistic vs. Indicator (Holm $p = 0.665$), Random Forest vs. Indicator (Holm $p = 0.665$), XGBoost vs. Indicator (Holm $p = 0.665$).

---

## 6. Important Conceptual Distinctions

- **Predictive Performance vs. Investment Performance**: A classifier can generate positive cumulative trading return despite having an ROC-AUC near 0.50 if it stays mostly invested during strong market uptrends. The return reflects beta capture rather than predictive edge.
- **Gross vs. Net Performance**: Fills at next-open prices with 5 bps commissions and 2 bps slippage reduced terminal wealth across all strategies relative to frictionless close-to-close metrics.
- **Statistical vs. Economic Significance**: On `QQQ`, XGBoost beat the indicator strategy by +33.06% in raw total return, but with a Holm-adjusted $p$-value of $1.000$, this difference is statistically indistinguishable from noise under the block sign-permutation null.
- **Historical Backtesting vs. Future Performance**: Out-of-sample results reflect empirical behavior during a historically strong technology bull market (2020–2026). They do not guarantee performance in stagflationary, sideways, or prolonged bear regimes.

---

## 7. Limitations & Threats to Validity

1. **Next-Open Execution**: Fills are assumed at open prices without modeling bid-ask spread variations, limit-order queue priority, or opening auction volume limits.
2. **Adjusted Price Series**: Uses split- and dividend-adjusted closes from Yahoo Finance. Dividends are implicitly reinvested in the price series rather than credited as cash.
3. **Daily Granularity**: Strategies evaluate once daily at market close and rebalance at the next open. Intraday volatility, stop-loss orders, and margin requirements are not modeled.
4. **Target Horizon Mismatch**: Models forecast 5-day directional movement, while portfolio rebalancing occurs daily based on threshold transitions.
5. **Asset Selection**: The universe consists of large-cap US equities with massive liquidity. Results cannot be extrapolated to small-cap, emerging, or illiquid markets.

---

## 8. Installation, Reproduction & Tests

### Setup Environment
```bash
git clone https://github.com/dsaw2e2/Backtesting-engine.git
cd Backtesting-engine

python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Reproduce Experiments

Run single-strategy backtests:
```bash
python main.py --csv data/QQQ.csv --ticker QQQ --strategy indicator
python main.py --csv data/QQQ.csv --ticker QQQ --strategy xgboost
```

Reproduce full 23-fold walk-forward research suites:
```bash
python paper_research.py --csv data/QQQ.csv --ticker QQQ
python paper_research.py --csv data/SPY.csv --ticker SPY
python paper_research.py --csv data/NVDA.csv --ticker NVDA
```

### Run Automated Verification Suite
The test suite consists of 26 deterministic unit tests validating next-open timing, purge gaps, feature isolation, and portfolio accounting:
```bash
python -m unittest discover -s tests -v
```

---

## License

MIT
