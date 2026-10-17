"""Run the complete paper-grade walk-forward research protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_research.market_data import load_csv
from quant_research.paper_research import (
    PaperResearchConfig,
    export_paper_comparison,
    run_paper_research,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run purged expanding-window research with all models and baselines."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Local Yahoo or OHLCV CSV")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--min-train-rows", type=int, default=756)
    parser.add_argument("--test-rows", type=int, default=63)
    parser.add_argument("--purge-rows", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=1_000)
    parser.add_argument("--block-rows", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PaperResearchConfig(
        min_train_rows=args.min_train_rows,
        test_rows=args.test_rows,
        purge_rows=args.purge_rows,
        statistical_resamples=args.resamples,
        statistical_block_rows=args.block_rows,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        risk_free_rate=args.risk_free_rate,
    )
    market_data = load_csv(args.csv)
    output = run_paper_research(
        market_data,
        ticker=args.ticker,
        output_root=args.output_dir,
        source_path=args.csv,
        config=config,
    )
    _, comparison = export_paper_comparison(args.output_dir)
    print(f"Paper research complete: {args.ticker}")
    print(f"Results: {output.resolve()}")
    print(f"Report: {(output / 'report.md').resolve()}")
    print(f"Methodology: {(output / 'methodology.md').resolve()}")
    print(f"Cross-asset comparison: {comparison.resolve()}")


if __name__ == "__main__":
    main()
