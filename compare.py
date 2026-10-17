"""Refresh the cross-model comparison table from saved backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_research.comparison import export_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved model returns with buy and hold.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    csv_path, markdown_path = export_comparison(args.output_dir)
    print(f"Comparison CSV: {csv_path.resolve()}")
    print(f"Comparison report: {markdown_path.resolve()}")


if __name__ == "__main__":
    main()

