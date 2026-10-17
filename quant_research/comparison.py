"""Cross-experiment return comparison against an aligned buy-and-hold benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


STRATEGY_DISPLAY_NAMES = {
    "indicator": "Indicator",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
PRIMARY_STRATEGY_COLUMNS = list(STRATEGY_DISPLAY_NAMES.values())


@dataclass(frozen=True)
class ExperimentSummary:
    ticker: str
    strategy: str
    directory: Path
    history: pd.DataFrame


def _identity(directory: Path) -> tuple[str, str]:
    metadata_path = directory / "experiment.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return str(metadata["ticker"]), str(metadata["strategy"])

    # Backward compatibility for runs created before experiment.json existed.
    parts = directory.name.rsplit("_", 3)
    if len(parts) != 4:
        raise ValueError(f"Cannot identify experiment folder: {directory.name}")
    ticker, strategy, _, _ = parts
    return ticker, strategy


def discover_latest_experiments(output_root: str | Path) -> list[ExperimentSummary]:
    """Return the newest valid run for each ticker and strategy pair."""
    root = Path(output_root)
    latest: dict[tuple[str, str], ExperimentSummary] = {}
    for directory in root.iterdir() if root.exists() else []:
        history_path = directory / "portfolio_history.csv"
        metrics_path = directory / "metrics.json"
        if not directory.is_dir() or not history_path.exists() or not metrics_path.exists():
            continue
        try:
            ticker, strategy = _identity(directory)
            history = pd.read_csv(history_path, parse_dates=["date"]).sort_values("date")
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        if history.empty or not {"date", "price", "portfolio_value"}.issubset(history.columns):
            continue
        summary = ExperimentSummary(ticker, strategy, directory, history)
        key = (ticker, strategy)
        previous = latest.get(key)
        if previous is None or directory.stat().st_mtime > previous.directory.stat().st_mtime:
            latest[key] = summary
    return sorted(latest.values(), key=lambda item: (item.ticker, item.strategy))


def build_return_comparison(experiments: list[ExperimentSummary]) -> pd.DataFrame:
    """Compare strategies on the common observed period for each asset."""
    rows: list[dict[str, object]] = []
    for ticker in sorted({item.ticker for item in experiments}):
        asset_runs = [item for item in experiments if item.ticker == ticker]
        common_start = max(item.history["date"].min() for item in asset_runs)
        common_end = min(item.history["date"].max() for item in asset_runs)
        if common_start >= common_end:
            continue

        row: dict[str, object] = {
            "Asset": ticker,
            "Start": common_start.date().isoformat(),
            "End": common_end.date().isoformat(),
        }
        benchmark_history = asset_runs[0].history
        benchmark = benchmark_history[
            (benchmark_history["date"] >= common_start) & (benchmark_history["date"] <= common_end)
        ]
        row["Buy & Hold"] = benchmark["price"].iloc[-1] / benchmark["price"].iloc[0] - 1.0

        for experiment in asset_runs:
            aligned = experiment.history[
                (experiment.history["date"] >= common_start)
                & (experiment.history["date"] <= common_end)
            ]
            display_name = STRATEGY_DISPLAY_NAMES.get(
                experiment.strategy,
                experiment.strategy.replace("_", " ").title(),
            )
            row[display_name] = (
                aligned["portfolio_value"].iloc[-1] / aligned["portfolio_value"].iloc[0] - 1.0
            )
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["Asset", "Start", "End", "Buy & Hold", *PRIMARY_STRATEGY_COLUMNS]
        )
    frame = pd.DataFrame(rows)
    fixed = ["Asset", "Start", "End", "Buy & Hold"]
    for column in PRIMARY_STRATEGY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    extras = sorted(
        column
        for column in frame.columns
        if column not in fixed and column not in PRIMARY_STRATEGY_COLUMNS
    )
    strategies = [*PRIMARY_STRATEGY_COLUMNS, *extras]
    return frame[fixed + strategies]


def export_comparison(output_root: str | Path) -> tuple[Path, Path]:
    """Refresh CSV and Markdown tables summarizing all latest experiments."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    comparison = build_return_comparison(discover_latest_experiments(root))
    csv_path = root / "model_comparison.csv"
    markdown_path = root / "model_comparison.md"
    comparison.to_csv(csv_path, index=False)

    display = comparison.copy()
    return_columns = [column for column in display.columns if column not in {"Asset", "Start", "End"}]
    for column in return_columns:
        display[column] = display[column].map(lambda value: "-" if pd.isna(value) else f"{value:.2%}")
    if display.empty:
        table = "No completed experiments found."
    else:
        headers = [str(column) for column in display.columns]
        table_rows = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
        ]
        table_rows.extend(
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in display.itertuples(index=False, name=None)
        )
        table = "\n".join(table_rows)
    markdown_path.write_text(
        "# Model Return Comparison\n\n"
        "Each row uses the common date range available across that asset's latest model runs. "
        "Buy and hold uses adjusted close prices over exactly the same dates.\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    return csv_path, markdown_path
