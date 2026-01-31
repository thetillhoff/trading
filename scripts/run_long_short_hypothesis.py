#!/usr/bin/env python3
"""
Long vs Short trade performance: load baseline trades CSV and print breakdown.

Uses existing results (e.g. from make evaluate). Loads trades_full_*.csv,
filters to closed trades with cost_basis > 0, groups by signal_type (buy/sell),
and prints count, win rate, total PnL, avg PnL %, avg win %, avg loss %.

Usage:
    make hypothesis-long-short
    # or, from project root with Docker:
    docker compose run --rm cli python scripts/run_long_short_hypothesis.py
    docker compose run --rm cli python scripts/run_long_short_hypothesis.py results/baseline/trades_full_20260128_145318.csv
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

from core.evaluation.trade_analysis import aggregate_trades_dataframe_by_signal_type

RESULTS_BASELINE = PROJECT_ROOT / "results" / "baseline"


def _find_latest_trades_csv() -> Path:
    candidates = list(RESULTS_BASELINE.glob("trades_full_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No trades_full_*.csv found in {RESULTS_BASELINE}. Run `make evaluate` first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_trades_csv(path: Path) -> pd.DataFrame:
    """Load trades CSV; skip comment lines (first non-# line is header)."""
    from io import StringIO
    lines = path.read_text().splitlines()
    data_start = next(i for i, line in enumerate(lines) if line.strip() and not line.strip().startswith("#"))
    header = lines[data_start]
    rest = "\n".join(lines[data_start + 1 :])
    return pd.read_csv(StringIO(header + "\n" + rest))


def main() -> int:
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = _find_latest_trades_csv()
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    df = load_trades_csv(csv_path)
    if df.empty:
        print("No trade rows in CSV.", file=sys.stderr)
        return 1

    agg = aggregate_trades_dataframe_by_signal_type(df)
    b = agg["buy"]
    s = agg["sell"]

    print(f"Long vs Short trade performance (source: {csv_path.name})")
    print()
    print("| Metric              | Long (buy) | Short (sell) |")
    print("| ------------------- | ---------- | ------------ |")
    print(f"| Trade count         | {b['count']:<10} | {s['count']:<12} |")
    print(f"| Win rate %          | {b['win_rate_pct']:<10.2f} | {s['win_rate_pct']:<12.2f} |")
    print(f"| Total PnL           | {b['total_pnl']:<10.2f} | {s['total_pnl']:<12.2f} |")
    print(f"| Avg PnL % per trade | {b['avg_pnl_pct']:<10.2f} | {s['avg_pnl_pct']:<12.2f} |")
    print(f"| Avg win % (winners) | {b['avg_win_pct']:<10.2f} | {s['avg_win_pct']:<12.2f} |")
    print(f"| Avg loss % (losers) | {b['avg_loss_pct']:<10.2f} | {s['avg_loss_pct']:<12.2f} |")
    print()
    print(f"Total PnL % (Long):   {b['total_pnl_pct']:.2f}%")
    print(f"Total PnL % (Short):  {s['total_pnl_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
