#!/usr/bin/env python3
"""
Pre-trade analysis: do at-entry features (confirmations, certainty, trend, RSI) predict outcome?

Loads baseline trades CSV, filters to closed trades, and breaks down win rate and avg PnL %
by indicator_confirmations, certainty bins, trend_direction, and RSI zone (if present).

Usage:
    make hypothesis-pretrade
    docker compose run --rm cli python scripts/run_pretrade_hypothesis.py
    docker compose run --rm cli python scripts/run_pretrade_hypothesis.py results/baseline/trades_full_*.csv
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

from core.evaluation.trade_analysis import analyze_pretrade_predictors

RESULTS_BASELINE = PROJECT_ROOT / "results" / "baseline"


def _find_latest_trades_csv() -> Path:
    candidates = list(RESULTS_BASELINE.glob("trades_full_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No trades_full_*.csv in {RESULTS_BASELINE}. Run `make evaluate` first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_trades_csv(path: Path) -> pd.DataFrame:
    from io import StringIO
    lines = path.read_text().splitlines()
    data_start = next(
        i for i, line in enumerate(lines)
        if line.strip() and not line.strip().startswith("#")
    )
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
        print("No trade rows.", file=sys.stderr)
        return 1

    result = analyze_pretrade_predictors(df)
    if not result:
        print("No closed trades with cost_basis > 0.", file=sys.stderr)
        return 1

    print(f"Pre-trade predictors vs outcome (source: {csv_path.name})")
    print()

    for feature, groups in result.items():
        print(f"--- {feature} ---")
        print("| Group        | Count | Win rate % | Avg PnL % |")
        print("| ------------ | ----- | ---------- | --------- |")
        for name, m in sorted(groups.items(), key=lambda x: -x[1]["count"]):
            print(f"| {str(name):<12} | {m['count']:<5} | {m['win_rate_pct']:<10.2f} | {m['avg_pnl_pct']:<9.2f} |")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
