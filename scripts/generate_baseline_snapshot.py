#!/usr/bin/env python3
"""
Generate the golden baseline trades snapshot for the short-timespan test.

Runs baseline config on 2012-01-01 to 2012-12-31, then writes the trades
DataFrame (CSV only, no metadata) to tests/snapshots/baseline_trades_short.csv.

Run once after `make download` to create or refresh the snapshot. Used by
test_baseline_trades_snapshot to ensure indicators and evaluation have not regressed.

Usage:
    make baseline-snapshot-generate
    # or, from project root:
    python scripts/generate_baseline_snapshot.py
"""
import sys
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.signals.config_loader import load_config_from_yaml
from core.evaluation.walk_forward import WalkForwardEvaluator
from core.grid_test.reporter import trades_to_dataframe

SNAPSHOT_START = "2012-01-01"
SNAPSHOT_END = "2012-12-31"
BASELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"
SNAPSHOT_PATH = PROJECT_ROOT / "tests" / "snapshots" / "baseline_trades_short.csv"


def main() -> int:
    config = load_config_from_yaml(str(BASELINE_CONFIG_PATH))
    config.start_date = SNAPSHOT_START
    config.end_date = SNAPSHOT_END

    evaluator = WalkForwardEvaluator(
        lookback_days=config.lookback_days,
        step_days=config.step_days,
    )
    result = evaluator.evaluate_multi_instrument(config, verbose=True)

    df = trades_to_dataframe(result)
    if df.empty:
        print("No trades produced; snapshot would be empty.", file=sys.stderr)
        return 1

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SNAPSHOT_PATH, index=False)
    print(f"Wrote {len(df)} trades to {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
