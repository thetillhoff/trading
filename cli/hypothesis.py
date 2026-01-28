#!/usr/bin/env python3
"""
Hypothesis orchestration CLI.

Runs multi-period, multi-category hypothesis suites by delegating to
`core.orchestration.hypothesis.run_hypothesis_suite`.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so `core` is importable when run as a module
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.orchestration.hypothesis import HypothesisRunConfig, run_hypothesis_suite


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multi-period hypothesis test suites (grid-search per period + analysis).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all categories on all periods (priority periods first)
  python -m cli.hypothesis

  # Run only RSI tests on quick_test period
  python -m cli.hypothesis --category rsi_tests --period quick_test

  # Custom results directory (otherwise auto timestamped)
  python -m cli.hypothesis --results-dir results/hypothesis_tests_custom
        """,
    )

    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Hypothesis category under configs/hypothesis_tests/ (default: all)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="all",
        help="Period name (e.g. quick_test) or comma-separated list, or 'all' (default).",
    )
    parser.add_argument(
        "--instrument",
        "-i",
        type=str,
        default="djia",
        help="Instrument to evaluate (default: djia)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for grid-search (default: 8)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Parent results directory (default: results/hypothesis_tests_YYYYMMDD_HHMMSS)",
    )

    args = parser.parse_args()

    if args.period == "all":
        periods = "all"
    else:
        periods = [p.strip() for p in args.period.split(",") if p.strip()]

    cfg = HypothesisRunConfig(
        results_dir=Path(args.results_dir) if args.results_dir else None,
        instrument=args.instrument,
        workers=args.workers,
        category=args.category,
        periods=periods,
    )

    run_hypothesis_suite(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

