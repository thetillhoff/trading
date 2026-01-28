"""
Hypothesis orchestration: multi-period, multi-category grid-search suites.

This module owns the logic that used to live in `scripts/run_hypothesis_tests.sh`:
- mapping logical period names to date ranges
- selecting configs by category
- iterating periods and running grid-search per period
- running CSV-based analysis over the aggregated results directory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Union

from core.grid_test.analysis import analyze_results_dir


PeriodList = Union[List[str], Literal["all"]]


@dataclass
class HypothesisRunConfig:
    """Configuration for a multi-period hypothesis suite."""

    results_dir: Path | None = None
    instrument: str = "djia"
    workers: int = 8
    category: str = "all"
    periods: PeriodList = "all"
    priority_periods: List[str] = field(
        default_factory=lambda: ["quick_test", "recent_2yr", "full_period_20yr"]
    )
    all_periods: List[str] = field(
        default_factory=lambda: [
            "quick_test",
            "recent_2yr",
            "covid_crash",
            "recent_bull",
            "recovery_period",
            "housing_crisis",
            "dotcom_crash",
            "bear_market_long",
            "bull_market_long",
            "full_period_20yr",
        ]
    )


def get_period_dates(name: str) -> tuple[str, str] | None:
    """Map logical period name to (start_date, end_date) strings."""
    mapping = {
        "quick_test": ("2018-01-01", "2020-01-01"),
        "recent_2yr": ("2020-01-01", "2022-01-01"),
        "covid_crash": ("2019-01-01", "2021-01-01"),
        "recent_bull": ("2015-01-01", "2020-01-01"),
        "recovery_period": ("2009-01-01", "2014-01-01"),
        "housing_crisis": ("2007-01-01", "2010-01-01"),
        "dotcom_crash": ("2000-01-01", "2003-01-01"),
        "bear_market_long": ("2000-01-01", "2010-01-01"),
        "bull_market_long": ("2010-01-01", "2020-01-01"),
        "full_period_20yr": ("2000-01-01", "2020-01-01"),
    }
    return mapping.get(name)


def resolve_periods(
    selected: PeriodList, all_periods: List[str], priority_periods: List[str]
) -> List[str]:
    """Resolve which periods to run, preserving priority ordering."""
    if selected == "all":
        resolved: List[str] = list(priority_periods)
        for period in all_periods:
            if period not in resolved:
                resolved.append(period)
        return resolved

    # Allow comma-separated list for convenience
    if isinstance(selected, str):
        selected_list = [p.strip() for p in selected.split(",") if p.strip()]
    else:
        selected_list = list(selected)

    return selected_list


def _default_results_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"hypothesis_tests_{ts}"


def _resolve_config_dir(category: str) -> Path:
    base = Path("configs") / "hypothesis_tests"
    if category == "all":
        return base
    return base / category


def _run_grid_search_for_period(
    *,
    config_dir: Path,
    period_results_dir: Path,
    instrument: str,
    start_date: str,
    end_date: str,
    workers: int,
) -> int:
    """
    Run grid-search for a single period by delegating to cli.grid_search.main().

    This keeps all the heavy lifting inside the existing CLI implementation while
    allowing orchestration to be expressed in Python.
    """
    import sys
    from cli import grid_search as grid_search_cli  # type: ignore

    args = [
        "--config-dir",
        str(config_dir),
        "--workers",
        str(workers),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--instrument",
        instrument,
        "--output-dir",
        str(period_results_dir),
    ]

    old_argv = sys.argv
    try:
        sys.argv = ["cli.grid_search"] + args
        return int(grid_search_cli.main() or 0)
    finally:
        sys.argv = old_argv


def run_hypothesis_suite(cfg: HypothesisRunConfig) -> Path:
    """
    Run a full hypothesis suite across periods and categories.

    For each resolved period:
      - runs grid-search into results_dir/period_name
    After all periods:
      - runs analyze_results_dir(results_dir, verbose=True)
    """
    results_dir = cfg.results_dir or _default_results_dir()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    config_dir = _resolve_config_dir(cfg.category)
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    periods = resolve_periods(cfg.periods, cfg.all_periods, cfg.priority_periods)

    print("========================================")
    print("Hypothesis Testing Framework (Python orchestration)")
    print("========================================")
    print(f"Results directory: {results_dir}")
    print(f"Instrument: {cfg.instrument}")
    print(f"Category: {cfg.category}")
    print(f"Workers: {cfg.workers}")
    print(f"Periods: {', '.join(periods)}")
    print("")

    for period_name in periods:
        dates = get_period_dates(period_name)
        if not dates:
            print(f"Warning: Unknown period: {period_name}")
            continue
        start_date, end_date = dates
        period_results_dir = results_dir / period_name
        period_results_dir.mkdir(parents=True, exist_ok=True)

        print("")
        print("========================================")
        print(f"Period: {period_name} ({start_date} to {end_date})")
        print("========================================")
        print(f"Config directory: {config_dir}")
        print(f"Results directory: {period_results_dir}")
        print(f"Running with parallel execution ({cfg.workers} workers)...")
        print("")

        code = _run_grid_search_for_period(
            config_dir=config_dir,
            period_results_dir=period_results_dir,
            instrument=cfg.instrument,
            start_date=start_date,
            end_date=end_date,
            workers=cfg.workers,
        )
        if code == 0:
            print(f"✓ Period {period_name} completed successfully")
        else:
            print(f"✗ Period {period_name} failed with exit code {code}")

    print("")
    print("========================================")
    print("Testing Complete")
    print("========================================")
    print(f"Results saved to: {results_dir}")
    print("  - Each period has its own subdirectory with CSV results")
    print("")

    # Aggregate analysis over the full results dir
    print("Running analysis across all periods...")
    ok = analyze_results_dir(results_dir, verbose=True)
    if ok:
        print(f"✓ Analysis report and CSVs written to {results_dir}")
    else:
        print("✗ Analysis failed or no CSV results found in period subdirs")

    return results_dir

