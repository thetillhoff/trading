#!/usr/bin/env python3
"""
Grid search CLI.

Runs grid search over strategy configurations and generates comparison reports.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add core to path
core_dir = Path(__file__).parent.parent / "core"
sys.path.insert(0, str(core_dir.parent))

from core.data.loader import DataLoader
from core.signals.config import StrategyConfig, generate_grid_configs
from core.signals.detector import SignalDetector
from core.evaluation.walk_forward import WalkForwardEvaluator, WalkForwardResult
from core.grid_test.reporter import ComparisonReporter


def evaluate_config(config: StrategyConfig, data: pd.Series, start_date, end_date, step_days: int) -> WalkForwardResult:
    """Evaluate a single configuration (for parallel execution)."""
    evaluator = WalkForwardEvaluator(
        lookback_days=config.lookback_days,
        step_days=step_days,
    )
    return evaluator.evaluate(
        data,
        config,
        start_date=start_date,
        end_date=end_date,
        verbose=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run grid search over strategy configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Grid search on DJIA
    python -m cli.grid_search --instrument djia

    # Grid search on specific time range
    python -m cli.grid_search --instrument sp500 --start-date 2010-01-01 --end-date 2020-01-01

    # With charts and CSV
    python -m cli.grid_search --instrument djia --charts --csv
        """
    )
    
    parser.add_argument(
        "--instrument", "-i",
        default="djia",
        help="Instrument to evaluate (default: djia)",
    )
    parser.add_argument(
        "--start-date", "-s",
        type=str,
        help="Start date for evaluation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", "-e",
        type=str,
        help="End date for evaluation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--column",
        default="Close",
        help="Price column to use (default: Close)",
    )
    parser.add_argument(
        "--charts",
        action="store_true",
        help="Generate visualization charts",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Save results to CSV",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run evaluations in parallel",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: current directory)",
    )
    args = parser.parse_args()
    
    # Load data
    try:
        loader = DataLoader.from_scraper(args.instrument)
        df = loader.load(
            start_date=args.start_date,
            end_date=args.end_date,
        )
        data = df[args.column] if args.column in df.columns else df.iloc[:, 0]
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    print(f"Loaded {len(data)} data points from {data.index.min()} to {data.index.max()}")
    
    # Generate grid configurations
    print("\nRunning grid search...")
    configs = generate_grid_configs(
        name_prefix="grid",
        include_parameter_variations=True,
    )
    print(f"  Generated {len(configs)} configurations")
    
    # Parse dates
    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None
    
    # Use daily evaluation (step_days=1) for maximum accuracy
    step_days = 1
    
    # Run evaluations
    results = []
    if args.parallel:
        workers = args.workers or (multiprocessing.cpu_count() - 1)
        print(f"  Running {len(configs)} configurations in parallel with {workers} workers...")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate_config, config, data, start_date, end_date, step_days): config
                for config in configs
            }
            
            completed = 0
            for future in as_completed(futures):
                config = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if completed % 10 == 0:
                        print(f"  Progress: {completed}/{len(configs)} ({completed*100//len(configs)}%)")
                except Exception as e:
                    print(f"  Error evaluating {config.name}: {e}")
    else:
        print(f"  Running {len(configs)} configurations sequentially...")
        evaluator = WalkForwardEvaluator(step_days=step_days)
        for i, config in enumerate(configs):
            result = evaluator.evaluate(
                data,
                config,
                start_date=start_date,
                end_date=end_date,
                verbose=False,
            )
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(configs)} ({(i+1)*100//len(configs)}%)")
    
    print(f"  Completed {len(results)}/{len(configs)} configurations")
    
    # Generate reports
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    reporter = ComparisonReporter(output_dir=str(output_dir))
    
    if args.charts:
        reporter.generate_comparison_chart(results, filename_prefix="backtest_comparison")
        reporter.generate_dimension_charts(results, filename_prefix="grid_dimension")
        print(f"\nCharts saved to {output_dir}")
    
    if args.csv:
        csv_path = reporter.save_results_csv(results)
        param_csv_path = reporter.save_parameter_sensitivity_csv(results)
        print(f"\nCSV files saved:")
        if csv_path:
            print(f"  {csv_path}")
        if param_csv_path:
            print(f"  {param_csv_path}")
    
    # Print summary
    reporter.print_summary(results, top_n=10)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
