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
from core.signals.config_loader import load_config_from_yaml
from core.signals.detector import SignalDetector
from core.evaluation.walk_forward import WalkForwardEvaluator, WalkForwardResult
from core.grid_test.reporter import ComparisonReporter
from core.grid_test.analysis import analyze_results_dir
from datetime import datetime


def evaluate_config(config: StrategyConfig) -> WalkForwardResult:
    """Evaluate a single configuration (for parallel execution)."""
    evaluator = WalkForwardEvaluator(
        lookback_days=config.lookback_days,
        step_days=config.step_days,
    )
    # Use config's instruments and dates
    return evaluator.evaluate_multi_instrument(config, verbose=False)


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

    # Analyze only (e.g. after hypothesis runs): write analysis_report.md and CSVs into DIR
    python -m cli.grid_search --analyze results/hypothesis_tests_YYYYMMDD_HHMMSS
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
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: current directory)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Load all YAML configs from directory (recursive, default: configs/)",
    )
    parser.add_argument(
        "--analyze",
        type=str,
        metavar="DIR",
        help="Run analysis only on an existing results dir (skips grid search). For multi-period dirs (e.g. hypothesis_tests_*), writes analysis_report.md and CSVs into DIR.",
    )
    args = parser.parse_args()

    # Analyze-only mode: run core analysis on DIR and exit
    if args.analyze:
        results_dir = Path(args.analyze)
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
            return 1
        ok = analyze_results_dir(results_dir, verbose=True)
        return 0 if ok else 1

    # Print grid search header
    print("=" * 80)
    print("GRID SEARCH - MULTI-STRATEGY COMPARISON")
    print("Mode: Testing multiple configurations in parallel")
    print("=" * 80)
    print()
    
    # Load configurations from directory (default: configs/)
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return 1
    
    yaml_files = list(config_dir.rglob("*.yaml"))
    if not yaml_files:
        print(f"Error: No YAML config files found in {config_dir}")
        return 1
    
    print(f"\nLoading configs from {config_dir}...")
    
    configs = []
    config_file_map = {}  # Map config to its source file path for result directory structure
    for yaml_file in yaml_files:
        try:
            config = load_config_from_yaml(yaml_file)
            configs.append(config)
            # Store relative path from configs/ for result directory structure
            try:
                rel_path = yaml_file.relative_to(config_dir)
                # Store as: optimization/ew_all_indicators_wave_001 (without .yaml)
                result_path = rel_path.parent / rel_path.stem
                config_file_map[config.name] = result_path
                # Also store in config for reporter access
                config._source_path = result_path
            except ValueError:
                # If config_dir is not parent, use config name as fallback
                config_file_map[config.name] = Path(config.name)
                config._source_path = Path(config.name)
        except Exception as e:
            print(f"  Warning: Failed to load {yaml_file}: {e}")
    
    print(f"  Loaded {len(configs)} configurations from {len(yaml_files)} files")
    
    # Apply CLI arguments to YAML-loaded configs (CLI takes precedence over config values)
    if args.instrument or args.start_date or args.end_date:
        print(f"  Using CLI arguments (takes precedence over config values):")
        if args.instrument:
            print(f"    Instrument: {args.instrument}")
        if args.start_date:
            print(f"    Start date: {args.start_date}")
        if args.end_date:
            print(f"    End date: {args.end_date}")
        
        for config in configs:
            if args.instrument:
                config.instruments = [args.instrument]
            if args.start_date:
                config.start_date = args.start_date
            if args.end_date:
                config.end_date = args.end_date
    
    # Validate all configs have required fields (use defaults only if not set)
    for config in configs:
        if not config.instruments:
            config.instruments = ["djia"]  # Default
        if not config.start_date:
            # Only warn if CLI didn't provide it and config doesn't have it
            if not args.start_date:
                print(f"Warning: Config '{config.name}' missing start_date, using 2000-01-01")
            config.start_date = args.start_date or "2000-01-01"
        if not config.end_date:
            # Only warn if CLI didn't provide it and config doesn't have it
            if not args.end_date:
                print(f"Warning: Config '{config.name}' missing end_date, using 2020-01-01")
            config.end_date = args.end_date or "2020-01-01"
    
    print(f"\nTesting {len(configs)} configurations...")
    print()
    
    # Run evaluations in parallel (always enabled)
    workers = args.workers
    
    # Set multiprocessing start method for Docker/macOS compatibility
    # 'spawn' works better in Docker containers than 'fork'
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    
    print(f"Running {len(configs)} configurations in parallel with {workers} workers...")
    print(f"  Using {multiprocessing.cpu_count()} available CPUs")
    
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_config, config): config
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
    
    print(f"\nCompleted {len(results)}/{len(configs)} configurations")
    
    # Create grid search summary directory (use --output-dir when provided for hypothesis-test flows)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        summary_dir = Path(args.output_dir)
    else:
        summary_dir = Path("results") / f"grid_search_{timestamp}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-config results to matching results/ structure
    print(f"\nSaving per-config results...")
    for result in results:
        config = result.config
        # Get relative path from configs/ for this config
        rel_path = getattr(config, '_source_path', None) or config_file_map.get(config.name, Path(config.name))
        result_dir = summary_dir / rel_path if args.output_dir else Path("results") / rel_path
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual config results with charts
        config_reporter = ComparisonReporter(output_dir=str(result_dir))
        config_reporter.save_results_csv([result], filename=f"backtest_results_{timestamp}.csv")
        
        # Generate comparison chart for this config (compares to baseline if available, or shows vs buy-and-hold)
        baseline_result = next((r for r in results if 'baseline' in r.config.name.lower()), None)
        if baseline_result and baseline_result != result:
            # Compare this config to baseline
            config_reporter.generate_comparison_chart([result, baseline_result], filename=f"comparison_{timestamp}")
        else:
            # Single config - generate comparison chart anyway (will show vs buy-and-hold)
            config_reporter.generate_comparison_chart([result], filename=f"comparison_{timestamp}")
    
    # Generate reports in summary directory (always enabled)
    reporter = ComparisonReporter(output_dir=str(summary_dir))
    
    # Always generate charts
    print(f"\nGenerating visualizations...")
    reporter.generate_comparison_chart(results, filename=f"backtest_comparison_{timestamp}")
    reporter.generate_dimension_charts(results, filename_prefix=f"grid_dimension_{timestamp}")
    
    # Generate new visualizations
    reporter.generate_multi_strategy_equity_curve(results, filename_prefix=f"equity_curve_vs_2pa_{timestamp}")
    reporter.generate_performance_by_instrument(results, filename_prefix=f"performance_by_instrument_{timestamp}")
    
    # Always save CSV results
    csv_path = reporter.save_results_csv(results, filename=f"backtest_results_{timestamp}.csv")
    param_csv_path = reporter.save_parameter_sensitivity_csv(results, filename_prefix=f"parameter_sensitivity_{timestamp}")
    
    # Generate analysis report
    reporter.generate_analysis_report(results, filename=f"analysis_report_{timestamp}.md")
    
    # Run core CSV-based analysis (analysis_report.md, all_results_combined.csv, alpha_pivot)
    print("\nRunning CSV analysis...")
    analyze_results_dir(summary_dir, verbose=True)
    
    print(f"\nResults saved:")
    print(f"  Summary directory: {summary_dir}")
    print(f"  Per-config results: results/{{subdirectory}}/")
    print(f"  Charts: {summary_dir}")
    print(f"  CSV files: {summary_dir}")
    print(f"  Analysis report: {summary_dir}/analysis_report_{timestamp}.md")
    
    # Print summary
    reporter.print_summary(results, top_n=10)
    
    return 0


def update_baseline_config(results: list):
    """Find best config by alpha and update baseline in config.py"""
    # Find best result by alpha
    best_result = max(results, key=lambda r: r.active_alpha)
    best_config = best_result.config
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION BY ALPHA")
    print("="*80)
    print(f"Name: {best_config.name}")
    print(f"Alpha: {best_result.active_alpha:.2f}%")
    print(f"Win Rate: {best_result.simulation.win_rate:.1f}%")
    print(f"Trades: {best_result.simulation.total_trades}")
    print(f"Expectancy: {best_result.simulation.expectancy_pct:.2f}%")
    print()
    print("Indicators:")
    print(f"  Elliott Wave: {best_config.use_elliott_wave}")
    if best_config.use_elliott_wave:
        print(f"    - min_confidence: {best_config.min_confidence}")
        print(f"    - min_wave_size: {best_config.min_wave_size}")
    print(f"  RSI: {best_config.use_rsi}")
    if best_config.use_rsi:
        print(f"    - period: {best_config.rsi_period}")
        print(f"    - oversold: {best_config.rsi_oversold}")
        print(f"    - overbought: {best_config.rsi_overbought}")
    print(f"  EMA: {best_config.use_ema}")
    if best_config.use_ema:
        print(f"    - short_period: {best_config.ema_short_period}")
        print(f"    - long_period: {best_config.ema_long_period}")
    print(f"  MACD: {best_config.use_macd}")
    if best_config.use_macd:
        print(f"    - fast: {best_config.macd_fast}")
        print(f"    - slow: {best_config.macd_slow}")
        print(f"    - signal: {best_config.macd_signal}")
    print()
    print("Risk Management:")
    print(f"  risk_reward: {best_config.risk_reward}")
    print(f"  position_size_pct: {best_config.position_size_pct}")
    print(f"  max_positions: {best_config.max_positions}")
    print(f"  confidence_size_multiplier: {best_config.confidence_size_multiplier}")
    print()
    print("To update baseline manually, edit:")
    print("  - core/signals/config.py (BASELINE_CONFIG)")
    print("  - core/shared/defaults.py (if parameter values changed)")
    print("="*80)


if __name__ == "__main__":
    sys.exit(main())
