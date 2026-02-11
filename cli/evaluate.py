#!/usr/bin/env python3
"""
Single strategy evaluation CLI.

Evaluates a trading strategy on any instrument and time range.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

from core.data.loader import DataLoader
from core.signals.config import StrategyConfig, BASELINE_CONFIG, PRESET_CONFIGS
from core.signals.config_loader import load_config_from_yaml
from core.signals.detector import SignalDetector
from core.evaluation.walk_forward import WalkForwardEvaluator
from core.evaluation.portfolio import PortfolioSimulator
from core.grid_test.reporter import ComparisonReporter
from core.shared.types import SignalType
from core.orchestration.contract import (
    TRADES_CSV,
    RESULTS_CSV,
    INDICATORS_CSV,
    CHART_ALPHA_OVER_TIME,
    CHART_VALUE_GAIN_PER_INSTRUMENT,
    CHART_SCATTER_PNL_DURATION,
    CHART_SCATTER_CONFIDENCE_RISK,
    CHART_GAIN_PER_INSTRUMENT,
    CHART_TRADES_PER_INSTRUMENT,
    CHART_INDICATOR_BEST_WORST,
    CHART_PERFORMANCE_TIMINGS,
    CHART_COMPARISON,
    CONFIG_TXT,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trading strategy on any instrument and time range",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate baseline strategy (default: sp500; run make download-baseline once)
    python -m cli.evaluate

    # Evaluate on specific time range
    python -m cli.evaluate --instrument sp500 --start-date 2020-01-01 --end-date 2024-01-01

    # Use a preset configuration
    python -m cli.evaluate --instrument sp500 --preset ema_only

    # Custom configuration
    python -m cli.evaluate --instrument sp500 --use-ema --use-macd --rsi-period 7
        """
    )
    
    parser.add_argument(
        "--instrument", "-i",
        default="sp500",
        help="Instrument to evaluate (default: sp500; run make download-baseline once)",
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
        "--preset", "-p",
        type=str,
        choices=list(PRESET_CONFIGS.keys()),
        help="Use a preset configuration",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Load configuration from YAML file (default: configs/baseline.yaml)",
    )
    parser.add_argument(
        "--column",
        default="Close",
        help="Price column to use (default: Close)",
    )
    parser.add_argument(
        "--max-timeline-trades",
        type=int,
        default=100,
        help="Maximum number of trades to show on timeline chart - shows best/worst performers (default: 100)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Save trades to CSV file (for later timeline generation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: current directory)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for multi-instrument evaluation (default: CPU-based)",
    )
    
    # Indicator enable/disable
    parser.add_argument("--use-elliott-wave", action="store_true", help="Enable Elliott Wave")
    parser.add_argument("--use-rsi", action="store_true", help="Enable RSI")
    parser.add_argument("--use-ema", action="store_true", help="Enable EMA")
    parser.add_argument("--use-macd", action="store_true", help="Enable MACD")
    
    # Elliott Wave parameters
    parser.add_argument("--min-confidence", type=float, help="Elliott Wave min confidence")
    parser.add_argument("--min-wave-size", type=float, help="Elliott Wave min wave size")
    
    # Technical indicator parameters
    parser.add_argument("--rsi-period", type=int, help="RSI period")
    parser.add_argument("--rsi-oversold", type=int, help="RSI oversold threshold")
    parser.add_argument("--rsi-overbought", type=int, help="RSI overbought threshold")
    parser.add_argument("--ema-short-period", type=int, help="EMA short period")
    parser.add_argument("--ema-long-period", type=int, help="EMA long period")
    parser.add_argument("--macd-fast", type=int, help="MACD fast period")
    parser.add_argument("--macd-slow", type=int, help="MACD slow period")
    parser.add_argument("--macd-signal", type=int, help="MACD signal period")
    
    # Signal filtering
    parser.add_argument(
        "--use-trend-filter",
        action="store_true",
        help="Only trade in direction of EMA trend (BUY when short>long, SELL when long>short)"
    )
    parser.add_argument(
        "--use-regime-detection",
        action="store_true",
        help="Enable market regime detection and adaptive signal inversion (adapts EW signals to bull/bear markets)"
    )
    
    # Position sizing
    parser.add_argument(
        "--use-confirmation-modulation",
        action="store_true",
        help="Use confirmation-based position size modulation (0 conf=skip, 1=half, 2+=double)"
    )
    
    args = parser.parse_args()
    
    # Print evaluation mode header
    print("=" * 80)
    print("SINGLE STRATEGY EVALUATION")
    print("Mode: Detailed analysis vs buy-and-hold")
    print("=" * 80)
    print()
    
    # Create configuration - default to baseline.yaml, or use preset, or build from flags
    config_based_execution = False
    config = None
    
    if args.preset:
        config = PRESET_CONFIGS[args.preset]
        print(f"Using preset configuration: {args.preset}")
    elif args.config:
        # Load from YAML file (default: baseline.yaml)
        try:
            config = load_config_from_yaml(args.config)
            print(f"Loaded configuration from: {args.config}")
            
            # Check if config specifies instruments and dates (new multi-instrument format)
            if config.instruments and config.start_date and config.end_date:
                config_based_execution = True
                print(f"  Using config-specified data:")
                print(f"    Instruments: {', '.join(config.instruments)}")
                print(f"    Date range: {config.start_date} to {config.end_date}")
                print()
        except Exception as e:
            print(f"Error loading config file: {e}")
            return 1
    
    if config is None:
        # Fallback: build from flags (backward compatibility)
        from dataclasses import asdict
        config_dict = asdict(BASELINE_CONFIG)
        config_dict['name'] = 'custom'
        config_dict['description'] = 'Custom configuration (defaults to baseline)'
        
        # Only override if flags are explicitly provided
        has_indicator_flags = (args.use_elliott_wave or args.use_rsi or 
                              args.use_ema or args.use_macd)
        
        if has_indicator_flags:
            config_dict['use_elliott_wave'] = args.use_elliott_wave
            config_dict['use_rsi'] = args.use_rsi
            config_dict['use_ema'] = args.use_ema
            config_dict['use_macd'] = args.use_macd
        
        # Add parameter overrides if provided
        if args.min_confidence is not None:
            config_dict['min_confidence'] = args.min_confidence
        if args.min_wave_size is not None:
            config_dict['min_wave_size'] = args.min_wave_size
        if args.rsi_period is not None:
            config_dict['rsi_period'] = args.rsi_period
        if args.rsi_oversold is not None:
            config_dict['rsi_oversold'] = args.rsi_oversold
        if args.rsi_overbought is not None:
            config_dict['rsi_overbought'] = args.rsi_overbought
        if args.ema_short_period is not None:
            config_dict['ema_short_period'] = args.ema_short_period
        if args.ema_long_period is not None:
            config_dict['ema_long_period'] = args.ema_long_period
        if args.macd_fast is not None:
            config_dict['macd_fast'] = args.macd_fast
        if args.macd_slow is not None:
            config_dict['macd_slow'] = args.macd_slow
        if args.macd_signal is not None:
            config_dict['macd_signal'] = args.macd_signal
        
        # New feature flags
        if args.use_trend_filter:
            config_dict['use_trend_filter'] = args.use_trend_filter
        if args.use_regime_detection:
            config_dict['use_regime_detection'] = args.use_regime_detection
        if args.use_confirmation_modulation:
            config_dict['use_confirmation_modulation'] = args.use_confirmation_modulation
            config_dict['use_confidence_sizing'] = False
        
        # Merge with baseline for defaults
        from dataclasses import fields, asdict
        baseline_dict = asdict(BASELINE_CONFIG)
        baseline_dict.update(config_dict)
        config = StrategyConfig(**baseline_dict)
    
    print(f"Evaluating strategy: {config.name}")
    print(f"  Description: {config.description}")
    print(f"  Indicators: ", end="")
    indicators = []
    if config.use_elliott_wave:
        indicators.append("Elliott Wave")
    if config.use_rsi:
        indicators.append("RSI")
    if config.use_ema:
        indicators.append("EMA")
    if config.use_macd:
        indicators.append("MACD")
    print(", ".join(indicators) if indicators else "None")
    print()
    
    # Run evaluation (config-based or CLI-based)
    evaluator = WalkForwardEvaluator(
        lookback_days=config.lookback_days,
        step_days=config.step_days,
    )
    
    used_pipeline = False
    if config_based_execution:
        # Pipeline: Data -> Indicators -> Signals -> Simulation -> Outputs (path-based, same as grid)
        import tempfile
        from pathlib import Path
        from core.orchestration.orchestrator import (
            run_tasks,
            build_data_task_graph,
            build_single_eval_task_graph,
        )

        if args.config:
            config_path = Path(args.config)
            try:
                rel = config_path.relative_to("configs")
                output_subdir = rel.parent / rel.stem
            except ValueError:
                output_subdir = Path(config.name)
            output_dir = Path("results") / output_subdir
        else:
            output_dir = Path("results") / config.name
        output_dir.mkdir(parents=True, exist_ok=True)
        workspace_root = Path(tempfile.mkdtemp(prefix="eval_"))
        try:
            print("  Pipeline: Data -> Indicators -> Signals -> Simulation -> Outputs")
            
            # Build and execute data task
            data_graph = build_data_task_graph(workspace_root, config, min_history_days=100)
            run_tasks(data_graph, verbose=True, cache_enabled=True)
            
            # Build and execute evaluation tasks
            config_yaml = Path(args.config) if args.config else None
            eval_graph = build_single_eval_task_graph(workspace_root, output_dir, config, config_yaml_path=config_yaml)
            run_tasks(eval_graph, verbose=True, cache_enabled=True)
            
            from core.orchestration.contract import config_result_path
            result_path = config_result_path(workspace_root, config.name)
            with open(result_path, "rb") as f:
                import pickle
                result = pickle.load(f)
        finally:
            import shutil
            if workspace_root.exists():
                shutil.rmtree(workspace_root, ignore_errors=True)
        instruments_used = result.instruments_used if result.instruments_used is not None else config.instruments
        data = None
        used_pipeline = True
    else:
        # Old flow: Load data from CLI args
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
        
        # Run evaluation with CLI-provided dates
        result = evaluator.evaluate(
            data,
            config,
            start_date=pd.to_datetime(args.start_date) if args.start_date else None,
            end_date=pd.to_datetime(args.end_date) if args.end_date else None,
            verbose=True,
        )
        instruments_used = [args.instrument]
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Strategy: {config.name}")
    print(f"Total Trades: {result.summary.total_trades}")
    print(f"Win Rate: {result.summary.win_rate:.1f}%")
    print(f"Avg Exposure: {result.simulation.avg_exposure_pct:.1f}%")
    print(f"Total Return: {result.simulation.total_return_pct:.2f}%")
    print(f"Alpha: {getattr(result, 'active_alpha', result.outperformance):.2f}%")
    print(f"Expectancy: {result.simulation.expectancy_pct:.2f}%")

    if used_pipeline:
        print(f"\nOutputs written to {output_dir} (trades.csv, results.csv, indicators.csv, charts)")
        return 0

    # Legacy path: create output directory and write files/charts
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Determine output directory based on config
    if config_based_execution and args.config:
        # Mirror config path under results: configs/baseline.yaml -> results/baseline/,
        # configs/abc/def.yaml -> results/abc/def/
        config_path = Path(args.config)
        try:
            rel = config_path.relative_to("configs")
            output_subdir = rel.parent / rel.stem
        except ValueError:
            output_subdir = Path(config.name)
        output_dir = Path("results") / output_subdir
    else:
        # Use instrument-based path (no config file or legacy flow)
        output_dir = Path("results") / args.instrument
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create reporter
    reporter = ComparisonReporter(output_dir=str(output_dir))
    
    # Always generate charts and CSV
    # Save indicators CSV (always generated; includes quality_factor from result when available)
    indicators_csv = evaluator.save_indicators_csv(
        output_dir, "indicators", config=config, result=result, filename=INDICATORS_CSV
    )
    if indicators_csv:
        print(f"Indicators CSV saved: {indicators_csv}")

    total_trades = result.summary.total_trades

    # Save trades and results CSVs (canonical names)
    trades_csv = reporter.save_trades_csv(result, filename=TRADES_CSV)
    if trades_csv:
        print(f"\nTrades CSV saved: {trades_csv} (all {total_trades} trades)")
    reporter.save_results_csv([result], filename=RESULTS_CSV)

    # Copy config YAML to output dir with timestamp (if available)
    if args.config:
        from datetime import datetime
        import shutil
        from core.orchestration.contract import CONFIG_YAML, timestamped_filename
        config_yaml_path = Path(args.config)
        if config_yaml_path.exists():
            timestamped_config = output_dir / timestamped_filename(CONFIG_YAML)
            shutil.copy2(config_yaml_path, timestamped_config)
            print(f"Config YAML saved: {timestamped_config}")
    
    # Save configuration summary to text file
    config_path = output_dir / CONFIG_TXT
    with open(config_path, 'w') as f:
            f.write(f"CONFIGURATION SUMMARY\n")
            f.write(f"{'='*80}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Strategy: {config.name}\n")
            if config_based_execution:
                f.write(f"Instruments: {', '.join(instruments_used)}\n")
                f.write(f"Period: {config.start_date} to {config.end_date}\n")
            else:
                f.write(f"Instrument: {args.instrument}\n")
                f.write(f"Period: {args.start_date} to {args.end_date}\n")
            f.write(f"\n")
            f.write(f"INDICATORS\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Elliott Wave: {config.use_elliott_wave}\n")
            f.write(f"RSI: {config.use_rsi}\n")
            f.write(f"EMA: {config.use_ema}\n")
            f.write(f"MACD: {config.use_macd}\n")
            f.write(f"\n")
            f.write(f"FILTERS & MODULATION\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Trend Filter: {config.use_trend_filter}\n")
            f.write(f"Confirmation Modulation: {config.use_confirmation_modulation}\n")
            f.write(f"Confidence Sizing: {config.use_confidence_sizing}\n")
            f.write(f"\n")
            f.write(f"RISK MANAGEMENT\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Position Size: {config.position_size_pct * 100}%\n")
            f.write(f"Risk:Reward: {config.risk_reward}:1\n")
            f.write(f"Max Positions: {config.max_positions}\n")
            f.write(f"Min Confidence: {config.min_confidence}\n")
            f.write(f"\n")
            f.write(f"RESULTS\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Total Trades: {result.summary.total_trades}\n")
            f.write(f"Win Rate: {result.summary.win_rate:.1f}%\n")
            f.write(f"Total Return: {result.simulation.total_return_pct:.2f}%\n")
            f.write(f"Alpha: {getattr(result, 'active_alpha', result.outperformance):.2f}%\n")
            f.write(f"Expectancy: {result.simulation.expectancy_pct:.2f}%\n")
            f.write(f"Avg Exposure: {result.simulation.avg_exposure_pct:.1f}%\n")
            f.write(f"Market Return (Buy-and-Hold): {result.buy_and_hold_gain:.2f}%\n")
            f.write(f"Hybrid Return: {result.hybrid_return:.2f}%\n")
            f.write(f"\n")
            f.write(f"FILES GENERATED\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Trades CSV: {TRADES_CSV}\n")
            f.write(f"Results CSV: {RESULTS_CSV}\n")
            f.write(f"Indicators CSV: {INDICATORS_CSV}\n")
            f.write(f"Alpha Over Time: {CHART_ALPHA_OVER_TIME}\n")
            f.write(f"Value-gain per instrument: {CHART_VALUE_GAIN_PER_INSTRUMENT}\n")
            f.write(f"Scatter P&L%% vs Duration: {CHART_SCATTER_PNL_DURATION}\n")
            f.write(f"Scatter Confidence/Risk vs P&L: {CHART_SCATTER_CONFIDENCE_RISK}\n")
            f.write(f"Gain per Instrument: {CHART_GAIN_PER_INSTRUMENT}\n")
            f.write(f"Trades per Instrument: {CHART_TRADES_PER_INSTRUMENT}\n")
            f.write(f"Indicator Best/Worst: {CHART_INDICATOR_BEST_WORST}\n")
            f.write(f"Performance timings: {CHART_PERFORMANCE_TIMINGS}\n")
            f.write(f"Comparison: {CHART_COMPARISON}\n")
    print(f"Config summary saved: {config_path}")
    
    # Ensure price data is a Series for charts
    price_series = data if isinstance(data, pd.Series) else (
        data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
    )
    # Optional MSCI World benchmark (same date range)
    benchmark_series = None
    start_date = config.start_date if config_based_execution else args.start_date
    end_date = config.end_date if config_based_execution else args.end_date
    if start_date and end_date:
        try:
            msci = DataLoader.from_instrument(
                'msci_world',
                start_date=start_date,
                end_date=end_date,
                column=config.column if config_based_execution else args.column or 'Close',
            )
            if isinstance(msci, pd.Series):
                benchmark_series = {'MSCI World': msci}
            elif msci is not None and hasattr(msci, 'columns') and len(msci.columns) > 0:
                benchmark_series = {'MSCI World': msci['Close'] if 'Close' in msci.columns else msci.iloc[:, 0]}
        except Exception:
            pass
    # Per-instrument price series for gain_per_instrument (Strategy vs B&H)
    price_data_by_instrument = {}
    instruments_for_gain = instruments_used if config_based_execution else [args.instrument]
    col = config.column if config_based_execution else (args.column or 'Close')
    for inst in instruments_for_gain:
        try:
            s = DataLoader.from_instrument(inst, start_date=start_date, end_date=end_date, column=col)
            if isinstance(s, pd.Series):
                price_data_by_instrument[inst] = s
            elif s is not None and hasattr(s, 'columns') and len(s.columns) > 0:
                price_data_by_instrument[inst] = s['Close'] if 'Close' in s.columns else s.iloc[:, 0]
        except Exception:
            pass
    # Generate charts (canonical filenames)
    alpha_path = reporter.generate_alpha_over_time(
        result,
        price_series,
        price_data_by_instrument=price_data_by_instrument if price_data_by_instrument else None,
        filename=CHART_ALPHA_OVER_TIME,
    )
    value_gain_per_inst_path = reporter.generate_value_gain_and_benchmarks(
        result,
        price_data=price_series,
        benchmark_series=benchmark_series,
        price_data_by_instrument=price_data_by_instrument if price_data_by_instrument else None,
        filename=CHART_VALUE_GAIN_PER_INSTRUMENT,
    )
    scatter_duration_path = reporter.generate_pnl_vs_duration_scatter(
        result, filename=CHART_SCATTER_PNL_DURATION,
    )
    scatter_conf_path = reporter.generate_confidence_risk_vs_pnl_scatter(
        result, filename=CHART_SCATTER_CONFIDENCE_RISK,
    )
    gain_inst_path = reporter.generate_gain_per_instrument(
        result,
        filename=CHART_GAIN_PER_INSTRUMENT,
        price_data_by_instrument=price_data_by_instrument if price_data_by_instrument else None,
    )
    trades_inst_path = reporter.generate_trades_per_instrument(
        result, filename=CHART_TRADES_PER_INSTRUMENT,
    )
    indicator_bw_path = reporter.generate_indicator_best_worst_overview(
        result, filename=CHART_INDICATOR_BEST_WORST,
    )
    perf_timings_path = ""
    if getattr(result, "performance_timings", None):
        perf_timings_path = reporter.generate_performance_timings_chart(
            result, filename=CHART_PERFORMANCE_TIMINGS,
        )
    print(f"\nCharts saved to {output_dir}")
    if alpha_path:
        print(f"  Alpha over time: {alpha_path}")
    if value_gain_per_inst_path:
        print(f"  Value-gain % per instrument over time: {value_gain_per_inst_path}")
    if scatter_duration_path:
        print(f"  P&L % vs duration: {scatter_duration_path}")
    if scatter_conf_path:
        print(f"  Confidence/risk vs P&L: {scatter_conf_path}")
    if gain_inst_path:
        print(f"  Gain per instrument: {gain_inst_path}")
    if trades_inst_path:
        print(f"  Trades per instrument: {trades_inst_path}")
    if indicator_bw_path:
        print(f"  Indicator best/worst 20%: {indicator_bw_path}")
    if perf_timings_path:
        print(f"  Performance timings: {perf_timings_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
