#!/usr/bin/env python3
"""
Grid search CLI.

Runs grid search over strategy configurations and generates comparison reports.
Uses a temp dir on disk for configs and per-instrument data so workers load on demand
and memory stays bounded (avoids OOM on large grids).
"""
import argparse
import pickle
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
import multiprocessing
from typing import Dict, List, Tuple
from core.signals.config_loader import load_config_from_yaml
from core.data.preparation import prepare_and_validate, DataPreparationError, VerifiedDataPrepResult
from core.evaluation.walk_forward import WalkForwardResult
from core.grid_test.analysis import analyze_results_dir


def _safe_config_dirname(name: str) -> str:
    """Safe filesystem directory name for config (e.g. grid_mtfon_ema8_filter_cert0.7)."""
    return re.sub(r"[^\w\-.]", "_", str(name))


def main():
    parser = argparse.ArgumentParser(
        description="Run grid search over strategy configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Grid search on DJIA (dates come from each config YAML)
    python -m cli.grid_search --instrument djia

    # Analyze only (e.g. after hypothesis runs): write analysis_report.md and CSVs into DIR
    python -m cli.grid_search --analyze results/hypothesis_tests_YYYYMMDD_HHMMSS
        """
    )
    
    parser.add_argument(
        "--instrument", "-i",
        default=None,
        help="Override config instruments with this one (e.g. djia). If omitted, each config uses its own instruments/dates.",
    )
    parser.add_argument(
        "--column",
        default="Close",
        help="Price column to use (default: Close)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto = CPU count, or 1 if 1 CPU)",
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
    parser.add_argument(
        "--progress-file",
        type=str,
        help="Write progress updates to JSON file (e.g., results/progress.json). Must be in project directory to be visible on host.",
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
    
    # Apply CLI instrument override only when explicitly passed (otherwise each config keeps its own instruments)
    if args.instrument is not None:
        print(f"  Using CLI instrument (overrides config): {args.instrument}")
        for config in configs:
            config.instruments = [args.instrument]

    # Validate: configs with no instruments get default (e.g. list_available_tickers() may have returned [])
    for config in configs:
        if not config.instruments:
            config.instruments = ["djia"]  # Fallback when no instruments in config and no CLI override
        if not config.start_date:
            print(f"Warning: Config '{config.name}' missing start_date, using 2000-01-01")
            config.start_date = "2000-01-01"
        if not config.end_date:
            print(f"Warning: Config '{config.name}' missing end_date, using 2020-01-01")
            config.end_date = "2020-01-01"
    
    # Grid search requires identical date range and eval params across all configs
    ref = configs[0]
    for config in configs[1:]:
        if config.start_date != ref.start_date or config.end_date != ref.end_date:
            print(f"Error: Config '{config.name}' has different date range ({config.start_date}–{config.end_date}) "
                  f"than '{ref.name}' ({ref.start_date}–{ref.end_date}). Grid search requires the same range for all.")
            return 1
        if config.lookback_days != ref.lookback_days or config.step_days != ref.step_days:
            print(f"Error: Config '{config.name}' has different lookback_days/step_days than '{ref.name}'. "
                  f"Grid search requires the same evaluation params for all.")
            return 1
    
    # Union of instruments across all configs; single data prep for the whole grid
    all_instruments = sorted(set(inst for c in configs for inst in c.instruments))
    
    print(f"\nTesting {len(configs)} configurations...")
    print()
    
    _cpus = multiprocessing.cpu_count()
    workers = args.workers if args.workers is not None else max(
        1, _cpus if _cpus and _cpus > 0 else 1
    )
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    print(f"Step 1: Data preparation (once for all configs, {len(all_instruments)} instruments)...")
    t0_prep = time.perf_counter()
    try:
        prep_result = prepare_and_validate(
            instruments=all_instruments,
            start_date=ref.start_date,
            end_date=ref.end_date,
            lookback_days=ref.lookback_days,
            step_days=ref.step_days,
            min_history_days=100,
            column=ref.column,
        )
    except DataPreparationError as e:
        print(f"  Data prep failed: {e}")
        return 1
    prep_elapsed = time.perf_counter() - t0_prep
    print(f"  Data prep complete in {prep_elapsed:.1f}s ({len(prep_result.instruments)} instruments validated)")
    
    # Per-config instrument list: intersection of config.instruments with validated list
    valid_set = set(prep_result.instruments)
    config_instruments: Dict[str, list] = {}
    for config in configs:
        inst_list = [i for i in config.instruments if i in valid_set]
        if not inst_list:
            print(f"Warning: Config '{config.name}' has no instruments with data in range; skipping.")
            continue
        config_instruments[config.name] = inst_list
    configs = [c for c in configs if c.name in config_instruments]
    if not configs:
        print("Error: No configs have instruments with data in the requested range")
        return 1
    
    # Per-config prep view (same dates/eval_dates, filtered instruments) for downstream code
    prep_results: Dict[str, VerifiedDataPrepResult] = {}
    prep_share = prep_elapsed / len(configs) if configs else 0.0
    for config in configs:
        prep_results[config.name] = VerifiedDataPrepResult(
            eval_dates=prep_result.eval_dates,
            start_date=prep_result.start_date,
            end_date=prep_result.end_date,
            load_start=prep_result.load_start,
            instruments=config_instruments[config.name],
            min_history_days=prep_result.min_history_days,
        )
    prep_timings = {c.name: prep_share for c in configs}

    # Pipeline: Data -> Indicators (deduplicated) -> per-config Signals -> Simulation -> Outputs -> GridReport
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        summary_dir = Path(args.output_dir)
        results_root = summary_dir
    else:
        summary_dir = Path("results") / f"grid_search_{timestamp}"
        results_root = Path("results")
    summary_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="grid_search_"))
    
    # Checkpoint and progress paths
    checkpoint_dir = workspace / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "execution.json"
    progress_file = summary_dir / "progress.json"
    
    print(f"  Progress file: {progress_file}")
    print(f"  Monitor with: watch -n 1 cat {progress_file}")
    
    try:
        print(f"\nStep 2: Building execution plan (TaskGraph)...")
        from core.orchestration.orchestrator import (
            build_data_task_graph,
            build_grid_search_task_graph,
            run_tasks,
        )
        
        # Build data task
        data_graph = build_data_task_graph(
            workspace, ref, min_history_days=100, instruments=prep_result.instruments
        )
        
        print(f"  Executing data preparation...")
        run_tasks(
            data_graph,
            verbose=True,
            max_workers=1,  # Data prep is single task
            cache_enabled=True,
        )
        
        print(f"\nStep 3: Building full execution graph...")
        grid_graph = build_grid_search_task_graph(
            workspace,
            configs,
            results_root,
            summary_dir,
            config_file_map,
        )
        
        stats = grid_graph.get_stats()
        levels = grid_graph.get_topological_levels()
        print(f"  Total tasks: {stats['total']}")
        print(f"  Execution levels: {len(levels)}")
        print(f"  Max parallelism: {max(len(level) for level in levels if level)}")
        
        print(f"\nExecuting grid search with parallel execution and caching...")
        run_tasks(
            grid_graph,
            verbose=True,
            max_workers=workers,
            cache_enabled=True,
            checkpoint_path=checkpoint_path,
            progress_file=progress_file,
        )
        
        # Load results from workspace for baseline update (before workspace is removed)
        from core.orchestration.contract import config_result_path
        results_for_baseline = []
        for c in configs:
            path = config_result_path(workspace, c.name)
            if path.exists():
                with open(path, "rb") as f:
                    results_for_baseline.append(pickle.load(f))
        if results_for_baseline:
            update_baseline_config(results_for_baseline)
    finally:
        import shutil
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
    
    print(f"\nCompleted {len(configs)} configurations. Per-config outputs: {results_root}/<rel_path>/")
    print(f"  Summary directory: {summary_dir}")
    print(f"  Grid charts and analysis_report: {summary_dir}")
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
    print()
    print("To update baseline manually, edit:")
    print("  - core/signals/config.py (BASELINE_CONFIG)")
    print("  - core/shared/defaults.py (if parameter values changed)")
    print("="*80)


if __name__ == "__main__":
    sys.exit(main())
