"""
Task implementations for the pipeline: Data, Indicators, Signals, Simulation, Outputs, GridReport.

Each task reads inputs from disk (paths) and writes outputs to disk (paths).
Task payloads are path references and small config identifiers; no large in-memory data.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .contract import (
    data_dir,
    prep_manifest_path,
    instrument_data_path,
    indicator_output_path,
    config_signals_path,
    safe_instrument_filename,
)
from .indicator_spec import (
    technical_spec_from_config,
    elliott_wave_spec_from_config,
    elliott_wave_inverted_spec_from_config,
)


def run_data_task(
    root: Path,
    instruments: List[str],
    start_date: str,
    end_date: str,
    lookback_days: int,
    step_days: int,
    min_history_days: int,
    column: str = "Close",
) -> Path:
    """
    Data task: gather, prepare, format data for evaluation.

    Calls prepare_and_validate, writes prep manifest and per-instrument parquet
    to root/data/. Returns path to data dir (root/data/).

    Raises:
        DataPreparationError: from prepare_and_validate if no instruments have data.
    """
    from ..data.loader import DataLoader
    from ..data.preparation import prepare_and_validate, DataPreparationError

    prep_result = prepare_and_validate(
        instruments=instruments,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        step_days=step_days,
        min_history_days=min_history_days,
        column=column,
    )
    root = Path(root)
    ddir = data_dir(root)
    ddir.mkdir(parents=True, exist_ok=True)
    manifest_path = prep_manifest_path(root)
    manifest = {
        "prep": prep_result,
        "column": column,
        "lookback_days": lookback_days,
        "step_days": step_days,
    }
    with open(manifest_path, "wb") as f:
        pickle.dump(manifest, f)
    load_start_str = prep_result.load_start.strftime("%Y-%m-%d")
    end_date_str = prep_result.end_date.strftime("%Y-%m-%d")
    for inst in prep_result.instruments:
        try:
            series = DataLoader.from_instrument(
                inst,
                start_date=load_start_str,
                end_date=end_date_str,
                column=column,
            )
            if series is not None and len(series) > 0:
                path = instrument_data_path(root, inst)
                series.to_frame().to_parquet(path, index=True)
        except Exception:
            pass
    return ddir


def load_prep_manifest(root: Path) -> Dict[str, Any]:
    """Load prep manifest from a data task output dir (root = workspace root containing data/)."""
    with open(prep_manifest_path(root), "rb") as f:
        return pickle.load(f)


def run_indicators_task(
    root: Path,
    instrument: str,
    indicator_type: str,
    spec_key: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Indicators task: compute one indicator (technical, elliott_wave, elliott_wave_inverted)
    for one instrument and write to cache.
    
    Uses disk-based cross-run caching: checks ~/.cache/trading/indicators/ first,
    computes only on cache miss, saves result to cache for future runs.

    Reads data from root/data/<instrument>.parquet (from Data task).
    Writes to root/indicator_cache/<instrument>_<spec_key>/data.pkl.
    Returns dict with path to the written data.pkl and cache hit status.
    """
    from ..indicators.disk_cache import (
        compute_indicator_cache_key,
        get_cached_indicator,
        save_cached_indicator,
    )
    
    root = Path(root)
    out_path = indicator_output_path(root, instrument, spec_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate cache key from params (not spec_key, which includes root path)
    cache_key = compute_indicator_cache_key(instrument, indicator_type, params)
    
    # Try cache first
    cached_result = get_cached_indicator(cache_key)
    if cached_result is not None:
        # Cache hit: write to expected output location and return
        with open(out_path, "wb") as f:
            pickle.dump(cached_result, f)
        return {"indicator_path": str(out_path), "_was_disk_cached": True}
    
    # Cache miss: load data and compute
    data_path = instrument_data_path(root, instrument)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found for instrument {instrument}: {data_path}")
    df = pd.read_parquet(data_path)
    series = df[df.columns[0]] if len(df.columns) > 0 else df.iloc[:, 0]
    series.index = pd.to_datetime(series.index)
    
    # Compute indicator based on type
    if indicator_type == "technical":
        from ..indicators.technical import TechnicalIndicators
        calc = TechnicalIndicators(
            rsi_period=params.get("rsi_period", 7),
            rsi_oversold=params.get("rsi_oversold", 25),
            rsi_overbought=params.get("rsi_overbought", 75),
            ema_short_period=params.get("ema_short_period", 20),
            ema_long_period=params.get("ema_long_period", 50),
            macd_fast=params.get("macd_fast", 12),
            macd_slow=params.get("macd_slow", 26),
            macd_signal=params.get("macd_signal", 12),
            atr_period=params.get("atr_period", 14),
            volatility_window=params.get("volatility_window", 20),
        )
        result = calc.calculate_all(series)
    elif indicator_type == "elliott_wave":
        from ..indicators.elliott_wave import ElliottWaveDetector
        det = ElliottWaveDetector()
        result = det.detect_waves(
            series,
            min_confidence=params.get("min_confidence", 0.65),
            min_wave_size_ratio=params.get("min_wave_size", 0.03),
        )
    elif indicator_type == "elliott_wave_inverted":
        from ..indicators.elliott_wave import ElliottWaveDetector
        det = ElliottWaveDetector()
        # Inverted: use -series (inverted price)
        inverted = -series
        result = det.detect_waves(
            inverted,
            min_confidence=params.get("min_confidence_inverted", 0.65),
            min_wave_size_ratio=params.get("min_wave_size_inverted", 0.02),
        )
    else:
        raise ValueError(f"Unknown indicator_type: {indicator_type}")
    
    # Save to disk cache for cross-run reuse
    save_cached_indicator(cache_key, result)
    
    # Write to output location for this run
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    
    return {"indicator_path": str(out_path), "_was_disk_cached": False}


def _params_for_indicator_type(config: Any, indicator_type: str) -> Dict[str, Any]:
    """Extract params dict for a given indicator type from config."""
    if indicator_type == "technical":
        return technical_spec_from_config(config)
    if indicator_type == "elliott_wave":
        return elliott_wave_spec_from_config(config)
    if indicator_type == "elliott_wave_inverted":
        return elliott_wave_inverted_spec_from_config(config)
    return {}


def run_signals_task(
    root: Path,
    config_id: str,
    instrument: str,
    config: Any = None,
    config_path: Path = None,
    date_range_start: str = None,
    date_range_end: str = None,
    year: int = None,
) -> Path:
    """
    Signals task: generate signals for (config, instrument) and write to disk.
    
    Can process a subset of dates (yearly splits) or full range if date_range not specified.

    Reads: prep manifest from root, instrument data from root/data/<instrument>.parquet,
    config from config_path (if config not provided). Writes signals to
    root/configs/<config_id>/signals/<instrument>.pkl (or <instrument>_<year>.pkl for splits).
    Returns path to that file.
    """
    from datetime import timedelta
    from ..evaluation.walk_forward import _signal_config_from_strategy
    from ..signals.detector import SignalDetector
    from ..signals.target_calculator import TargetCalculator

    root = Path(root)
    manifest = load_prep_manifest(root)
    prep = manifest["prep"]
    lookback_days = manifest["lookback_days"]
    step_days = manifest["step_days"]
    eval_dates = prep.eval_dates
    
    # Filter eval_dates by date range if specified
    if date_range_start and date_range_end:
        start_ts = pd.Timestamp(date_range_start)
        end_ts = pd.Timestamp(date_range_end)
        eval_dates = [d for d in eval_dates if start_ts <= d <= end_ts]
    
    if config is None and config_path is not None:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if config is None:
        raise ValueError("Either config or config_path must be provided")
    data_path = instrument_data_path(root, instrument)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    df = pd.read_parquet(data_path)
    data_inst = df[df.columns[0]] if len(df.columns) > 0 else df.iloc[:, 0]
    data_inst.index = pd.to_datetime(data_inst.index)
    signal_detector = SignalDetector(_signal_config_from_strategy(config))
    target_calculator = TargetCalculator(
        risk_reward_ratio=config.risk_reward,
        use_atr_stops=getattr(config, "use_atr_stops", False),
        atr_stop_multiplier=getattr(config, "atr_stop_multiplier", 2.0),
        atr_period=getattr(config, "atr_period", 14),
    )
    all_signals = []
    min_history = 30
    
    # Timing statistics
    timings = {}
    eval_count = 0
    
    for eval_date in eval_dates:
        lookback_start = eval_date - timedelta(days=lookback_days)
        historical_data = data_inst[
            (data_inst.index >= lookback_start) & (data_inst.index <= eval_date)
        ]
        if len(historical_data) < min_history:
            continue
        
        eval_count += 1
        signals, _indicator_df, all_waves = signal_detector.detect_signals_with_indicators(
            historical_data, timings=timings
        )
        if not signals:
            continue
        signals_with_targets = target_calculator.calculate_targets(
            signals,
            historical_data,
            all_waves=all_waves if getattr(config, "use_wave_relationship_targets", True) else None,
        )
        for s in signals_with_targets:
            s.instrument = instrument
        all_signals.extend(signals_with_targets)
    
    # Log timing statistics if we have Elliott Wave detector
    if signal_detector.elliott_detector and eval_count > 0:
        cache_stats = signal_detector.elliott_detector.get_cache_stats()
        print(f"\n{'='*80}")
        print(f"Signal Generation Timing for {instrument} ({eval_count} eval dates)")
        print(f"{'='*80}")
        
        # Total time per category
        total_time = sum(timings.values())
        if total_time > 0:
            print(f"\nTime Breakdown (total: {total_time:.2f}s):")
            sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
            for key, elapsed in sorted_timings:
                pct = (elapsed / total_time * 100) if total_time > 0 else 0
                avg_ms = (elapsed / eval_count * 1000) if eval_count > 0 else 0
                print(f"  {key:40s}: {elapsed:7.2f}s ({pct:5.1f}%) | avg {avg_ms:6.1f}ms/eval")
        
        # Cache statistics
        print(f"\nElliott Wave Cache Statistics:")
        print(f"  Hits:       {cache_stats['cache_hits']:6d}")
        print(f"  Misses:     {cache_stats['cache_misses']:6d}")
        print(f"  Hit Rate:   {cache_stats['hit_rate_pct']:6.1f}%")
        print(f"  Cache Size: {cache_stats['cache_size']:6d} entries")
        print(f"{'='*80}\n")
    
    # Determine output path (with year suffix for splits)
    if year is not None:
        out_path = config_signals_path(root, config_id, f"{instrument}_{year}")
    else:
        out_path = config_signals_path(root, config_id, instrument)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_signals, f)
    return out_path


def run_merge_signals_task(
    root: Path,
    config_id: str,
    instrument: str,
) -> Path:
    """
    Merge yearly signal files into a single combined file.
    
    Reads: root/configs/<config_id>/signals/<instrument>_<year>.pkl files
    Writes: root/configs/<config_id>/signals/<instrument>.pkl (merged, sorted by timestamp)
    Returns: path to merged file
    """
    import glob
    
    root = Path(root)
    signals_dir = config_signals_path(root, config_id, instrument).parent
    
    # Find all yearly signal files for this instrument
    pattern = str(signals_dir / f"{instrument}_*.pkl")
    yearly_files = sorted(glob.glob(pattern))
    
    if not yearly_files:
        raise FileNotFoundError(f"No yearly signal files found for {instrument} at {pattern}")
    
    # Load and combine all signals
    all_signals = []
    for file_path in yearly_files:
        with open(file_path, "rb") as f:
            signals = pickle.load(f)
            all_signals.extend(signals)
    
    # Sort by timestamp
    all_signals = sorted(all_signals, key=lambda s: s.timestamp)
    
    # Write merged signals
    out_path = config_signals_path(root, config_id, instrument)
    with open(out_path, "wb") as f:
        pickle.dump(all_signals, f)
    
    return out_path


def run_simulation_task(
    root: Path,
    config_id: str,
    instruments: List[str],
    config: Any = None,
    config_path: Path = None,
) -> Path:
    """
    Simulation task: merge signals from disk, run portfolio sim, write WalkForwardResult.

    Reads: prep manifest, signals from root/configs/<config_id>/signals/<instrument>.pkl
    for each instrument, price data from root/data/<instrument>.parquet.
    Writes: WalkForwardResult to root/configs/<config_id>/result.pkl. Returns that path.
    """
    from .contract import config_result_path, config_signals_path
    from ..evaluation.walk_forward import _portfolio_simulator_from_config
    from ..evaluation.walk_forward_types import WalkForwardResult

    root = Path(root)
    if config is None and config_path is not None:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if config is None:
        raise ValueError("Either config or config_path must be provided")
    manifest = load_prep_manifest(root)
    prep = manifest["prep"]
    start_date = prep.start_date
    end_date = prep.end_date
    lookback_days = manifest["lookback_days"]
    step_days = manifest["step_days"]
    all_signals = []
    for inst in instruments:
        p = config_signals_path(root, config_id, inst)
        if not p.exists():
            continue
        with open(p, "rb") as f:
            sigs = pickle.load(f)
        all_signals.extend(sigs)
    
    # Sort by timestamp first, then by confidence (DESC) to prioritize highest-quality signals
    # when timestamps match. This removes alphabetical bias.
    all_signals = sorted(all_signals, key=lambda s: (s.timestamp, -getattr(s, 'confidence', 0.0)))
    prices_by_instrument = {}
    for inst in instruments:
        dp = instrument_data_path(root, inst)
        if not dp.exists():
            continue
        df = pd.read_parquet(dp)
        series = df[df.columns[0]] if len(df.columns) > 0 else df.iloc[:, 0]
        series.index = pd.to_datetime(series.index)
        prices_by_instrument[inst] = series[
            (series.index >= start_date) & (series.index <= end_date)
        ]
    if not prices_by_instrument:
        raise ValueError(f"No price data for instruments {instruments}")
    portfolio_sim = _portfolio_simulator_from_config(config)
    simulation = portfolio_sim.simulate_strategy(
        prices_by_instrument,
        all_signals,
        start_date=start_date,
        end_date=end_date,
    )
    data_first = prices_by_instrument[instruments[0]]
    eval_data_first = data_first[
        (data_first.index >= start_date) & (data_first.index <= end_date)
    ]
    bh_simulation = portfolio_sim.simulate_buy_and_hold(
        eval_data_first,
        start_date=start_date,
        end_date=end_date,
    )
    buy_and_hold_gain = bh_simulation.total_return_pct
    avg_exposure = simulation.avg_exposure_pct / 100.0
    exposure_adjusted_market = buy_and_hold_gain * avg_exposure
    outperformance = simulation.total_return_pct - exposure_adjusted_market
    passive_portion_opportunity = buy_and_hold_gain * (1 - avg_exposure)
    hybrid_return = simulation.total_return_pct + passive_portion_opportunity
    active_alpha = hybrid_return - buy_and_hold_gain
    result = WalkForwardResult(
        config=config,
        simulation=simulation,
        evaluation_start_date=start_date,
        evaluation_end_date=end_date,
        lookback_days=lookback_days,
        step_days=step_days,
        buy_and_hold_gain=buy_and_hold_gain,
        exposure_adjusted_market=exposure_adjusted_market,
        outperformance=outperformance,
        hybrid_return=hybrid_return,
        active_alpha=active_alpha,
        performance_timings=None,
        instruments_used=list(prices_by_instrument.keys()),
    )
    out_path = config_result_path(root, config_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    return out_path


def run_outputs_task(
    result_path: Path,
    output_dir: Path,
    data_root: Path = None,
) -> Path:
    """
    Outputs task: read WalkForwardResult from disk, write trades.csv, indicators.csv,
    results.csv and all per-config charts to output_dir.

    If data_root is provided, loads price data from data_root/data/ for chart generation.
    Returns output_dir.
    """
    from .contract import (
        TRADES_CSV,
        INDICATORS_CSV,
        RESULTS_CSV,
        CHART_ALPHA_OVER_TIME,
        CHART_VALUE_GAIN_PER_INSTRUMENT,
        CHART_SCATTER_PNL_DURATION,
        CHART_SCATTER_CONFIDENCE_RISK,
        CHART_GAIN_PER_INSTRUMENT,
        CHART_TRADES_PER_INSTRUMENT,
        CHART_INDICATOR_BEST_WORST,
        CHART_PERFORMANCE_TIMINGS,
        CHART_COMPARISON,
        trades_csv_path,
        indicators_csv_path,
        results_csv_path,
        chart_path,
        timestamped_filename,
    )
    from ..grid_test.reporter_base import ComparisonReporter
    from ..grid_test.reporter_utils import trades_to_dataframe

    import gc
    
    result_path = Path(result_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "rb") as f:
        result = pickle.load(f)
    
    # Generate timestamped filenames
    trades_filename = timestamped_filename(TRADES_CSV)
    results_filename = timestamped_filename(RESULTS_CSV)
    indicators_filename = timestamped_filename(INDICATORS_CSV)
    
    reporter = ComparisonReporter(output_dir=str(output_dir))
    reporter.save_trades_csv(result, filename=trades_filename)
    reporter.save_results_csv([result], filename=results_filename)
    path = output_dir / indicators_filename
    if hasattr(result, "indicators_log") and result.indicators_log:
        import csv
        fieldnames = list(result.indicators_log[0].keys()) + ["quality_factor"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in result.indicators_log:
                out = dict(row)
                out["quality_factor"] = ""
                writer.writerow(out)
    else:
        path.touch()
    price_series = None
    price_data_by_instrument = {}
    if data_root is not None:
        data_root = Path(data_root)
        for inst in (result.instruments_used or [result.config.instruments[0] if result.config.instruments else None]):
            if not inst:
                continue
            dp = instrument_data_path(data_root, inst)
            if not dp.exists():
                continue
            df = pd.read_parquet(dp)
            series = df[df.columns[0]] if len(df.columns) > 0 else df.iloc[:, 0]
            series.index = pd.to_datetime(series.index)
            series = series[
                (series.index >= result.evaluation_start_date)
                & (series.index <= result.evaluation_end_date)
            ]
            price_data_by_instrument[inst] = series
        if price_data_by_instrument:
            first_inst = next(iter(price_data_by_instrument))
            price_series = price_data_by_instrument[first_inst]
    
    # Generate all charts (this is the real work of the outputs task)
    if price_series is not None:
        reporter.generate_alpha_over_time(
            result,
            price_series,
            price_data_by_instrument=price_data_by_instrument or None,
            filename=timestamped_filename(CHART_ALPHA_OVER_TIME),
        )
        reporter.generate_value_gain_and_benchmarks(
            result,
            price_data=price_series,
            benchmark_series=None,
            price_data_by_instrument=price_data_by_instrument or None,
            filename=timestamped_filename(CHART_VALUE_GAIN_PER_INSTRUMENT),
        )
        reporter.generate_pnl_vs_duration_scatter(result, filename=timestamped_filename(CHART_SCATTER_PNL_DURATION))
        reporter.generate_confidence_risk_vs_pnl_scatter(result, filename=timestamped_filename(CHART_SCATTER_CONFIDENCE_RISK))
        reporter.generate_gain_per_instrument(
            result,
            filename=timestamped_filename(CHART_GAIN_PER_INSTRUMENT),
            price_data_by_instrument=price_data_by_instrument or None,
        )
        reporter.generate_trades_per_instrument(result, filename=timestamped_filename(CHART_TRADES_PER_INSTRUMENT))
        reporter.generate_indicator_best_worst_overview(result, filename=timestamped_filename(CHART_INDICATOR_BEST_WORST))
    if getattr(result, "performance_timings", None):
        reporter.generate_performance_timings_chart(result, filename=timestamped_filename(CHART_PERFORMANCE_TIMINGS))
    reporter.generate_comparison_chart([result], filename=timestamped_filename(CHART_COMPARISON))
    
    # Generate DAG visualization AFTER other charts (excluded from timing)
    # This allows the DAG to show the actual outputs task timing
    if data_root is not None:
        graph_path = data_root / "task_graph.pkl"
        if graph_path.exists():
            try:
                from .dag_visualizer import visualize_task_dag
                with open(graph_path, "rb") as f:
                    task_graph = pickle.load(f)
                
                dag_chart_path = output_dir / timestamped_filename("task_dag.png")
                visualize_task_dag(
                    task_graph,
                    dag_chart_path,
                    title=f"Task Execution DAG: {result.config.name}",
                )
            except Exception:
                pass  # Don't fail if DAG visualization fails
    
    # Clean up memory after chart generation
    del reporter
    del result
    if price_data_by_instrument:
        price_data_by_instrument.clear()
    gc.collect()
    
    return output_dir


def run_grid_report_task(
    result_paths: List[Path],
    summary_dir: Path,
) -> Path:
    """
    GridReport task: read each result from disk, write grid-level charts and analysis.

    result_paths: paths to result.pkl (WalkForwardResult) for each config.
    Writes to summary_dir: comparison chart, dimension charts, equity curve, performance by
    instrument, results CSV, parameter sensitivity, analysis report; runs analyze_results_dir.
    Returns summary_dir.
    """
    from ..grid_test.reporter_base import ComparisonReporter
    from ..grid_test.analysis import analyze_results_dir

    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    result_paths = [Path(p) for p in result_paths]
    results = []
    for p in result_paths:
        if not p.exists():
            continue
        with open(p, "rb") as f:
            results.append(pickle.load(f))
    if not results:
        return summary_dir
    from .contract import RESULTS_CSV
    reporter = ComparisonReporter(output_dir=str(summary_dir))
    reporter.generate_comparison_chart(results, filename="comparison.png")
    reporter.generate_dimension_charts(results, filename_prefix="grid_dimension")
    reporter.generate_multi_strategy_equity_curve(results, filename_prefix="equity_curve_vs_2pa")
    reporter.generate_performance_by_instrument(results, filename_prefix="performance_by_instrument")
    reporter.save_results_csv(results, filename=RESULTS_CSV)
    reporter.save_parameter_sensitivity_csv(results, filename_prefix="parameter_sensitivity")
    reporter.generate_analysis_report(results, filename="analysis_report.md")
    analyze_results_dir(summary_dir, output_dir=summary_dir, verbose=False)
    return summary_dir
