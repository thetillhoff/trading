"""
Walk-forward evaluation engine for backtesting trading strategies.

Simulates day-by-day evaluation where signals are generated using only
historical data available up to each evaluation point, then evaluated
against actual future outcomes.
"""
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import csv
import time
from pathlib import Path

from ..signals.config import StrategyConfig, SignalConfig
from ..signals.detector import SignalDetector
from ..signals.target_calculator import TargetCalculator
from .portfolio import PortfolioSimulator, SimulationResult
from .walk_forward_types import (
    TradeOutcome,
    TradeEvaluation,
    EvaluationSummary,
    WalkForwardResult,
)
from ..shared.types import SignalType, TradingSignal
from ..shared.defaults import (
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    EMA_SHORT_PERIOD, EMA_LONG_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ELLIOTT_INVERTED_MIN_CONFIDENCE, ELLIOTT_INVERTED_MIN_WAVE_SIZE,
)
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# Aliases for backward compatibility
Signal = TradingSignal
ConfigurableSignalDetector = SignalDetector

# Extra calendar days when requesting load_start so the first trading day in data
# still gives at least lookback_days of history (avoids "only N days available" when
# eval_start - lookback_days falls on weekend/holiday)
LOOKBACK_CALENDAR_BUFFER_DAYS = 14


def _signal_config_from_strategy(config: StrategyConfig) -> SignalConfig:
    """Build SignalConfig from StrategyConfig. Single place for detector config (extensibility)."""
    return SignalConfig(
        use_elliott_wave=getattr(config, 'use_elliott_wave', True),
        min_confidence=config.min_confidence,
        min_wave_size=config.min_wave_size,
        use_elliott_wave_inverted=getattr(config, 'use_elliott_wave_inverted', False),
        use_elliott_wave_inverted_exit=getattr(config, 'use_elliott_wave_inverted_exit', False),
        min_confidence_inverted=getattr(config, 'min_confidence_inverted', ELLIOTT_INVERTED_MIN_CONFIDENCE),
        min_wave_size_inverted=getattr(config, 'min_wave_size_inverted', ELLIOTT_INVERTED_MIN_WAVE_SIZE),
        use_rsi=getattr(config, 'use_rsi', False),
        use_ema=getattr(config, 'use_ema', False),
        use_macd=getattr(config, 'use_macd', False),
        rsi_period=getattr(config, 'rsi_period', RSI_PERIOD),
        rsi_oversold=getattr(config, 'rsi_oversold', RSI_OVERSOLD),
        rsi_overbought=getattr(config, 'rsi_overbought', RSI_OVERBOUGHT),
        ema_short_period=getattr(config, 'ema_short_period', EMA_SHORT_PERIOD),
        ema_long_period=getattr(config, 'ema_long_period', EMA_LONG_PERIOD),
        macd_fast=getattr(config, 'macd_fast', MACD_FAST),
        macd_slow=getattr(config, 'macd_slow', MACD_SLOW),
        macd_signal=getattr(config, 'macd_signal', MACD_SIGNAL),
        signal_types=getattr(config, 'signal_types', 'all'),
        min_confirmations=getattr(config, 'min_confirmations', None),
        min_certainty=getattr(config, 'min_certainty', None),
        use_trend_filter=getattr(config, 'use_trend_filter', False),
        indicator_weights=getattr(config, 'indicator_weights', None),
        use_regime_detection=getattr(config, 'use_regime_detection', False),
        invert_signals_in_bull=getattr(config, 'invert_signals_in_bull', True),
        adx_threshold=getattr(config, 'adx_threshold', 30.0),
        regime_mode=getattr(config, 'regime_mode', 'adx_ma'),
        regime_vol_window=getattr(config, 'regime_vol_window', 20),
        regime_vol_threshold=getattr(config, 'regime_vol_threshold', 0.015),
        regime_slope_window=getattr(config, 'regime_slope_window', 5),
        regime_slope_threshold=getattr(config, 'regime_slope_threshold', 0.0005),
        use_volatility_filter=getattr(config, 'use_volatility_filter', False),
        volatility_max=getattr(config, 'volatility_max', 0.02),
        use_multi_timeframe=getattr(config, 'use_multi_timeframe', False),
        multi_timeframe_weekly_ema_period=getattr(config, 'multi_timeframe_weekly_ema_period', 8),
        use_multi_timeframe_filter=getattr(config, 'use_multi_timeframe_filter', True),
    )


def _portfolio_simulator_from_config(config: StrategyConfig) -> PortfolioSimulator:
    """Build PortfolioSimulator from StrategyConfig. Single place for sim config (extensibility)."""
    return PortfolioSimulator(
        initial_capital=getattr(config, 'initial_capital', 10000.0),
        position_size_pct=getattr(config, 'position_size_pct', 0.2),
        max_positions=getattr(config, 'max_positions', None),
        max_positions_per_instrument=getattr(config, 'max_positions_per_instrument', None),
        max_days=config.max_days,
        use_confidence_sizing=getattr(config, 'use_confidence_sizing', False),
        use_confirmation_modulation=getattr(config, 'use_confirmation_modulation', False),
        confirmation_size_factors=getattr(config, 'confirmation_size_factors', None),
        use_volatility_sizing=getattr(config, 'use_volatility_sizing', False),
        volatility_threshold=getattr(config, 'volatility_threshold', 0.03),
        volatility_size_reduction=getattr(config, 'volatility_size_reduction', 0.5),
        use_flexible_sizing=getattr(config, 'use_flexible_sizing', False),
        flexible_sizing_method=getattr(config, 'flexible_sizing_method', 'confidence'),
        flexible_sizing_target_rr=getattr(config, 'flexible_sizing_target_rr', 2.5),
        trade_fee_pct=getattr(config, 'trade_fee_pct', None),
        trade_fee_absolute=getattr(config, 'trade_fee_absolute', None),
        trade_fee_min=getattr(config, 'trade_fee_min', None),
        trade_fee_max=getattr(config, 'trade_fee_max', None),
        min_position_size=getattr(config, 'min_position_size', None),
    )


def _signals_for_config_instrument_worker(
    args: tuple,
) -> tuple[str, str, List[TradingSignal], Dict[str, float]]:
    """
    Worker function for parallel signal generation (one config×instrument).
    
    Generates all signals for a single (config, instrument) over the full
    walk-forward (looping over eval_dates). No portfolio sim; just returns signals
    and per-phase timings for the performance chart.
    
    This is a module-level function so it can be pickled for ProcessPoolExecutor.
    
    Args:
        args: Tuple of (config, instrument, eval_dates, load_start, end_date, 
                       lookback_days, step_days, min_history_days, column)
    
    Returns:
        Tuple of (config_name, instrument, signals_list, timings_dict)
    """
    (
        config,
        instrument,
        eval_dates,
        load_start,
        end_date,
        lookback_days,
        step_days,
        min_history_days,
        column,
    ) = args
    
    # Load data for this instrument
    from ..data.loader import DataLoader
    
    load_start_str = load_start.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    data_inst = DataLoader.from_instrument(
        instrument,
        start_date=load_start_str,
        end_date=end_date_str,
        column=column
    )
    
    if data_inst is None or len(data_inst) == 0:
        return (config.name, instrument, [], {})
    
    # Build signal detector and target calculator from config
    signal_detector = SignalDetector(_signal_config_from_strategy(config))
    target_calculator = TargetCalculator(
        risk_reward_ratio=config.risk_reward,
        use_atr_stops=getattr(config, 'use_atr_stops', False),
        atr_stop_multiplier=getattr(config, 'atr_stop_multiplier', 2.0),
        atr_period=getattr(config, 'atr_period', 14)
    )
    
    all_signals = []
    worker_timings: Dict[str, float] = {}
    min_history = 30  # minimum bars for pattern detection
    
    # Loop over eval_dates and generate signals at each date
    for eval_date in eval_dates:
        lookback_start = eval_date - timedelta(days=lookback_days)
        historical_data = data_inst[
            (data_inst.index >= lookback_start) & (data_inst.index <= eval_date)
        ]
        
        if len(historical_data) < min_history:
            continue
        
        # Detect signals (collect breakdown into worker_timings for performance chart)
        signals, _indicator_df, all_waves = signal_detector.detect_signals_with_indicators(
            historical_data, timings=worker_timings
        )
        
        if not signals:
            continue
        
        # Calculate targets (time for performance chart)
        t0_targets = time.perf_counter()
        signals_with_targets = target_calculator.calculate_targets(
            signals,
            historical_data,
            all_waves=all_waves if getattr(config, "use_wave_relationship_targets", True) else None,
        )
        worker_timings["signal_detection_target_calculation"] = (
            worker_timings.get("signal_detection_target_calculation", 0.0)
            + (time.perf_counter() - t0_targets)
        )
        
        # Tag with instrument and collect
        for s in signals_with_targets:
            s.instrument = instrument
        
        all_signals.extend(signals_with_targets)
    
    return (config.name, instrument, all_signals, worker_timings)


def _safe_instrument_filename(name: str) -> str:
    """Safe filesystem name for instrument (e.g. BRK-B, AAPL)."""
    import re
    return re.sub(r"[^\w\-.]", "_", str(name))


def _signals_for_config_instrument_worker_disk(
    job_index: int,
    temp_dir: str,
) -> tuple[str, str, List[TradingSignal], Dict[str, float]]:
    """
    Worker for grid-search: load config and data from temp dir, run signal generation.
    Keeps memory low by not passing full config/data through the process pool.
    """
    import pickle
    import pandas as pd

    temp_path = Path(temp_dir)
    with open(temp_path / "jobs.pkl", "rb") as f:
        jobs = pickle.load(f)
    with open(temp_path / "grid_params.pkl", "rb") as f:
        grid_params = pickle.load(f)
    eval_dates, load_start, end_date, lookback_days, step_days, min_history_days = grid_params

    config_idx, instrument, column = jobs[job_index]
    with open(temp_path / "configs" / f"{config_idx}.pkl", "rb") as f:
        config = pickle.load(f)

    data_path = temp_path / "data" / f"{_safe_instrument_filename(instrument)}.parquet"
    if not data_path.exists():
        from ..data.loader import DataLoader
        load_start_str = load_start.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        data_inst = DataLoader.from_instrument(
            instrument, start_date=load_start_str, end_date=end_date_str, column=column
        )
    else:
        df = pd.read_parquet(data_path)
        data_inst = df[df.columns[0]] if len(df.columns) == 1 else df.iloc[:, 0]
        if hasattr(data_inst.index, 'tz_localize'):
            pass
        data_inst.index = pd.to_datetime(data_inst.index)

    if data_inst is None or len(data_inst) == 0:
        return (config.name, instrument, [], {})

    signal_detector = SignalDetector(_signal_config_from_strategy(config))
    target_calculator = TargetCalculator(
        risk_reward_ratio=config.risk_reward,
        use_atr_stops=getattr(config, 'use_atr_stops', False),
        atr_stop_multiplier=getattr(config, 'atr_stop_multiplier', 2.0),
        atr_period=getattr(config, 'atr_period', 14)
    )
    all_signals = []
    worker_timings: Dict[str, float] = {}
    min_history = 30

    for eval_date in eval_dates:
        lookback_start = eval_date - timedelta(days=lookback_days)
        historical_data = data_inst[
            (data_inst.index >= lookback_start) & (data_inst.index <= eval_date)
        ]
        if len(historical_data) < min_history:
            continue
        signals, _indicator_df, all_waves = signal_detector.detect_signals_with_indicators(
            historical_data, timings=worker_timings
        )
        if not signals:
            continue
        t0_targets = time.perf_counter()
        signals_with_targets = target_calculator.calculate_targets(
            signals,
            historical_data,
            all_waves=all_waves if getattr(config, "use_wave_relationship_targets", True) else None,
        )
        worker_timings["signal_detection_target_calculation"] = (
            worker_timings.get("signal_detection_target_calculation", 0.0)
            + (time.perf_counter() - t0_targets)
        )
        for s in signals_with_targets:
            s.instrument = instrument
        all_signals.extend(signals_with_targets)

    return (config.name, instrument, all_signals, worker_timings)


class WalkForwardEvaluator:
    """
    Walk-forward backtesting evaluator.
    
    Simulates real trading by:
    1. For each evaluation point (e.g., each day/week):
       - Only use historical data up to that point
       - Generate signals based on that historical data
       - Evaluate those signals against actual future outcomes
    2. Aggregate results across all evaluation points
    """
    
    def __init__(
        self,
        lookback_days: int = 365,
        step_days: int = 30,
        min_history_days: int = 100,
    ):
        """
        Initialize the walk-forward evaluator.
        
        Args:
            lookback_days: Number of days of history to use for signal generation
            step_days: Number of days between evaluation points
            min_history_days: Minimum days of history required before starting evaluation
        """
        self.lookback_days = lookback_days
        self.step_days = step_days
        self.min_history_days = min_history_days

        # For CSV logging
        self.indicators_log = []
        self.indicators_csv_path = None

    def _log_indicators_at_date(self, eval_date: pd.Timestamp, data: pd.Series,
                               indicator_df: pd.DataFrame, config_name: str):
        """
        Log all indicator values at a specific evaluation date.
        """
        if indicator_df is None or len(indicator_df) == 0:
            return

        # Get the most recent indicator values (last row)
        latest_indicators = indicator_df.iloc[-1]

        # Extract individual indicator values
        indicators_row = {
            'date': eval_date.strftime('%Y-%m-%d'),
            'config': config_name,
            'price': data.iloc[-1] if len(data) > 0 else None,
            'rsi_value': latest_indicators.get('rsi', None),
            'rsi_signal': 'BUY' if latest_indicators.get('rsi', 50) < getattr(self, '_rsi_oversold', 25) else 'SELL' if latest_indicators.get('rsi', 50) > getattr(self, '_rsi_overbought', 75) else 'NEUTRAL',
            'ema_short': latest_indicators.get('ema_short', None),
            'ema_long': latest_indicators.get('ema_long', None),
            'ema_signal': 'BUY' if latest_indicators.get('ema_short', 0) > latest_indicators.get('ema_long', 0) else 'SELL',
            'macd_value': latest_indicators.get('macd', None),
            'macd_signal_line': latest_indicators.get('macd_signal', None),
            'macd_histogram': latest_indicators.get('macd_histogram', None),
            'macd_signal': 'BUY' if latest_indicators.get('macd', 0) > latest_indicators.get('macd_signal', 0) else 'SELL',
        }

        self.indicators_log.append(indicators_row)

    def save_indicators_csv(
        self,
        output_dir: Path,
        filename_prefix: str = "indicators",
        config: Optional[StrategyConfig] = None,
        result: Optional["WalkForwardResult"] = None,
        filename: Optional[str] = None,
    ):
        """
        Save the indicators log to CSV with optional configuration metadata.

        When result is provided, adds a quality_factor column per date from positions
        opened that day (first trade's quality_factor for that date).

        Args:
            output_dir: Directory to save the CSV file
            filename_prefix: Prefix for the filename (used when filename is None)
            config: Optional strategy configuration to include as metadata
            result: Optional walk-forward result to add quality_factor from positions
            filename: Optional canonical filename (e.g. indicators.csv). If set, used as-is.
        """
        if not self.indicators_log:
            return None

        date_to_quality: Dict[str, float] = {}
        if result and getattr(result, "simulation", None) and result.simulation.positions:
            for pos in result.simulation.positions:
                if pos.entry_timestamp is not None:
                    date_str = pos.entry_timestamp.strftime("%Y-%m-%d")
                    qf = getattr(pos, "quality_factor", None)
                    if qf is not None and date_str not in date_to_quality:
                        date_to_quality[date_str] = qf

        if filename is not None:
            filepath = output_dir / filename
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = output_dir / f"{filename_prefix}_{timestamp}.csv"

        fieldnames = list(self.indicators_log[0].keys()) + ["quality_factor"]

        with open(filepath, 'w', newline='') as csvfile:
            # Write configuration metadata if provided
            if config:
                from dataclasses import asdict
                csvfile.write("# Strategy Configuration\n")
                csvfile.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                csvfile.write("#\n")
                config_dict = asdict(config)
                for key, value in sorted(config_dict.items()):
                    csvfile.write(f"# {key}: {value}\n")
                csvfile.write("#\n")
            
            # Write CSV data
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self.indicators_log:
                out = dict(row)
                out["quality_factor"] = date_to_quality.get(row.get("date", ""), "")
                writer.writerow(out)

        return filepath

    def evaluate(
        self,
        data: pd.Series,
        config: StrategyConfig,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        verbose: bool = False,
    ) -> WalkForwardResult:
        """
        Run walk-forward evaluation on the data.
        
        Args:
            data: Time series data with datetime index
            config: Strategy configuration to evaluate
            start_date: Start date for evaluation (default: min_history_days after data start)
            end_date: End date for evaluation (default: data end)
            verbose: Print progress information
            
        Returns:
            WalkForwardResult with evaluation results
        """
        # Use config values for step_days and lookback_days if available
        step_days = getattr(config, 'step_days', self.step_days)
        lookback_days = getattr(config, 'lookback_days', self.lookback_days)
        
        # Determine evaluation period
        data_start = data.index.min()
        data_end = data.index.max()
        
        if start_date is None:
            start_date = data_start + timedelta(days=self.min_history_days)
        if end_date is None:
            end_date = data_end
        
        # Ensure we have enough history
        if start_date < data_start + timedelta(days=self.min_history_days):
            start_date = data_start + timedelta(days=self.min_history_days)
        
        self.indicators_log = []
        if verbose:
            print(f"Walk-forward evaluation: {start_date.date()} to {end_date.date()}")
            print(f"  Lookback: {lookback_days} days, Step: {step_days} days")
            print(f"  Config: {config.name}")
        
        # Generate evaluation points
        eval_dates = self._generate_eval_dates(data, start_date, end_date, step_days)
        
        if verbose:
            print(f"  Evaluation points: {len(eval_dates)}")
        
        # Collect all signals and evaluations
        all_signals = []
        all_evaluations = []
        signals_by_date = {}
        seen_signal_keys = set()  # Track unique signals to avoid duplicates
        
        target_calculator = TargetCalculator(
            risk_reward_ratio=config.risk_reward,
            use_atr_stops=getattr(config, 'use_atr_stops', False),
            atr_stop_multiplier=getattr(config, 'atr_stop_multiplier', 2.0),
            atr_period=getattr(config, 'atr_period', 14)
        )
        signal_detector = SignalDetector(_signal_config_from_strategy(config))
        
        total_eval_dates = len(eval_dates)
        progress_interval = max(1, total_eval_dates // 10)  # Show ~10 progress updates
        start_time = time.time()
        performance_timings: Dict[str, float] = {}
        
        for i, eval_date in enumerate(eval_dates):
            # Get historical data up to eval_date (for signal generation)
            lookback_start = eval_date - timedelta(days=lookback_days)
            historical_data = data[(data.index >= lookback_start) & (data.index <= eval_date)]
            
            if len(historical_data) < 30:  # Need minimum data for pattern detection
                continue
            
            # Detect signals using only historical data
            # The configurable detector handles signal type filtering internally
            signals, indicator_df, all_waves = signal_detector.detect_signals_with_indicators(
                historical_data, timings=performance_timings
            )

            # Log indicators at this evaluation date
            self._log_indicators_at_date(eval_date, historical_data, indicator_df, config.name)

            # Calculate targets for all signals at once (optionally pass waves for wave-specific targets)
            if signals:
                t0_targets = time.perf_counter()
                signals_with_targets = target_calculator.calculate_targets(
                    signals,
                    historical_data,
                    all_waves=all_waves if getattr(config, "use_wave_relationship_targets", True) else None,
                )
                performance_timings["signal_detection_target_calculation"] = (
                    performance_timings.get("signal_detection_target_calculation", 0.0)
                    + (time.perf_counter() - t0_targets)
                )
                
                # Add unique signals to our collection
                # Determine instrument identifier (use first instrument from config for single-instrument)
                instrument_id = config.instruments[0] if config.instruments else None
                
                for signal in signals_with_targets:
                    # Set instrument identifier on signal
                    signal.instrument = instrument_id
                    
                    # Create a unique key for each signal to avoid duplicates
                    signal_key = (signal.timestamp, signal.signal_type, signal.price)
                    if signal_key in seen_signal_keys:
                        continue
                    seen_signal_keys.add(signal_key)
                    
                    all_signals.append(signal)
                    
                    # Track signals by evaluation date
                    date_key = eval_date.strftime('%Y-%m-%d')
                    if date_key not in signals_by_date:
                        signals_by_date[date_key] = []
                    signals_by_date[date_key].append(signal)
            
            # Show progress
            if (i + 1) % progress_interval == 0 or i == total_eval_dates - 1:
                pct = ((i + 1) / total_eval_dates) * 100
                elapsed = time.time() - start_time
                print(f"  Progress: {i + 1}/{total_eval_dates} ({pct:.0f}%) - {len(all_signals)} signals - {elapsed:.1f}s", end='\r', flush=True)
        
        elapsed_total = time.time() - start_time
        performance_timings["signal_detection"] = elapsed_total
        print(f"  Signal detection completed in {elapsed_total:.1f}s" + " " * 30, flush=True)  # Padding to clear previous line
        print(f"  Total unique signals detected: {len(all_signals)}")
        
        print("  Simulating portfolio...", flush=True)
        portfolio_sim = _portfolio_simulator_from_config(config)

        # Simulate the strategy
        eval_data = data[(data.index >= start_date) & (data.index <= end_date)]
        sim_start_time = time.time()
        simulation = portfolio_sim.simulate_strategy(
            eval_data,
            all_signals,
            start_date=start_date,
            end_date=end_date,
        )
        sim_elapsed = time.time() - sim_start_time
        performance_timings["portfolio_simulation"] = sim_elapsed
        print(f"  Portfolio simulation completed in {sim_elapsed:.1f}s")
        
        # Calculate buy-and-hold for comparison
        bh_simulation = portfolio_sim.simulate_buy_and_hold(
            eval_data,
            start_date=start_date,
            end_date=end_date,
        )
        buy_and_hold_gain = bh_simulation.total_return_pct
        
        # Calculate exposure-adjusted comparison
        # If strategy was only 30% invested on average, compare to 30% of market return
        avg_exposure = simulation.avg_exposure_pct / 100.0  # Convert to decimal
        exposure_adjusted_market = buy_and_hold_gain * avg_exposure
        
        # Outperformance vs exposure-adjusted market (fair comparison)
        outperformance = simulation.total_return_pct - exposure_adjusted_market
        
        # Hybrid strategy: what if uninvested cash earned market returns instead of 0%?
        # The simulation.total_return_pct is the return with cash earning 0%.
        # In hybrid model, that idle cash (1 - avg_exposure) should earn market returns.
        # So we ADD the market return on the uninvested portion.
        passive_portion_opportunity = buy_and_hold_gain * (1 - avg_exposure)
        hybrid_return = simulation.total_return_pct + passive_portion_opportunity
        
        # Alpha: did hybrid beat pure buy-and-hold?
        # Positive alpha = active trading added value
        # Negative alpha = would have been better to just buy-and-hold everything
        active_alpha = hybrid_return - buy_and_hold_gain
        
        if verbose:
            print(f"\n  Results:")
            print(f"    Trades: {simulation.total_trades}")
            print(f"    Win Rate: {simulation.win_rate:.1f}%")
            print(f"    Avg Exposure: {simulation.avg_exposure_pct:.1f}%")
            print(f"    Total Return: {simulation.total_return_pct:.2f}%")
            print(f"    Market (100% invested): {buy_and_hold_gain:.2f}%")
            print(f"    Hybrid Return: {hybrid_return:.2f}%")
            print(f"    Alpha vs Buy-and-Hold: {active_alpha:+.2f}%")
        
        return WalkForwardResult(
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
            performance_timings=performance_timings or None,
            instruments_used=[config.instruments[0]] if config.instruments else None,
        )
    
    def evaluate_multi_instrument(
        self,
        config: StrategyConfig,
        verbose: bool = True,
        max_workers: Optional[int] = None,
    ) -> WalkForwardResult:
        """
        Evaluate strategy across multiple instruments with unified results.

        For single instrument: delegates to standard evaluate()
        For multiple instruments: combines signals from all instruments.
        When multiple instruments and max_workers > 1, runs per-instrument
        signal detection in parallel at each eval_date.

        Args:
            config: Strategy configuration (must include instruments, start_date, end_date)
            verbose: Print progress information
            max_workers: Parallel workers for instrument loop (default: min(K, cpu_count())); 1 = sequential.

        Returns:
            WalkForwardResult with combined trades from all instruments
        """
        from ..data.loader import DataLoader
        
        self.indicators_log = []
        if not config.instruments:
            raise ValueError("Config must specify at least one instrument")
        
        # Single instrument: use standard evaluation
        if len(config.instruments) == 1:
            from ..data.preparation import prepare_and_validate, DataPreparationError
            
            instrument = config.instruments[0]
            lookback_days = getattr(config, 'lookback_days', self.lookback_days)
            step_days = getattr(config, 'step_days', self.step_days)
            performance_timings: Dict[str, float] = {}
            
            if verbose:
                print(f"Evaluating on 1 instrument: {instrument}")
                print(f"Date range: {config.start_date or 'earliest'} to {config.end_date or 'latest'}")
            
            # Data prep (validate + eval_dates) for consistent phase timings with multi-instrument
            t0_prep = time.perf_counter()
            if config.start_date and config.end_date:
                try:
                    prepare_and_validate(
                        instruments=[instrument],
                        start_date=config.start_date,
                        end_date=config.end_date,
                        lookback_days=lookback_days,
                        step_days=step_days,
                        min_history_days=self.min_history_days,
                        column=config.column,
                    )
                except DataPreparationError:
                    pass  # continue; evaluate() will use its own eval_dates
            performance_timings["data_prep"] = time.perf_counter() - t0_prep
            
            load_start = config.start_date
            if config.start_date:
                eval_start_ts = pd.Timestamp(config.start_date)
                load_start = (eval_start_ts - timedelta(days=lookback_days + LOOKBACK_CALENDAR_BUFFER_DAYS)).strftime('%Y-%m-%d')
            
            t0_load = time.perf_counter()
            data = DataLoader.from_instrument(
                instrument,
                start_date=load_start,
                end_date=config.end_date,
                column=config.column
            )
            data_load_elapsed = time.perf_counter() - t0_load
            
            if config.start_date and data is not None and len(data) > 0:
                data_start = data.index.min()
                eval_start = pd.Timestamp(config.start_date)
                requested_start = eval_start - timedelta(days=lookback_days)
                if data_start > requested_start:
                    actual_days = max(0, (eval_start - data_start).days)
                    print(f"  Requested {lookback_days} days of history before evaluation start but only {actual_days} days available; early evaluation dates may have reduced lookback.")
            
            start_date = pd.Timestamp(config.start_date) if config.start_date else None
            end_date = pd.Timestamp(config.end_date) if config.end_date else None
            
            result = self.evaluate(
                data,
                config,
                start_date=start_date,
                end_date=end_date,
                verbose=verbose
            )
            if result.performance_timings is not None:
                result.performance_timings["data_prep"] = performance_timings["data_prep"]
                result.performance_timings["data_load"] = data_load_elapsed
            return result
        
        # Multiple instruments: use small-jobs parallelism
        # 1. Data prep (validate + get eval_dates)
        # 2. Dispatch N jobs (one per instrument) to worker pool
        # 3. Merge signals
        # 4. One portfolio sim
        
        from ..data.preparation import prepare_and_validate, DataPreparationError
        
        step_days = getattr(config, 'step_days', self.step_days)
        lookback_days = getattr(config, 'lookback_days', self.lookback_days)
        performance_timings: Dict[str, float] = {}
        
        if verbose:
            print(f"Evaluating on {len(config.instruments)} instruments: {config.instruments}")
            print(f"  Date range: {config.start_date} to {config.end_date}")
            print(f"  Lookback: {lookback_days} days, Step: {step_days} days")
        
        # Step 1: Data preparation (fail-fast validation + eval_dates)
        if verbose:
            print(f"  Validating data availability...", flush=True)
        
        t0_prep = time.perf_counter()
        try:
            prep_result = prepare_and_validate(
                instruments=config.instruments,
                start_date=config.start_date,
                end_date=config.end_date,
                lookback_days=lookback_days,
                step_days=step_days,
                min_history_days=self.min_history_days,
                column=config.column,
            )
        except DataPreparationError as e:
            raise ValueError(f"Data preparation failed: {e}")
        
        performance_timings["data_prep"] = time.perf_counter() - t0_prep
        
        eval_dates = prep_result.eval_dates
        start_date = prep_result.start_date
        end_date = prep_result.end_date
        load_start = prep_result.load_start
        instruments_to_use = prep_result.instruments  # may be reduced when some have no data in range
        
        if verbose:
            print(f"  Data validated. Evaluation points: {len(eval_dates)}")
            if len(instruments_to_use) < len(config.instruments):
                print(f"  Using {len(instruments_to_use)} instruments ({len(config.instruments) - len(instruments_to_use)} skipped)")
            print(f"  Effective eval range: {start_date.date()} to {end_date.date()}")
        
        # Step 2: Dispatch N signal generation jobs (one per instrument)
        # Default worker count: min(N, cpu_count)
        if max_workers is None:
            max_workers = max(1, min(len(instruments_to_use), os.cpu_count() or 1))
        
        if verbose:
            mode = "in-process" if max_workers == 1 else f"{max_workers} parallel workers"
            print(f"  Generating signals: {mode}...", flush=True)
        
        t0_signals = time.time()
        all_signals = []
        seen_signal_keys = set()
        
        # For max_workers=1, run synchronously (in-process) to support mocking in tests
        if max_workers == 1:
            for inst in instruments_to_use:
                try:
                    args = (
                        config,
                        inst,
                        eval_dates,
                        load_start,
                        end_date,
                        lookback_days,
                        step_days,
                        self.min_history_days,
                        config.column,
                    )
                    config_name, instrument, signals, worker_timings = _signals_for_config_instrument_worker(args)
                    for k, v in worker_timings.items():
                        performance_timings[k] = performance_timings.get(k, 0.0) + v
                    
                    # Dedupe and merge
                    for signal in signals:
                        signal_key = (signal.timestamp, signal.signal_type, signal.price, signal.instrument)
                        if signal_key in seen_signal_keys:
                            continue
                        seen_signal_keys.add(signal_key)
                        all_signals.append(signal)
                    
                    if verbose:
                        print(f"  [{inst}] {len(signals)} signals", flush=True)
                except Exception as e:
                    print(f"  Error generating signals for {inst}: {e}")
        else:
            # Multi-process execution for max_workers > 1
            # Set multiprocessing start method for Docker/macOS compatibility
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
            
            job_args = [
                (
                    config,
                    inst,
                    eval_dates,
                    load_start,
                    end_date,
                    lookback_days,
                    step_days,
                    self.min_history_days,
                    config.column,
                )
                for inst in instruments_to_use
            ]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_signals_for_config_instrument_worker, args): args[1]
                    for args in job_args
                }
                
                completed = 0
                for future in as_completed(futures):
                    inst = futures[future]
                    try:
                        config_name, instrument, signals, worker_timings = future.result()
                        completed += 1
                        for k, v in worker_timings.items():
                            performance_timings[k] = performance_timings.get(k, 0.0) + v
                        
                        # Dedupe and merge
                        for signal in signals:
                            signal_key = (signal.timestamp, signal.signal_type, signal.price, signal.instrument)
                            if signal_key in seen_signal_keys:
                                continue
                            seen_signal_keys.add(signal_key)
                            all_signals.append(signal)
                        
                        if verbose:
                            print(f"  [{completed}/{len(instruments_to_use)}] {instrument}: {len(signals)} signals", flush=True)
                    except Exception as e:
                        print(f"  Error generating signals for {inst}: {e}")
                        completed += 1
                    finally:
                        futures.pop(future, None)
        
        elapsed_signals = time.time() - t0_signals
        performance_timings["signal_detection"] = elapsed_signals
        
        all_signals = sorted(all_signals, key=lambda s: s.timestamp)
        
        if verbose:
            print(f"  Signal generation completed in {elapsed_signals:.1f}s")
            print(f"  Total unique signals: {len(all_signals)}")
        
        # Step 3: Load all data for portfolio sim
        if verbose:
            print(f"  Loading price data for portfolio simulation...", flush=True)
        
        t0_load = time.perf_counter()
        data_by_instrument: Dict[str, pd.Series] = {}
        for inst in instruments_to_use:
            data_by_instrument[inst] = DataLoader.from_instrument(
                inst,
                start_date=load_start.strftime('%Y-%m-%d'),
                end_date=config.end_date,
                column=config.column
            )
        performance_timings["data_load"] = time.perf_counter() - t0_load
        
        prices_by_instrument = {
            inst: data_by_instrument[inst][
                (data_by_instrument[inst].index >= start_date) & (data_by_instrument[inst].index <= end_date)
            ]
            for inst in instruments_to_use
        }
        
        # Step 4: Portfolio simulation
        if verbose:
            print(f"  Simulating portfolio...", flush=True)
        
        portfolio_sim = _portfolio_simulator_from_config(config)
        sim_start_time = time.time()
        simulation = portfolio_sim.simulate_strategy(
            prices_by_instrument,
            all_signals,
            start_date=start_date,
            end_date=end_date,
        )
        sim_elapsed = time.time() - sim_start_time
        performance_timings["portfolio_simulation"] = sim_elapsed
        
        if verbose:
            print(f"  Portfolio simulation completed in {sim_elapsed:.1f}s")
        
        # Buy-and-hold comparison (first instrument)
        data_first = data_by_instrument[instruments_to_use[0]]
        eval_data_first = data_first[(data_first.index >= start_date) & (data_first.index <= end_date)]
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
        
        if verbose:
            print(f"\n  Results:")
            print(f"    Trades: {simulation.total_trades}")
            print(f"    Win Rate: {simulation.win_rate:.1f}%")
            print(f"    Avg Exposure: {simulation.avg_exposure_pct:.1f}%")
            print(f"    Total Return: {simulation.total_return_pct:.2f}%")
            print(f"    Market (first instrument B&H): {buy_and_hold_gain:.2f}%")
            print(f"    Hybrid Return: {hybrid_return:.2f}%")
            print(f"    Alpha vs Buy-and-Hold: {active_alpha:+.2f}%")
        
        return WalkForwardResult(
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
            performance_timings=performance_timings or None,
            instruments_used=instruments_to_use,
        )
    
    def _generate_eval_dates(
        self,
        data: pd.Series,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        step_days: int,
    ) -> List[pd.Timestamp]:
        """Generate evaluation dates at step_days intervals."""
        eval_dates = []
        current_date = start_date
        
        while current_date <= end_date:
            # Find the nearest trading day
            nearest_idx = data.index.get_indexer([current_date], method='nearest')[0]
            if nearest_idx >= 0 and nearest_idx < len(data):
                eval_date = data.index[nearest_idx]
                if eval_date >= start_date and eval_date <= end_date:
                    if eval_date not in eval_dates:
                        eval_dates.append(eval_date)
            
            current_date += timedelta(days=step_days)
        
        return sorted(eval_dates)
    
    def _calculate_cumulative_gains(
        self,
        evaluations: List[TradeEvaluation],
    ) -> List[float]:
        """Calculate cumulative gains over time."""
        if not evaluations:
            return []
        
        # Sort by exit timestamp
        sorted_evals = sorted(
            [e for e in evaluations if e.exit_timestamp is not None],
            key=lambda e: e.exit_timestamp
        )
        
        cumulative = 0.0
        cumulative_gains = []
        
        for eval in sorted_evals:
            cumulative += eval.gain_percentage
            cumulative_gains.append(cumulative)
        
        return cumulative_gains
    
    def compare_configs(
        self,
        data: pd.Series,
        configs: List[StrategyConfig],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        verbose: bool = False,
    ) -> List[WalkForwardResult]:
        """
        Compare multiple configurations using walk-forward evaluation.
        
        Args:
            data: Time series data with datetime index
            configs: List of strategy configurations to compare
            start_date: Start date for evaluation
            end_date: End date for evaluation
            verbose: Print progress information
            
        Returns:
            List of WalkForwardResult for each configuration
        """
        results = []
        total = len(configs)
        
        for i, config in enumerate(configs, 1):
            # Always show progress for multiple configs
            if total > 1:
                print(f"\n[{i}/{total}] Evaluating: {config.name}", flush=True)
            elif verbose:
                print(f"\nEvaluating: {config.name}", flush=True)
            
            result = self.evaluate(
                data,
                config,
                start_date=start_date,
                end_date=end_date,
                verbose=verbose,
            )
            results.append(result)
            
            # Show quick summary for grid search
            if total > 5:
                s = result.summary
                print(f"         Trades: {s.total_trades}, Win rate: {s.win_rate:.1f}%, Total gain: {s.total_gain:.1f}%", flush=True)
        
        if total > 1:
            print(f"\nCompleted {total} configurations.")
        
        return results
    
    def compare_configs_parallel(
        self,
        data: pd.Series,
        configs: List[StrategyConfig],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        max_workers: Optional[int] = None,
    ) -> List[WalkForwardResult]:
        """
        Compare multiple configurations using parallel walk-forward evaluation.
        
        Uses multiple CPU cores to evaluate configurations simultaneously.
        
        Args:
            data: Time series data with datetime index
            configs: List of strategy configurations to compare
            start_date: Start date for evaluation
            end_date: End date for evaluation
            max_workers: Number of parallel workers (default: CPU count)
            
        Returns:
            List of WalkForwardResult for each configuration
        """
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() or 1)
        
        total = len(configs)
        print(f"\nRunning {total} configurations in parallel with {max_workers} workers...", flush=True)
        
        # Create a helper function that can be pickled for multiprocessing
        # We need to pass all the required data to each worker
        eval_args = [
            (data, config, start_date, end_date, self.lookback_days, self.step_days, self.min_history_days)
            for config in configs
        ]
        
        results = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_evaluate_config_worker, args): args[1].name
                for args in eval_args
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                config_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    s = result.summary
                    print(f"  [{completed}/{total}] {config_name}: "
                          f"Trades={s.total_trades}, Win={s.win_rate:.1f}%, Gain={s.total_gain:.1f}%", 
                          flush=True)
                except Exception as e:
                    print(f"  [{completed}/{total}] {config_name}: FAILED - {e}", flush=True)
                    completed += 1
                finally:
                    futures.pop(future, None)
        
        print(f"\nCompleted {len(results)}/{total} configurations.")
        return results


def _evaluate_config_worker(args) -> WalkForwardResult:
    """
    Worker function for parallel config evaluation.
    
    This is a module-level function so it can be pickled for multiprocessing.
    """
    data, config, start_date, end_date, lookback_days, step_days, min_history_days = args
    
    # Create a new evaluator instance for this worker
    evaluator = WalkForwardEvaluator(
        lookback_days=lookback_days,
        step_days=step_days,
        min_history_days=min_history_days,
    )
    
    # Run the evaluation (suppress verbose output in workers)
    return evaluator.evaluate(
        data,
        config,
        start_date=start_date,
        end_date=end_date,
        verbose=False,
    )
