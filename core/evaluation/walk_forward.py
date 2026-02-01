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


def _signals_for_instrument_at_date(
    eval_date: pd.Timestamp,
    inst: str,
    data_inst: pd.Series,
    config: StrategyConfig,
    lookback_days: int,
    min_history: int,
    signal_detector: SignalDetector,
    target_calculator: TargetCalculator,
) -> List[TradingSignal]:
    """
    Generate signals for one instrument at one eval_date. Used by per-instrument parallel loop.
    Returns list of signals with .instrument set; caller dedupes and merges.
    """
    lookback_start = eval_date - timedelta(days=lookback_days)
    historical_data = data_inst[
        (data_inst.index >= lookback_start) & (data_inst.index <= eval_date)
    ]
    if len(historical_data) < min_history:
        return []
    signals, _indicator_df, all_waves = signal_detector.detect_signals_with_indicators(
        historical_data, timings=None
    )
    if not signals:
        return []
    signals_with_targets = target_calculator.calculate_targets(
        signals,
        historical_data,
        all_waves=all_waves if getattr(config, "use_wave_relationship_targets", True) else None,
    )
    for s in signals_with_targets:
        s.instrument = inst
    return signals_with_targets


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
    ):
        """
        Save the indicators log to CSV with optional configuration metadata.

        When result is provided, adds a quality_factor column per date from positions
        opened that day (first trade's quality_factor for that date).
        
        Args:
            output_dir: Directory to save the CSV file
            filename_prefix: Prefix for the filename
            config: Optional strategy configuration to include as metadata
            result: Optional walk-forward result to add quality_factor from positions
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

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.csv"
        filepath = output_dir / filename

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
                signals_with_targets = target_calculator.calculate_targets(
                    signals,
                    historical_data,
                    all_waves=all_waves if getattr(config, "use_wave_relationship_targets", True) else None,
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
        
        if not config.instruments:
            raise ValueError("Config must specify at least one instrument")
        
        # Single instrument: use standard evaluation
        if len(config.instruments) == 1:
            instrument = config.instruments[0]
            lookback_days = getattr(config, 'lookback_days', self.lookback_days)
            load_start = config.start_date
            if config.start_date:
                eval_start_ts = pd.Timestamp(config.start_date)
                load_start = (eval_start_ts - timedelta(days=lookback_days + LOOKBACK_CALENDAR_BUFFER_DAYS)).strftime('%Y-%m-%d')
            if verbose:
                print(f"Evaluating on 1 instrument: {instrument}")
                print(f"Date range: {config.start_date or 'earliest'} to {config.end_date or 'latest'}")
            
            data = DataLoader.from_instrument(
                instrument,
                start_date=load_start,
                end_date=config.end_date,
                column=config.column
            )
            
            if config.start_date and data is not None and len(data) > 0:
                data_start = data.index.min()
                eval_start = pd.Timestamp(config.start_date)
                requested_start = eval_start - timedelta(days=lookback_days)
                if data_start > requested_start:
                    actual_days = max(0, (eval_start - data_start).days)
                    print(f"  Requested {lookback_days} days of history before evaluation start but only {actual_days} days available; early evaluation dates may have reduced lookback.")
            
            start_date = pd.Timestamp(config.start_date) if config.start_date else None
            end_date = pd.Timestamp(config.end_date) if config.end_date else None
            
            return self.evaluate(
                data,
                config,
                start_date=start_date,
                end_date=end_date,
                verbose=verbose
            )
        
        # Multiple instruments: load all, generate signals per instrument, one portfolio
        step_days = getattr(config, 'step_days', self.step_days)
        lookback_days = getattr(config, 'lookback_days', self.lookback_days)
        min_history = 30  # minimum bars for pattern detection
        load_start = config.start_date
        if config.start_date:
            eval_start_ts = pd.Timestamp(config.start_date)
            load_start = (eval_start_ts - timedelta(days=lookback_days + LOOKBACK_CALENDAR_BUFFER_DAYS)).strftime('%Y-%m-%d')
        performance_timings: Dict[str, float] = {}

        t0_load = time.perf_counter()
        data_by_instrument: Dict[str, pd.Series] = {}
        for inst in config.instruments:
            data_by_instrument[inst] = DataLoader.from_instrument(
                inst,
                start_date=load_start,
                end_date=config.end_date,
                column=config.column
            )
        performance_timings["data_load"] = time.perf_counter() - t0_load

        first_inst = config.instruments[0]
        data_first = data_by_instrument[first_inst]
        data_start = data_first.index.min()
        data_end = data_first.index.max()
        start_date = pd.Timestamp(config.start_date) if config.start_date else (data_start + timedelta(days=self.min_history_days))
        end_date = pd.Timestamp(config.end_date) if config.end_date else data_end
        if start_date < data_start + timedelta(days=self.min_history_days):
            start_date = data_start + timedelta(days=self.min_history_days)

        if config.start_date and data_first is not None and len(data_first) > 0:
            eval_start = pd.Timestamp(config.start_date)
            requested_start = eval_start - timedelta(days=lookback_days)
            if data_start > requested_start:
                actual_days = max(0, (eval_start - data_start).days)
                print(f"  Requested {lookback_days} days of history before evaluation start but only {actual_days} days available; early evaluation dates may have reduced lookback.")

        if verbose:
            print(f"Evaluating on {len(config.instruments)} instruments: {config.instruments}")
            print(f"  Date range: {start_date.date()} to {end_date.date()}")
            print(f"  Lookback: {lookback_days} days, Step: {step_days} days")

        eval_dates = self._generate_eval_dates(data_first, start_date, end_date, step_days)
        if verbose:
            print(f"  Evaluation points: {len(eval_dates)}")

        all_signals = []
        seen_signal_keys = set()
        target_calculator = TargetCalculator(
            risk_reward_ratio=config.risk_reward,
            use_atr_stops=getattr(config, 'use_atr_stops', False),
            atr_stop_multiplier=getattr(config, 'atr_stop_multiplier', 2.0),
            atr_period=getattr(config, 'atr_period', 14)
        )
        signal_detector = SignalDetector(_signal_config_from_strategy(config))
        total_eval_dates = len(eval_dates)
        progress_interval = max(1, total_eval_dates // 10)
        start_time = time.time()

        use_parallel = len(config.instruments) > 1 and (max_workers is None or max_workers > 1)
        workers = (
            min(len(config.instruments), max_workers or (os.cpu_count() or 1))
            if use_parallel
            else 1
        )

        if use_parallel and verbose:
            print(f"  Signal detection: {workers} parallel workers (per instrument at each date)", flush=True)

        for i, eval_date in enumerate(eval_dates):
            if use_parallel:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            _signals_for_instrument_at_date,
                            eval_date,
                            inst,
                            data_by_instrument[inst],
                            config,
                            lookback_days,
                            min_history,
                            signal_detector,
                            target_calculator,
                        ): inst
                        for inst in config.instruments
                    }
                    for future in as_completed(futures):
                        signals_batch = future.result()
                        for signal in signals_batch:
                            signal_key = (signal.timestamp, signal.signal_type, signal.price, signal.instrument)
                            if signal_key in seen_signal_keys:
                                continue
                            seen_signal_keys.add(signal_key)
                            all_signals.append(signal)
            else:
                lookback_start = eval_date - timedelta(days=lookback_days)
                for inst in config.instruments:
                    data_inst = data_by_instrument[inst]
                    historical_data = data_inst[(data_inst.index >= lookback_start) & (data_inst.index <= eval_date)]
                    if len(historical_data) < min_history:
                        continue
                    signals, indicator_df, all_waves = signal_detector.detect_signals_with_indicators(
                        historical_data, timings=performance_timings
                    )
                    if not signals:
                        continue
                    signals_with_targets = target_calculator.calculate_targets(
                        signals,
                        historical_data,
                        all_waves=all_waves if getattr(config, "use_wave_relationship_targets", True) else None,
                    )
                    for signal in signals_with_targets:
                        signal.instrument = inst
                        signal_key = (signal.timestamp, signal.signal_type, signal.price, inst)
                        if signal_key in seen_signal_keys:
                            continue
                        seen_signal_keys.add(signal_key)
                        all_signals.append(signal)
            if (i + 1) % progress_interval == 0 or i == total_eval_dates - 1:
                pct = ((i + 1) / total_eval_dates) * 100
                elapsed = time.time() - start_time
                print(f"  Progress: {i + 1}/{total_eval_dates} ({pct:.0f}%) - {len(all_signals)} signals - {elapsed:.1f}s", end='\r', flush=True)

        elapsed_total = time.time() - start_time
        performance_timings["signal_detection"] = elapsed_total
        print(f"  Signal detection completed in {elapsed_total:.1f}s" + " " * 30, flush=True)
        print(f"  Total unique signals detected: {len(all_signals)}")

        all_signals = sorted(all_signals, key=lambda s: s.timestamp)
        prices_by_instrument = {
            inst: data_by_instrument[inst][
                (data_by_instrument[inst].index >= start_date) & (data_by_instrument[inst].index <= end_date)
            ]
            for inst in config.instruments
        }

        print("  Simulating portfolio...", flush=True)
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
        print(f"  Portfolio simulation completed in {sim_elapsed:.1f}s")

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
            max_workers: Number of parallel workers (default: CPU count - 1)
            
        Returns:
            List of WalkForwardResult for each configuration
        """
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        
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
