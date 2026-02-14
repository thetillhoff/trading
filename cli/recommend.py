#!/usr/bin/env python3
"""
Daily trade recommendation CLI.

Analyzes all instruments from baseline config and recommends the best
trade opportunity for today based on the latest available data.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from core.signals.config_loader import load_config_from_yaml
from core.signals.detector import SignalDetector
from core.signals.target_calculator import TargetCalculator
from core.data.loader import DataLoader
from core.shared.types import TradingSignal, SignalType


def get_latest_signal_for_instrument(
    instrument: str,
    config,
    lookback_days: int,
    target_date: datetime,
    column: str = "Close"
) -> Optional[Tuple[TradingSignal, datetime, str]]:
    """
    Get the most recent signal for an instrument as of target_date.
    
    Args:
        instrument: Instrument name or ticker
        config: Strategy config
        lookback_days: Number of days to look back
        target_date: Date to simulate "today" (for historical recommendations)
        column: Price column to use
    
    Returns:
        Tuple of (signal, data_date, warning) or None if no signal/data
        warning is empty string if no issues, otherwise contains warning message
    """
    try:
        # Calculate date range from target_date
        start_date = target_date - timedelta(days=lookback_days)
        
        # Load data up to target_date
        data = DataLoader.from_instrument(
            instrument,
            start_date=str(start_date.date()),
            end_date=str(target_date.date()),
            column=column
        )
        
        if data is None or len(data) == 0:
            return None
        
        # Get the most recent date in the data (should be <= target_date)
        latest_date = data.index[-1]
        days_old = (target_date.date() - latest_date.date()).days
        
        # Create warning if data is stale relative to target_date
        warning = ""
        if days_old > 3:
            warning = f"Data is {days_old} days old"
        
        # Detect signals
        detector = SignalDetector(config)
        signals = detector.detect_signals(data)
        
        if not signals:
            return None
        
        # Calculate targets and stop-loss for signals
        target_calculator = TargetCalculator(
            risk_reward_ratio=getattr(config, 'risk_reward', 3.0),
            use_atr_stops=True,
            atr_stop_multiplier=2.0
        )
        signals = target_calculator.calculate_targets(signals, data)
        
        # Filter signals to only those on the most recent date
        recent_signals = [s for s in signals if s.timestamp.date() == latest_date.date()]
        
        if not recent_signals:
            return None
        
        # Return the highest confidence signal for this instrument
        best_signal = max(recent_signals, key=lambda s: s.confirmation_score if s.confirmation_score else s.confidence)
        
        return (best_signal, latest_date, warning)
        
    except FileNotFoundError:
        # No data file for this instrument
        return None
    except Exception as e:
        print(f"  Warning: Error processing {instrument}: {e}", file=sys.stderr)
        return None


def format_signal_output(
    instrument: str,
    signal: TradingSignal,
    data_date: datetime,
    warning: str
) -> str:
    """Format a signal for display."""
    lines = []
    
    lines.append(f"Best opportunity:")
    lines.append(f"  Instrument: {instrument}")
    lines.append(f"  Action: {signal.signal_type.value.upper()}")
    lines.append(f"  Price: ${signal.price:.2f}")
    
    # Use confirmation_score if available, otherwise confidence
    confidence_value = signal.confirmation_score if signal.confirmation_score is not None else signal.confidence
    lines.append(f"  Confidence: {confidence_value:.2f} ({confidence_value*100:.0f}%)")
    
    if signal.stop_loss:
        lines.append(f"  Stop-loss: ${signal.stop_loss:.2f}")
    if signal.target_price:
        lines.append(f"  Target: ${signal.target_price:.2f}")
        
        # Calculate risk/reward
        if signal.stop_loss:
            if signal.signal_type == SignalType.BUY:
                risk = signal.price - signal.stop_loss
                reward = signal.target_price - signal.price
            else:  # SELL
                risk = signal.stop_loss - signal.price
                reward = signal.price - signal.target_price
            
            if risk > 0:
                risk_reward = reward / risk
                lines.append(f"  Risk/Reward: {risk_reward:.1f}")
    
    if signal.reasoning:
        lines.append(f"  Reasoning: {signal.reasoning}")
    
    # Add data freshness note if applicable
    if warning:
        lines.append(f"  Note: {warning}")
    
    # Add signal date if not today
    signal_date = signal.timestamp.date()
    today = datetime.now().date()
    if signal_date != today:
        lines.append(f"  Signal date: {signal_date}")
    
    return "\n".join(lines)


def main():
    """Main entry point for daily trade recommendation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Get the best trade recommendation based on baseline config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Get today's recommendation
    python -m cli.recommend
    
    # Simulate recommendation for a specific date (backtesting)
    python -m cli.recommend --date 2026-01-15
    
    # Simulate recommendation for yesterday
    python -m cli.recommend --date 2026-02-13
        """
    )
    
    parser.add_argument(
        "--date",
        type=str,
        help="Simulate recommendation for this date (YYYY-MM-DD). Default: today"
    )
    
    args = parser.parse_args()
    
    # Parse target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
            simulating = True
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD")
            return 1
    else:
        target_date = datetime.now()
        simulating = False
    
    print("=" * 60)
    print("DAILY TRADE RECOMMENDATION")
    print("=" * 60)
    
    # Load baseline config
    config_path = Path("configs/baseline.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    try:
        config = load_config_from_yaml(str(config_path))
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Display analysis parameters
    start_date = target_date - timedelta(days=config.lookback_days)
    print(f"Date: {target_date.date()}" + (" (SIMULATED)" if simulating else ""))
    print(f"Analysis window: {start_date.date()} to {target_date.date()}")
    print(f"Instruments: {len(config.instruments)}")
    print()
    
    # Analyze each instrument
    print("Analyzing instruments...")
    all_signals = []
    skipped = []
    
    for instrument in config.instruments:
        result = get_latest_signal_for_instrument(
            instrument,
            config,
            config.lookback_days,
            target_date,
            column=getattr(config, 'column', 'Close')
        )
        
        if result is None:
            skipped.append(instrument)
        else:
            signal, data_date, warning = result
            all_signals.append((instrument, signal, data_date, warning))
    
    print(f"  Processed: {len(all_signals)}/{len(config.instruments)} instruments")
    if skipped:
        print(f"  Skipped {len(skipped)} instruments (no data or signals): {', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}")
    print()
    
    # Select and display best signal
    if not all_signals:
        print("No signals detected" + (" on this date." if simulating else " today."))
        print()
        print("Possible reasons:")
        print("  - No downloaded data for configured instruments")
        print("  - No trading opportunities meeting criteria")
        print()
        if not simulating:
            print("Try: make download to update data")
        return 0
    
    # Find best signal (highest confidence/confirmation_score)
    best = max(
        all_signals,
        key=lambda x: x[1].confirmation_score if x[1].confirmation_score is not None else x[1].confidence
    )
    
    instrument, signal, data_date, warning = best
    
    print(format_signal_output(instrument, signal, data_date, warning))
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
