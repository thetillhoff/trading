#!/usr/bin/env python3
"""
CLI script for analyzing trading signals from Elliott Wave patterns.

Identifies buy/sell signals and calculates target prices.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
djia_dir = Path('/app/djia')
if not djia_dir.exists():
    djia_dir = current_dir.parent / 'djia'
sys.path.insert(0, str(djia_dir))
sys.path.insert(0, str(current_dir))

from data_loader import DataLoader
from signal_detector import SignalDetector, SignalType
from target_calculator import TargetCalculator
from signal_visualizer import SignalVisualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze trading signals from Elliott Wave patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic signal analysis
  python analyze_signals.py --column Close

  # With date range
  python analyze_signals.py --start-date 2020-01-01 --end-date 2024-12-31 --column Close

  # Custom risk/reward ratio
  python analyze_signals.py --column Close --risk-reward 3.0

  # Show only buy signals
  python analyze_signals.py --column Close --signal-type buy
        """
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (inclusive) in format YYYY-MM-DD'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (inclusive) in format YYYY-MM-DD'
    )
    
    parser.add_argument(
        '--column',
        type=str,
        default='Close',
        help='Column to analyze (default: Close). Available: Close, High, Low, Open, Volume'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.6,
        help='Minimum confidence (0.0-1.0) for wave detection (default: 0.6)'
    )
    
    parser.add_argument(
        '--min-wave-size',
        type=float,
        default=0.05,
        help='Minimum wave size as ratio of price range (default: 0.05 = 5%%)'
    )
    
    parser.add_argument(
        '--risk-reward',
        type=float,
        default=2.0,
        help='Risk/reward ratio for stop-loss calculation (default: 2.0)'
    )
    
    parser.add_argument(
        '--signal-type',
        type=str,
        choices=['buy', 'sell', 'all'],
        default='all',
        help='Type of signals to show (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for charts (default: current directory)'
    )
    
    parser.add_argument(
        '--output-filename',
        type=str,
        help='Output filename (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--title',
        type=str,
        help='Chart title (auto-generated if not specified)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Load data
        print("Loading data...")
        loader = DataLoader.from_scraper('djia')
        df = loader.load()
        print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
        
        # Get column
        if args.column not in df.columns:
            print(f"Error: Column '{args.column}' not found.")
            print(f"Available columns: {', '.join(df.columns)}")
            return 1
        
        data = df[args.column]
        
        # Filter by date range
        if args.start_date or args.end_date:
            if args.start_date:
                start = pd.to_datetime(args.start_date)
                data = data[data.index >= start]
            if args.end_date:
                end = pd.to_datetime(args.end_date)
                data = data[data.index <= end]
            print(f"Filtered to {len(data)} rows")
        
        if len(data) < 50:
            print("Error: Insufficient data for signal analysis (need at least 50 data points)")
            return 1
        
        # Detect signals
        print("\nDetecting trading signals...")
        detector = SignalDetector()
        signals = detector.detect_signals(
            data,
            min_confidence=args.min_confidence,
            min_wave_size_ratio=args.min_wave_size
        )
        
        if not signals:
            print("No trading signals detected with current parameters.")
            print("Try lowering --min-confidence or --min-wave-size")
            return 0
        
        print(f"Found {len(signals)} signals")
        
        # Filter by signal type
        if args.signal_type != 'all':
            signal_type_map = {'buy': SignalType.BUY, 'sell': SignalType.SELL}
            signals = [s for s in signals if s.signal_type == signal_type_map[args.signal_type]]
            print(f"Filtered to {len(signals)} {args.signal_type} signals")
        
        if not signals:
            print(f"No {args.signal_type} signals found")
            return 0
        
        # Calculate targets
        print("Calculating target prices...")
        calculator = TargetCalculator(risk_reward_ratio=args.risk_reward)
        signals_with_targets = calculator.calculate_targets(signals, data)
        
        # Print signal summary
        print("\n" + "="*80)
        print("TRADING SIGNALS SUMMARY")
        print("="*80)
        
        buy_count = len([s for s in signals_with_targets if s.signal_type == SignalType.BUY])
        sell_count = len([s for s in signals_with_targets if s.signal_type == SignalType.SELL])
        
        print(f"\nTotal signals: {len(signals_with_targets)}")
        print(f"  Buy signals: {buy_count}")
        print(f"  Sell signals: {sell_count}")
        
        print("\nSignal Details:")
        for i, signal in enumerate(signals_with_targets, 1):
            signal_type_str = signal.signal_type.value.upper()
            print(f"\n{i}. {signal_type_str} Signal - {signal.timestamp.strftime('%Y-%m-%d')}")
            print(f"   Price: {signal.price:.2f}")
            print(f"   Confidence: {signal.confidence:.2%}")
            if signal.target_price:
                change = (signal.target_price - signal.price) / signal.price * 100
                print(f"   Target: {signal.target_price:.2f} ({change:+.1f}%)")
            if signal.stop_loss:
                risk = abs(signal.stop_loss - signal.price) / signal.price * 100
                print(f"   Stop Loss: {signal.stop_loss:.2f} ({risk:.1f}% risk)")
            print(f"   Reasoning: {signal.reasoning}")
        
        # Create visualization
        print("\nGenerating visualization...")
        visualizer = SignalVisualizer(output_dir=args.output_dir)
        
        # Generate title
        if args.title:
            title = args.title
        else:
            title = f"Trading Signals - {args.column}"
            if args.start_date or args.end_date:
                date_range = f" ({args.start_date or 'start'} to {args.end_date or 'end'})"
                title += date_range
        
        output_path = visualizer.plot_with_signals(
            data,
            signals_with_targets,
            title=title,
            xlabel="Date",
            ylabel=f"{args.column} Price",
            output_filename=args.output_filename
        )
        
        print(f"\nChart saved to: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
