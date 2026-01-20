#!/usr/bin/env python3
"""
CLI script for evaluating trading signal performance.

Analyzes buy/sell signals to determine if targets or stop-losses were hit
and calculates percentage gains for each trade.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path
from typing import Optional

# Add parent directories to path for imports
current_dir = Path(__file__).parent
djia_dir = Path('/app/djia')
if not djia_dir.exists():
    djia_dir = current_dir.parent / 'djia'
sys.path.insert(0, str(djia_dir))

trading_signals_dir = Path('/app/trading_signals')
if not trading_signals_dir.exists():
    trading_signals_dir = current_dir.parent / 'trading_signals'
sys.path.insert(0, str(trading_signals_dir))

from data_loader import DataLoader
from signal_detector import SignalDetector, SignalType
from target_calculator import TargetCalculator
from trade_evaluator import TradeEvaluator, TradeOutcome
from trade_visualizer import TradeVisualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trading signal performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic trade evaluation
  python evaluate_trades.py --column Close

  # With date range
  python evaluate_trades.py --start-date 2015-01-01 --end-date 2024-12-31 --column Close

  # Evaluate only buy signals
  python evaluate_trades.py --column Close --signal-type buy

  # Limit trade holding period
  python evaluate_trades.py --column Close --max-days 30

  # Custom risk/reward ratio
  python evaluate_trades.py --column Close --risk-reward 3.0
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
        help='Type of signals to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--max-days',
        type=int,
        help='Maximum days to hold a trade (default: until target/stop-loss or data ends)'
    )
    
    parser.add_argument(
        '--require-both-targets',
        action='store_true',
        help='Only evaluate signals with both target and stop-loss (default: evaluate all signals)'
    )
    
    parser.add_argument(
        '--hold-through-stop-loss',
        action='store_true',
        help='When stop-loss is hit, continue holding until recovery (price returns to entry or better)'
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
    
    parser.add_argument(
        '--no-chart',
        action='store_true',
        help='Skip chart generation (only show text output)'
    )
    
    return parser.parse_args()


def calculate_buy_and_hold_gain(data: pd.Series) -> float:
    """Calculate buy-and-hold gain for the data period."""
    if len(data) < 2:
        return 0.0
    start_price = data.iloc[0]
    end_price = data.iloc[-1]
    return ((end_price - start_price) / start_price) * 100


def print_evaluation_summary(summary, buy_and_hold_gain: Optional[float] = None):
    """Print evaluation summary statistics."""
    print("\n" + "="*80)
    print("TRADE EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nTotal Trades Evaluated: {summary.total_trades}")
    print(f"  Winning Trades (Target Hit): {summary.winning_trades}")
    print(f"  Losing Trades (Stop-Loss Hit): {summary.losing_trades}")
    print(f"  No Outcome: {summary.no_outcome_trades}")
    
    print(f"\nWin Rate: {summary.win_rate:.1f}%")
    
    if summary.winning_trades > 0:
        print(f"Average Gain (Winners): {summary.average_gain:+.2f}%")
    if summary.losing_trades > 0:
        print(f"Average Loss (Losers): {summary.average_loss:+.2f}%")
    
    print(f"Total Gain/Loss (All Trades): {summary.total_gain:+.2f}%")
    
    if buy_and_hold_gain is not None:
        print(f"\nBuy-and-Hold Gain (Period): {buy_and_hold_gain:+.2f}%")
        outperformance = summary.total_gain - buy_and_hold_gain
        print(f"Outperformance vs Buy-and-Hold: {outperformance:+.2f}%")
        if outperformance > 0:
            print(f"  → Trading strategy outperformed buy-and-hold by {outperformance:.2f}%")
        elif outperformance < 0:
            print(f"  → Trading strategy underperformed buy-and-hold by {abs(outperformance):.2f}%")
        else:
            print(f"  → Trading strategy matched buy-and-hold performance")
    
    if summary.average_days_held:
        print(f"Average Days Held: {summary.average_days_held:.1f} days")
    
    # Best and worst trade details are available in the evaluation results
    # but not printed to keep output concise - see chart visualization for details


def print_trade_details(evaluations):
    """Print detailed information for each trade."""
    print("\n" + "="*80)
    print("DETAILED TRADE EVALUATIONS")
    print("="*80)
    
    for i, eval_result in enumerate(evaluations, 1):
        signal = eval_result.signal
        print(f"\n{i}. {signal.signal_type.value.upper()} Signal - {signal.timestamp.strftime('%Y-%m-%d')}")
        print(f"   Entry Price: {signal.price:.2f}")
        print(f"   Target: {signal.target_price:.2f}" if signal.target_price else "   Target: N/A")
        print(f"   Stop Loss: {signal.stop_loss:.2f}" if signal.stop_loss else "   Stop Loss: N/A")
        print(f"   Confidence: {signal.confidence:.2%}")
        
        print(f"\n   Outcome: {eval_result.outcome.value}")
        if eval_result.exit_price:
            print(f"   Exit Price: {eval_result.exit_price:.2f}")
        if eval_result.exit_timestamp:
            print(f"   Exit Date: {eval_result.exit_timestamp.strftime('%Y-%m-%d')}")
        
        print(f"   Gain/Loss: {eval_result.gain_percentage:+.2f}%")
        
        if eval_result.days_held is not None:
            print(f"   Days Held: {eval_result.days_held}")
        
        print(f"   Max Favorable: {eval_result.max_favorable_excursion:+.2f}%")
        print(f"   Max Adverse: {eval_result.max_adverse_excursion:+.2f}%")
        
        print(f"   Reasoning: {signal.reasoning}")


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
        
        # Determine data granularity (daily, weekly, etc.)
        # Check time difference between consecutive data points
        if len(data) > 1:
            time_diffs = data.index.to_series().diff().dropna()
            median_diff = time_diffs.median()
            if median_diff.days <= 1:
                granularity = "daily"
            elif median_diff.days <= 7:
                granularity = "weekly"
            elif median_diff.days <= 31:
                granularity = "monthly"
            else:
                granularity = "other"
            print(f"Data granularity: {granularity} (median interval: {median_diff.days} days)")
        else:
            granularity = "unknown"
            print("Data granularity: unknown (insufficient data points)")
        
        if len(data) < 50:
            print("Error: Insufficient data for signal analysis (need at least 50 data points)")
            return 1
        
        # Detect signals
        print("\nDetecting trading signals...")
        print(f"Elliott Wave detection parameters:")
        print(f"  - Data granularity: {granularity}")
        print(f"  - Min confidence: {args.min_confidence}")
        print(f"  - Min wave size ratio: {args.min_wave_size} ({args.min_wave_size*100:.1f}% of price range)")
        print(f"  - Min wave length: None (no restriction by default)")
        
        # First, detect waves to see how many we have
        from elliott_wave_detector import ElliottWaveDetector
        wave_detector = ElliottWaveDetector()
        all_waves = wave_detector.detect_waves(
            data,
            min_confidence=0.0,  # Get all waves first
            min_wave_size_ratio=0.0,
            only_complete_patterns=False
        )
        print(f"  - Total waves detected (no filters): {len(all_waves)}")
        
        # Now detect signals with filters
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
        
        print(f"Found {len(signals)} signals (from {len(all_waves)} total waves)")
        buy_count = len([s for s in signals if s.signal_type.value == 'buy'])
        sell_count = len([s for s in signals if s.signal_type.value == 'sell'])
        print(f"  - Buy signals: {buy_count} (from Wave 2, Wave 4, or Wave B)")
        print(f"  - Sell signals: {sell_count} (from Wave 5 or Wave B)")
        
        # Filter by signal type
        if args.signal_type != 'all':
            signal_type_map = {'buy': SignalType.BUY, 'sell': SignalType.SELL}
            signals = [s for s in signals if s.signal_type == signal_type_map[args.signal_type]]
            print(f"Filtered to {len(signals)} {args.signal_type} signals")
        
        if not signals:
            print(f"No {args.signal_type} signals found")
            return 0
        
        # Calculate targets
        print("\nCalculating target prices and stop-loss levels...")
        calculator = TargetCalculator(risk_reward_ratio=args.risk_reward)
        signals_with_targets = calculator.calculate_targets(signals, data)
        
        # Show target calculation results
        signals_with_target = len([s for s in signals_with_targets if s.target_price])
        signals_with_stop = len([s for s in signals_with_targets if s.stop_loss])
        signals_with_both = len([s for s in signals_with_targets if s.target_price and s.stop_loss])
        print(f"  - Signals with target: {signals_with_target}/{len(signals_with_targets)}")
        print(f"  - Signals with stop-loss: {signals_with_stop}/{len(signals_with_targets)}")
        print(f"  - Signals with both: {signals_with_both}/{len(signals_with_targets)}")
        
        # Filter signals with targets if required
        if args.require_both_targets:
            initial_count = len(signals_with_targets)
            signals_with_targets = [
                s for s in signals_with_targets
                if s.target_price and s.stop_loss
            ]
            filtered_out = initial_count - len(signals_with_targets)
            print(f"  - Filtered out {filtered_out} signals missing target/stop-loss")
            print(f"  - Remaining signals: {len(signals_with_targets)}")
        
        if not signals_with_targets:
            print("No signals to evaluate.")
            if args.require_both_targets:
                print("Try removing --require-both-targets to evaluate all signals")
            return 0
        
        # Evaluate trades
        print(f"\nEvaluating {len(signals_with_targets)} trades...")
        if args.max_days:
            print(f"  Max holding period: {args.max_days} days")
        else:
            print(f"  Max holding period: None (until target/stop-loss or data ends)")
        if args.hold_through_stop_loss:
            print(f"  Hold through stop-loss: Enabled (will continue holding after stop-loss until recovery)")
        evaluator = TradeEvaluator(
            max_days=args.max_days,
            require_both_targets=args.require_both_targets,
            hold_through_stop_loss=args.hold_through_stop_loss
        )
        evaluations = evaluator.evaluate_signals(signals_with_targets, data)
        
        if not evaluations:
            print("No trades could be evaluated.")
            return 0
        
        # Generate summary
        summary = evaluator.summarize_evaluations(evaluations)
        
        # Calculate buy-and-hold gain for comparison
        buy_and_hold_gain = calculate_buy_and_hold_gain(data)
        
        # Print results
        print_evaluation_summary(summary, buy_and_hold_gain)
        # Detailed trade information is available in code but not printed to stdout
        # Uncomment the line below if you need detailed trade-by-trade output:
        # print_trade_details(evaluations)
        
        # Generate visualization
        if not args.no_chart:
            print("\nGenerating visualization...")
            visualizer = TradeVisualizer(output_dir=args.output_dir)
            
            # Generate title
            if args.title:
                title = args.title
            else:
                title = f"Trade Evaluation - {args.column}"
                if args.start_date or args.end_date:
                    date_range = f" ({args.start_date or 'start'} to {args.end_date or 'end'})"
                    title += date_range
            
            output_path = visualizer.plot_evaluation(
                data,
                evaluations,
                summary,
                title=title,
                xlabel="Date",
                ylabel=f"{args.column} Price",
                output_filename=args.output_filename,
                buy_and_hold_gain=buy_and_hold_gain
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
