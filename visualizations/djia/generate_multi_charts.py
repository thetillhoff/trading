#!/usr/bin/env python3
"""
Generate multiple visualization charts with shared parameters.

Can generate Elliott Wave and Trading Signals charts separately or combined.
"""
import argparse
import sys
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
djia_dir = Path('/app/djia')
if not djia_dir.exists():
    djia_dir = current_dir
sys.path.insert(0, str(djia_dir))

from data_loader import DataLoader
from data_processor import DataProcessor, Granularity, AggregationMethod
from visualizer import Visualizer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multiple visualization charts with shared parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both Elliott Waves and Trading Signals separately
  python generate_multi_charts.py --start-date 2015-01-01 --end-date 2024-12-31 --column Close

  # Generate combined chart (both in one image)
  python generate_multi_charts.py --column Close --combined

  # Generate only specific chart types
  python generate_multi_charts.py --column Close --elliott-waves --no-trading-signals
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
        help='Column to visualize (default: Close). Available: Close, High, Low, Open, Volume'
    )
    
    parser.add_argument(
        '--granularity',
        type=str,
        choices=['daily', 'weekly', 'monthly', 'yearly'],
        default='daily',
        help='Time granularity (default: daily)'
    )
    
    parser.add_argument(
        '--aggregation',
        type=str,
        choices=['mean', 'max', 'min', 'median', 'sum', 'first', 'last'],
        default='mean',
        help='Aggregation method (default: mean)'
    )
    
    parser.add_argument(
        '--elliott-waves',
        action='store_true',
        help='Generate Elliott Wave chart'
    )
    
    parser.add_argument(
        '--no-elliott-waves',
        action='store_true',
        help='Skip Elliott Wave chart'
    )
    
    parser.add_argument(
        '--trading-signals',
        action='store_true',
        help='Generate Trading Signals chart'
    )
    
    parser.add_argument(
        '--no-trading-signals',
        action='store_true',
        help='Skip Trading Signals chart'
    )
    
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Generate combined chart (both visualizations in one image, stacked vertically)'
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
        help='Risk/reward ratio for trading signals (default: 2.0)'
    )
    
    parser.add_argument(
        '--only-complete-patterns',
        action='store_true',
        help='Only show complete 5-wave or 3-wave patterns'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for charts (default: current directory)'
    )
    
    parser.add_argument(
        '--title-prefix',
        type=str,
        help='Prefix for chart titles (default: auto-generated)'
    )
    
    return parser.parse_args()


def generate_combined_chart(
    data: pd.Series,
    visualizer: Visualizer,
    args,
    output_dir: Path
) -> Path:
    """Generate a combined chart with Elliott Waves and Trading Signals stacked."""
    # Import trading signals modules - handle both Docker and local paths
    trading_signals_dir = Path('/app/trading_signals')
    if not trading_signals_dir.exists():
        trading_signals_dir = Path(__file__).parent.parent / 'trading_signals'
    sys.path.insert(0, str(trading_signals_dir))
    from signal_detector import SignalDetector, SignalType
    from target_calculator import TargetCalculator
    
    # Import Elliott Wave detector
    from elliott_wave_detector import ElliottWaveDetector, WaveType, WaveLabel
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Top subplot: Elliott Waves
    ax1.plot(data.index, data.values, linewidth=1.5, color='gray', alpha=0.5, label='Price')
    
    detector = ElliottWaveDetector()
    waves = detector.detect_waves(
        data,
        min_confidence=args.min_confidence,
        min_wave_size_ratio=args.min_wave_size,
        only_complete_patterns=args.only_complete_patterns
    )
    print(f"Detected {len(waves)} Elliott waves in combined chart")
    
    if waves:
        impulse_colors = {
            WaveLabel.WAVE_1.value: '#2E7D32',
            WaveLabel.WAVE_2.value: '#D32F2F',
            WaveLabel.WAVE_3.value: '#1976D2',
            WaveLabel.WAVE_4.value: '#FF6F00',
            WaveLabel.WAVE_5.value: '#6A1B9A',
        }
        corrective_colors = {
            WaveLabel.WAVE_A.value: '#E91E63',
            WaveLabel.WAVE_B.value: '#00838F',
            WaveLabel.WAVE_C.value: '#F9A825',
        }
        
        plotted_labels = set()
        for wave in waves:
            if wave.start_idx < len(data) and wave.end_idx < len(data):
                wave_data = data.iloc[wave.start_idx:wave.end_idx+1]
                if wave.wave_type == WaveType.IMPULSE:
                    color = impulse_colors.get(wave.label.value, '#757575')
                else:
                    color = corrective_colors.get(wave.label.value, '#757575')
                
                label = f"Wave {wave.label.value}" if wave.label.value not in plotted_labels else ""
                if label:
                    plotted_labels.add(wave.label.value)
                
                ax1.plot(wave_data.index, wave_data.values, linewidth=2.5, color=color, label=label)
        
        ax1.legend(loc='best', fontsize=9, ncol=2)
    
    ax1.set_title(f"{args.title_prefix or 'Elliott Wave Analysis'} - {args.column}", 
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel(f"{args.column} Price", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: Trading Signals
    ax2.plot(data.index, data.values, linewidth=1.5, color='gray', alpha=0.7, label='Price')
    
    signal_detector = SignalDetector()
    signals = signal_detector.detect_signals(
        data,
        min_confidence=args.min_confidence,
        min_wave_size_ratio=args.min_wave_size
    )
    print(f"Detected {len(signals)} trading signals in combined chart")
    
    if signals:
        calculator = TargetCalculator(risk_reward_ratio=args.risk_reward)
        signals_with_targets = calculator.calculate_targets(signals, data)
        
        buy_signals = [s for s in signals_with_targets if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals_with_targets if s.signal_type == SignalType.SELL]
        
        if buy_signals:
            buy_dates = [s.timestamp for s in buy_signals]
            buy_prices = [s.price for s in buy_signals]
            ax2.scatter(buy_dates, buy_prices, color='green', marker='^', s=200, zorder=5,
                       label='Buy Signal', edgecolors='darkgreen', linewidths=2)
            
            for signal in buy_signals:
                if signal.target_price:
                    ax2.plot([signal.timestamp, signal.timestamp],
                            [signal.price, signal.target_price],
                            color='green', linestyle='--', alpha=0.5, linewidth=1.5)
                    ax2.scatter(signal.timestamp, signal.target_price,
                              color='lightgreen', marker='o', s=100, zorder=4,
                              edgecolors='green', linewidths=1)
                if signal.stop_loss:
                    ax2.scatter(signal.timestamp, signal.stop_loss,
                              color='red', marker='x', s=150, zorder=4, linewidths=2)
        
        if sell_signals:
            sell_dates = [s.timestamp for s in sell_signals]
            sell_prices = [s.price for s in sell_signals]
            ax2.scatter(sell_dates, sell_prices, color='red', marker='v', s=200, zorder=5,
                       label='Sell Signal', edgecolors='darkred', linewidths=2)
            
            for signal in sell_signals:
                if signal.target_price:
                    ax2.plot([signal.timestamp, signal.timestamp],
                            [signal.price, signal.target_price],
                            color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                    ax2.scatter(signal.timestamp, signal.target_price,
                              color='lightcoral', marker='o', s=100, zorder=4,
                              edgecolors='red', linewidths=1)
                if signal.stop_loss:
                    ax2.scatter(signal.timestamp, signal.stop_loss,
                              color='green', marker='x', s=150, zorder=4, linewidths=2)
        
        has_targets = any(s.target_price for s in signals_with_targets)
        has_stop_loss = any(s.stop_loss for s in signals_with_targets)
        
        if has_targets:
            ax2.scatter([], [], color='gray', marker='o', s=100,
                       edgecolors='black', linewidths=1, label='Target Price')
        if has_stop_loss:
            ax2.scatter([], [], color='gray', marker='x', s=150,
                       linewidths=2, label='Stop Loss')
    
    ax2.set_title(f"{args.title_prefix or 'Trading Signals'} - {args.column}",
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel(f"{args.column} Price", fontsize=10)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add description text
    description1 = (
        "Elliott Wave Reading: Impulse waves (1-5) show trend direction; "
        "corrective waves (a-c) show corrections.\n"
        "Wave 2/4 ends = potential buy; Wave 5 end = potential sell."
    )
    description2 = (
        "Trading Signals: Triangles mark entry points (^=buy, v=sell). "
        "Circles = targets; X = stop-loss."
    )
    
    # Adjust layout first to make room for description
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave bottom 5% for description
    fig.text(0.5, 0.02, f"{description1} | {description2}", ha='center', va='bottom',
            fontsize=8, style='italic', color='gray')
    
    # Generate filename
    date_range = ""
    if args.start_date or args.end_date:
        date_range = f"_({args.start_date or 'start'}_to_{args.end_date or 'end'})"
    filename = f"combined_analysis_{args.column.lower()}{date_range}.png"
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Load data
        print("Loading data...")
        loader = DataLoader.from_scraper('djia')
        df = loader.load()
        print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
        
        # Process data
        processor = DataProcessor(df)
        
        # Filter by date range
        if args.start_date or args.end_date:
            df = processor.filter_date_range(args.start_date, args.end_date)
            processor = DataProcessor(df)
            print(f"Filtered to {len(df)} rows")
        
        # Get column
        column = args.column
        if column not in processor.get_available_columns():
            print(f"Error: Column '{column}' not found.")
            print(f"Available columns: {', '.join(processor.get_available_columns())}")
            return 1
        
        # Resample if needed
        granularity_map = {
            'daily': Granularity.DAILY,
            'weekly': Granularity.WEEKLY,
            'monthly': Granularity.MONTHLY,
            'yearly': Granularity.YEARLY
        }
        aggregation_map = {
            'mean': AggregationMethod.MEAN,
            'max': AggregationMethod.MAX,
            'min': AggregationMethod.MIN,
            'median': AggregationMethod.MEDIAN,
            'sum': AggregationMethod.SUM,
            'first': AggregationMethod.FIRST,
            'last': AggregationMethod.LAST
        }
        
        granularity = granularity_map[args.granularity]
        aggregation = aggregation_map[args.aggregation]
        
        if granularity != Granularity.DAILY:
            print(f"Resampling to {args.granularity} with {args.aggregation} aggregation...")
            df_resampled = processor.resample(granularity, aggregation, column)
            data = df_resampled[column] if column in df_resampled.columns else df_resampled.iloc[:, 0]
        else:
            data = processor.get_column(column)
        
        print(f"Data points: {len(data)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        
        # Determine what to generate
        generate_elliott = not args.no_elliott_waves and (args.elliott_waves or not args.trading_signals)
        generate_signals = not args.no_trading_signals and (args.trading_signals or not args.elliott_waves)
        
        # If neither specified, generate both
        if not args.elliott_waves and not args.trading_signals and not args.no_elliott_waves and not args.no_trading_signals:
            generate_elliott = True
            generate_signals = True
        
        visualizer = Visualizer(output_dir=args.output_dir)
        output_paths = []
        
        # Generate combined chart if requested
        if args.combined and generate_elliott and generate_signals:
            print("Generating combined chart...")
            output_path = generate_combined_chart(data, visualizer, args, visualizer.output_dir)
            output_paths.append(output_path)
            print(f"Combined chart saved to: {output_path}")
        else:
            # Generate separate charts
            date_range_str = ""
            if args.start_date or args.end_date:
                date_range_str = f" ({args.start_date or 'start'} to {args.end_date or 'end'})"
            
            if generate_elliott:
                print("Generating Elliott Wave chart...")
                title = f"DJIA {column} - Elliott Waves{date_range_str}"
                output_path = visualizer.plot_line_with_elliott_waves(
                    data,
                    title=title,
                    xlabel="Date",
                    ylabel=f"{column} Price",
                    min_confidence=args.min_confidence,
                    min_wave_size_ratio=args.min_wave_size,
                    only_complete_patterns=args.only_complete_patterns,
                    show_waves=True
                )
                output_paths.append(output_path)
                print(f"Elliott Wave chart saved to: {output_path}")
            
            if generate_signals:
                print("Generating Trading Signals chart...")
                # Import trading signals modules - handle both Docker and local paths
                trading_signals_dir = Path('/app/trading_signals')
                if not trading_signals_dir.exists():
                    trading_signals_dir = Path(__file__).parent.parent / 'trading_signals'
                sys.path.insert(0, str(trading_signals_dir))
                from signal_detector import SignalDetector
                from target_calculator import TargetCalculator
                from signal_visualizer import SignalVisualizer
                
                signal_detector = SignalDetector()
                signals = signal_detector.detect_signals(
                    data,
                    min_confidence=args.min_confidence,
                    min_wave_size_ratio=args.min_wave_size
                )
                
                if signals:
                    calculator = TargetCalculator(risk_reward_ratio=args.risk_reward)
                    signals_with_targets = calculator.calculate_targets(signals, data)
                    
                    signal_visualizer = SignalVisualizer(output_dir=args.output_dir)
                    title = f"Trading Signals - {column}{date_range_str}"
                    output_path = signal_visualizer.plot_with_signals(
                        data,
                        signals_with_targets,
                        title=title,
                        xlabel="Date",
                        ylabel=f"{column} Price"
                    )
                    output_paths.append(output_path)
                    print(f"Trading Signals chart saved to: {output_path}")
                else:
                    print("No trading signals detected.")
        
        print(f"\nGenerated {len(output_paths)} chart(s)")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
