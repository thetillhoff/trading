#!/usr/bin/env python3
"""
Main visualization script for DJIA data.

Provides CLI interface for generating visualizations with customizable parameters.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from data_loader import DataLoader
from data_processor import DataProcessor, Granularity, AggregationMethod
from visualizer import Visualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize DJIA trading data with customizable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Daily close prices for 2023
  python visualize_djia.py --start-date 2023-01-01 --end-date 2023-12-31 --granularity daily --column Close

  # Daily prices with Elliott Wave detection
  python visualize_djia.py --granularity daily --column Close --elliott-waves

  # Monthly average prices for all time
  python visualize_djia.py --granularity monthly --aggregation mean --column Close

  # Weekly maximum high prices
  python visualize_djia.py --granularity weekly --aggregation max --column High
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
        '--column',
        type=str,
        default='Close',
        help='Column to visualize (default: Close). Available: Close, High, Low, Open, Volume'
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
        '--elliott-waves',
        action='store_true',
        help='Enable Elliott Wave detection and color coding'
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
        '--only-complete-patterns',
        action='store_true',
        help='Only show complete 5-wave or 3-wave patterns'
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
        
        # Process data
        processor = DataProcessor(df)
        
        # Filter by date range
        if args.start_date or args.end_date:
            df = processor.filter_date_range(args.start_date, args.end_date)
            print(f"Filtered to {len(df)} rows")
        
        # Update processor with filtered data
        processor = DataProcessor(df)
        
        # Determine column
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
        print(f"Value range: {data.min():.2f} to {data.max():.2f}")
        
        # Generate title
        if args.title:
            title = args.title
        else:
            agg_str = args.aggregation.capitalize() if args.granularity != 'daily' else ''
            gran_str = args.granularity.capitalize()
            title = f"DJIA {column} - {agg_str} {gran_str}".strip()
            if args.start_date or args.end_date:
                date_range = f" ({args.start_date or 'start'} to {args.end_date or 'end'})"
                title += date_range
        
        # Create visualization
        print("Generating visualization...")
        visualizer = Visualizer(output_dir=args.output_dir)
        
        if args.elliott_waves:
            # Use Elliott Wave visualization
            if args.granularity != 'daily':
                print("Warning: Elliott Wave detection works best with daily data. "
                      "Consider using --granularity daily for better results.")
            output_path = visualizer.plot_line_with_elliott_waves(
                data,
                title=title + " (with Elliott Waves)",
                xlabel="Date",
                ylabel=f"{column} ({aggregation.value if args.granularity != 'daily' else ''})".strip(),
                output_filename=args.output_filename,
                show_waves=True,
                min_confidence=args.min_confidence,
                min_wave_size_ratio=args.min_wave_size,
                only_complete_patterns=args.only_complete_patterns
            )
        else:
            # Use standard line chart
            output_path = visualizer.plot_line(
                data,
                title=title,
                xlabel="Date",
                ylabel=f"{column} ({aggregation.value if args.granularity != 'daily' else ''})".strip(),
                output_filename=args.output_filename
            )
        
        print(f"Chart saved to: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
