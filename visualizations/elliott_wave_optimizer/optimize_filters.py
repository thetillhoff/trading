#!/usr/bin/env python3
"""
CLI script for optimizing Elliott Wave filter parameters.

Analyzes data across different granularities and recommends optimal filter values.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
# djia directory is mounted at /app/djia in Docker
djia_dir = Path('/app/djia')
if not djia_dir.exists():
    # Fallback for local execution
    djia_dir = current_dir.parent / 'djia'
sys.path.insert(0, str(djia_dir))
sys.path.insert(0, str(current_dir))

from data_loader import DataLoader
from granularity_analyzer import GranularityAnalyzer, Granularity
from filter_optimizer import FilterOptimizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize Elliott Wave filter parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-optimize filters
  python optimize_filters.py

  # Optimize for specific wave count
  python optimize_filters.py --target-waves 10

  # Analyze specific granularity
  python optimize_filters.py --granularity monthly

  # With date range
  python optimize_filters.py --start-date 2020-01-01 --end-date 2024-12-31

  # Show detailed analysis
  python optimize_filters.py --verbose
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
        '--target-waves',
        type=int,
        help='Target number of waves to display (auto if not specified)'
    )
    
    parser.add_argument(
        '--granularity',
        type=str,
        choices=['yearly', 'quarterly', 'monthly', 'weekly', 'daily'],
        help='Preferred granularity for analysis (auto if not specified)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed analysis for all granularities'
    )
    
    parser.add_argument(
        '--test-filters',
        type=str,
        help='Test specific filters: "confidence=0.7,size=0.1"'
    )
    
    return parser.parse_args()


def print_analysis_results(analyses, verbose=False):
    """Print analysis results for all granularities."""
    print("\n" + "="*80)
    print("GRANULARITY ANALYSIS RESULTS")
    print("="*80)
    
    for granularity in [Granularity.YEARLY, Granularity.QUARTERLY, Granularity.MONTHLY,
                        Granularity.WEEKLY, Granularity.DAILY]:
        if granularity not in analyses:
            continue
        
        analysis = analyses[granularity]
        
        if analysis.data_points == 0:
            continue
        
        print(f"\n{granularity.value.upper()} Granularity:")
        print(f"  Data points: {analysis.data_points}")
        print(f"  Price range: {analysis.price_range:.2f}")
        print(f"  Waves detected: {analysis.waves_detected}")
        print(f"  Complete patterns: {analysis.complete_patterns}")
        print(f"  Average confidence: {analysis.avg_confidence:.3f}")
        
        if verbose and analysis.wave_size_distribution:
            dist = analysis.wave_size_distribution
            print(f"  Wave size distribution:")
            print(f"    Min: {dist.get('min', 0):.4f} ({dist.get('min', 0)*100:.2f}%)")
            print(f"    Max: {dist.get('max', 0):.4f} ({dist.get('max', 0)*100:.2f}%)")
            print(f"    Mean: {dist.get('mean', 0):.4f} ({dist.get('mean', 0)*100:.2f}%)")
            print(f"    Median: {dist.get('median', 0):.4f} ({dist.get('median', 0)*100:.2f}%)")
        
        print(f"  Recommended min_confidence: {analysis.recommended_min_confidence:.3f}")
        print(f"  Recommended min_wave_size_ratio: {analysis.recommended_min_wave_size_ratio:.4f}")


def print_recommendation(recommendation):
    """Print filter recommendation."""
    print("\n" + "="*80)
    print("RECOMMENDED FILTER VALUES")
    print("="*80)
    print(f"\nMin Confidence: {recommendation.min_confidence:.3f}")
    print(f"Min Wave Size Ratio: {recommendation.min_wave_size_ratio:.4f} ({recommendation.min_wave_size_ratio*100:.2f}%)")
    print(f"Only Complete Patterns: {recommendation.only_complete_patterns}")
    print(f"\nReasoning: {recommendation.reasoning}")
    print(f"Based on: {recommendation.granularity_used.value} granularity")
    print(f"Expected wave count: {recommendation.expected_wave_count}")
    
    print("\n" + "="*80)
    print("USAGE")
    print("="*80)
    print("\nUse these values with the visualization:")
    print(f"make visualize ARGS=\"--granularity daily --column Close --elliott-waves \\")
    print(f"  --min-confidence {recommendation.min_confidence:.3f} \\")
    print(f"  --min-wave-size {recommendation.min_wave_size_ratio:.4f}\"", end="")
    if recommendation.only_complete_patterns:
        print(" \\")
        print(f"  --only-complete-patterns")
    else:
        print()


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
        
        # Initialize analyzer and optimizer
        analyzer = GranularityAnalyzer()
        optimizer = FilterOptimizer()
        
        # Analyze all granularities
        print("\nAnalyzing different time granularities...")
        analyses = analyzer.analyze_all_granularities(data)
        
        if args.verbose:
            print_analysis_results(analyses, verbose=True)
        
        # Test specific filters if requested
        if args.test_filters:
            print("\n" + "="*80)
            print("TESTING SPECIFIC FILTERS")
            print("="*80)
            
            # Parse filter string
            filters = {}
            for part in args.test_filters.split(','):
                key, value = part.split('=')
                filters[key.strip()] = float(value.strip())
            
            min_conf = filters.get('confidence', 0.6)
            min_size = filters.get('size', 0.05)
            only_complete = filters.get('complete', False)
            
            impact = optimizer.analyze_filter_impact(data, min_conf, min_size, only_complete)
            
            print(f"\nFilters: confidence={min_conf}, size={min_size}, complete={only_complete}")
            print(f"Result: {impact['wave_count']} waves detected")
            print(f"  Impulse waves: {impact['impulse_waves']}")
            print(f"  Corrective waves: {impact['corrective_waves']}")
            print(f"  Average confidence: {impact['avg_confidence']:.3f}")
            print(f"  Complete patterns: {impact['complete_patterns']}")
            return 0
        
        # Get granularity preference
        preferred_granularity = None
        if args.granularity:
            granularity_map = {
                'yearly': Granularity.YEARLY,
                'quarterly': Granularity.QUARTERLY,
                'monthly': Granularity.MONTHLY,
                'weekly': Granularity.WEEKLY,
                'daily': Granularity.DAILY
            }
            preferred_granularity = granularity_map[args.granularity]
        
        # Optimize filters
        print("\nOptimizing filter parameters...")
        recommendation = optimizer.optimize_filters(
            data,
            target_wave_count=args.target_waves,
            preferred_granularity=preferred_granularity
        )
        
        # Print results
        if not args.verbose:
            print_analysis_results(analyses, verbose=False)
        
        print_recommendation(recommendation)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
