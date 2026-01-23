#!/usr/bin/env python3
"""
Generate trade timeline visualization from CSV file.

Allows generating timeline charts from previously saved trades CSV files.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from datetime import datetime

# Add core to path
core_dir = Path(__file__).parent.parent / "core"
sys.path.insert(0, str(core_dir.parent))

from core.data.loader import DataLoader
from core.grid_test.reporter import ComparisonReporter
from core.evaluation.portfolio import Position, PositionStatus


def load_trades_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load trades from CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl', 'status']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    return df


def generate_timeline_from_csv(
    csv_path: Path,
    instrument: str = 'djia',
    output_dir: Optional[Path] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    show_annotations: bool = True,
    annotation_threshold: float = 0.0,
    annotation_top_n: Optional[int] = None,
) -> str:
    """
    Generate timeline visualization from trades CSV.
    
    Args:
        csv_path: Path to trades CSV file
        instrument: Instrument name for loading price data
        output_dir: Output directory for chart
        start_date: Start date for price data (optional, auto-detect from CSV)
        end_date: End date for price data (optional, auto-detect from CSV)
        
    Returns:
        Path to generated chart
    """
    # Load trades
    trades_df = load_trades_from_csv(csv_path)
    
    # Determine date range from trades if not provided
    if not start_date:
        start_date = trades_df['entry_date'].min()
    if not end_date:
        # Use exit_date if available, otherwise entry_date
        if 'exit_date' in trades_df.columns and trades_df['exit_date'].notna().any():
            end_date = trades_df['exit_date'].max()
        else:
            end_date = trades_df['entry_date'].max()
    
    # Load price data
    try:
        loader = DataLoader.from_scraper(instrument)
        price_df = loader.load(start_date=start_date, end_date=end_date)
        price_data = price_df['Close'] if 'Close' in price_df.columns else price_df.iloc[:, 0]
    except Exception as e:
        print(f"Error loading price data: {e}")
        return ""
    
    # Convert dates to datetime
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'], errors='coerce')
    
    # Convert CSV rows to Position-like dicts for the shared plotting method
    positions = []
    for _, row in trades_df.iterrows():
        cost_basis = row['cost_basis'] if pd.notna(row.get('cost_basis', 0)) else 1
        pnl = row['pnl'] if pd.notna(row['pnl']) else 0
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
        positions.append({
            'entry_date': row['entry_date'] if pd.notna(row['entry_date']) else None,
            'entry_price': row['entry_price'] if pd.notna(row['entry_price']) else None,
            'exit_date': row['exit_date'] if pd.notna(row['exit_date']) else None,
            'exit_price': row['exit_price'] if pd.notna(row['exit_price']) else None,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'cost_basis': cost_basis,
            'status': row.get('status', 'closed_end'),
        })
    
    # Create high-resolution figure (larger size for zooming)
    fig, ax = plt.subplots(figsize=(24, 14))
    
    # Use shared plotting method from reporter
    reporter = ComparisonReporter(output_dir=str(output_dir) if output_dir else None)
    title = f'Trade Timeline from CSV: {csv_path.stem} | {len(positions)} trades'
    
    # Auto-calculate top_n if not provided (20% of trades, min 50)
    if annotation_top_n is None and len(positions) > 50:
        annotation_top_n = max(50, int(len(positions) * 0.2))
    
    reporter._plot_trade_timeline(
        ax, price_data, positions, 
        title=title, 
        show_annotations=show_annotations,
        annotation_threshold_pct=annotation_threshold,
        annotation_top_n=annotation_top_n,
    )
    
    # Add summary
    winning = [p for p in positions if p.get('pnl', 0) > 0 and p.get('exit_date')]
    losing = [p for p in positions if p.get('pnl', 0) <= 0 and p.get('exit_date') and p.get('pnl', 0) < 0]
    total_trades = len(positions)
    winning_count = len(winning)
    losing_count = len(losing)
    win_rate = (winning_count / total_trades * 100) if total_trades > 0 else 0
    
    summary_text = (
        f"Total: {total_trades} | "
        f"Wins: {winning_count} ({winning_count/total_trades*100:.1f}%) | "
        f"Losses: {losing_count} ({losing_count/total_trades*100:.1f}%)"
    )
    ax.text(0.02, 0.02, summary_text, transform=ax.transAxes,
           fontsize=7, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    
    # Save chart with high DPI for zooming
    output_dir = output_dir or csv_path.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"timeline_from_csv_{timestamp}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=600, bbox_inches='tight')  # High DPI for zooming
    plt.close()
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate trade timeline visualization from CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate timeline from trades CSV
    python -m cli.timeline trades_baseline_20260122_123456.csv

    # Specify instrument and output directory
    python -m cli.timeline trades_baseline_20260122_123456.csv --instrument sp500 --output-dir ./charts
        """
    )
    
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to trades CSV file",
    )
    parser.add_argument(
        "--instrument", "-i",
        default="djia",
        help="Instrument for price data (default: djia)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for chart (default: same as CSV)",
    )
    parser.add_argument(
        "--start-date", "-s",
        type=str,
        help="Start date for price data (YYYY-MM-DD, default: auto from CSV)",
    )
    parser.add_argument(
        "--end-date", "-e",
        type=str,
        help="End date for price data (YYYY-MM-DD, default: auto from CSV)",
    )
    parser.add_argument(
        "--show-annotations",
        action="store_true",
        default=True,
        help="Show P&L annotations on significant trades (default: True)",
    )
    parser.add_argument(
        "--annotation-threshold",
        type=float,
        default=0.0,
        help="Only show annotations for trades above this % threshold (default: 0.0 = show all if --show-annotations)",
    )
    parser.add_argument(
        "--annotation-top-n",
        type=int,
        default=None,
        help="Only show annotations for top N winning and bottom N losing trades (default: auto-calculated as 20% of total)",
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    try:
        chart_path = generate_timeline_from_csv(
            csv_path,
            instrument=args.instrument,
            output_dir=output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            show_annotations=args.show_annotations,
            annotation_threshold=args.annotation_threshold,
            annotation_top_n=args.annotation_top_n,
        )
        
        if chart_path:
            print(f"Timeline chart generated: {chart_path}")
            return 0
        else:
            print("Failed to generate chart")
            return 1
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
