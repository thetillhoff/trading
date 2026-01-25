#!/usr/bin/env python3
"""
Analyze grid-test CSV results across all periods.

Aggregates results from multiple period subdirectories and generates
comprehensive analysis for hypothesis testing.

Usage:
    docker compose run --rm cli python scripts/analyze_grid_test_results.py results/hypothesis_tests_YYYYMMDD_HHMMSS/
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def load_period_results(results_dir: Path) -> pd.DataFrame:
    """Load all CSV results from period subdirectories."""
    all_results = []
    
    # Find all period subdirectories
    period_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(period_dirs)} period directories")
    
    for period_dir in period_dirs:
        period_name = period_dir.name
        
        # Find backtest_results CSV files
        csv_files = list(period_dir.glob("backtest_results_*.csv"))
        
        if not csv_files:
            print(f"  Warning: No CSV files in {period_name}")
            continue
        
        # Use the most recent CSV file if multiple exist
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        
        print(f"  Loading {period_name}: {csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file)
            df['period'] = period_name
            all_results.append(df)
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")
    
    if not all_results:
        print("No results loaded!")
        return pd.DataFrame()
    
    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    print(f"\nLoaded {len(combined)} total results across {len(period_dirs)} periods")
    
    return combined


def calculate_alpha_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate alpha and other derived metrics."""
    # Alpha is the outperformance column
    df['alpha'] = df['outperformance']
    
    # Calculate expectancy (average gain per trade)
    df['expectancy'] = (df['win_rate'] / 100.0 * df['average_gain']) - \
                       ((1 - df['win_rate'] / 100.0) * df['average_loss'].abs())
    
    return df


def analyze_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by configuration."""
    config_stats = df.groupby('strategy').agg({
        'alpha': ['mean', 'std', 'min', 'max', 'count'],
        'win_rate': 'mean',
        'total_trades': 'sum',
        'outperformance': 'mean',
    }).round(2)
    
    # Flatten column names
    config_stats.columns = ['_'.join(col).strip() for col in config_stats.columns.values]
    config_stats = config_stats.sort_values('alpha_mean', ascending=False)
    
    return config_stats.reset_index()


def analyze_by_period(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by period."""
    period_stats = df.groupby('period').agg({
        'alpha': ['mean', 'std', 'min', 'max'],
        'win_rate': 'mean',
        'total_trades': 'sum',
    }).round(2)
    
    period_stats.columns = ['_'.join(col).strip() for col in period_stats.columns.values]
    period_stats = period_stats.sort_values('alpha_mean', ascending=False)
    
    return period_stats.reset_index()


def find_best_configs(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Find best performing configurations."""
    config_avg = df.groupby('strategy').agg({
        'alpha': 'mean',
        'win_rate': 'mean',
        'total_trades': 'sum',
    }).round(2)
    
    config_avg = config_avg.sort_values('alpha', ascending=False).head(top_n)
    return config_avg.reset_index()


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table without tabulate dependency."""
    if df.empty:
        return ""
    
    lines = []
    
    # Header
    headers = list(df.columns)
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    
    # Rows
    for _, row in df.iterrows():
        values = [str(v) if pd.notna(v) else "" for v in row]
        lines.append("| " + " | ".join(values) + " |")
    
    return "\n".join(lines)


def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """Generate comprehensive summary report."""
    with open(output_path, 'w') as f:
        f.write("# Grid Test Results Analysis\n\n")
        f.write(f"**Total Configurations:** {df['strategy'].nunique()}\n")
        f.write(f"**Total Periods:** {df['period'].nunique()}\n")
        f.write(f"**Total Evaluations:** {len(df)}\n\n")
        
        # Best overall configs
        f.write("## Top 10 Configurations (by Average Alpha)\n\n")
        best_configs = find_best_configs(df, top_n=10)
        f.write(df_to_markdown_table(best_configs))
        f.write("\n\n")
        
        # Performance by period
        f.write("## Performance by Period\n\n")
        period_stats = analyze_by_period(df)
        f.write(df_to_markdown_table(period_stats))
        f.write("\n\n")
        
        # Best config per period
        f.write("## Best Configuration per Period\n\n")
        for period in sorted(df['period'].unique()):
            period_df = df[df['period'] == period]
            best = period_df.nlargest(1, 'alpha')[['strategy', 'alpha', 'win_rate', 'total_trades']]
            f.write(f"### {period}\n\n")
            f.write(df_to_markdown_table(best))
            f.write("\n\n")
        
        # Detailed config analysis
        f.write("## Detailed Configuration Statistics\n\n")
        config_stats = analyze_by_config(df)
        f.write(df_to_markdown_table(config_stats))
        f.write("\n\n")
    
    print(f"Summary report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze grid-test CSV results across all periods"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to results directory (e.g., results/hypothesis_tests_YYYYMMDD_HHMMSS/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for reports (defaults to results_dir)",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    # Load results
    print("Loading results from CSV files...")
    df = load_period_results(results_dir)
    
    if df.empty:
        print("No results to analyze")
        return 1
    
    # Calculate metrics
    df = calculate_alpha_metrics(df)
    
    # Generate reports
    print("\nGenerating analysis reports...")
    generate_summary_report(df, output_dir / "analysis_report.md")
    
    # Save CSV exports
    csv_path = output_dir / "all_results_combined.csv"
    df.to_csv(csv_path, index=False)
    print(f"Combined results CSV saved to: {csv_path}")
    
    # Pivot table: alpha by config and period
    pivot = df.pivot_table(
        values='alpha',
        index='strategy',
        columns='period',
        aggfunc='mean'
    ).round(2)
    pivot_path = output_dir / "alpha_pivot_by_config_period.csv"
    pivot.to_csv(pivot_path)
    print(f"Alpha pivot table saved to: {pivot_path}")
    
    # Find best overall
    best_configs = find_best_configs(df, top_n=5)
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS (by Average Alpha)")
    print("="*80)
    for _, row in best_configs.iterrows():
        print(f"{row['strategy']:30s} | Alpha: {row['alpha']:7.2f}% | Win Rate: {row['win_rate']:5.2f}% | Trades: {int(row['total_trades'])}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
