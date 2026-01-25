#!/usr/bin/env python3
"""
Compare hypothesis test results across all periods and configs.

Parses CSV results from grid-search, identifies best configs, and generates
summary reports in Markdown and CSV formats.

Usage:
    docker compose run --rm cli python scripts/compare_hypothesis_results.py results/hypothesis_tests_YYYYMMDD_HHMMSS/

Output:
    - summary_report.md: Best performers by category and period
    - all_results.csv: Complete results table
    - pivot_alpha_by_config_period.csv: Comparison matrix
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class TestResult:
    """Container for a single test result."""
    category: str
    config_name: str
    period: str
    alpha: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    expectancy_pct: Optional[float] = None
    raw_output: str = ""
    parse_error: bool = False


def parse_result_file(result_path: Path) -> TestResult:
    """
    Parse a single result file and extract metrics.
    
    Expected filename format: category__config_name__period.txt
    """
    # Parse filename
    filename = result_path.stem  # Remove .txt
    parts = filename.split('__')
    
    if len(parts) != 3:
        print(f"Warning: Unexpected filename format: {result_path.name}")
        return TestResult(
            category="unknown",
            config_name="unknown",
            period="unknown",
            parse_error=True,
        )
    
    category, config_name, period = parts
    
    # Read file content
    try:
        with open(result_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Failed to read {result_path}: {e}")
        return TestResult(
            category=category,
            config_name=config_name,
            period=period,
            parse_error=True,
        )
    
    # Extract metrics using regex
    result = TestResult(
        category=category,
        config_name=config_name,
        period=period,
        raw_output=content,
    )
    
    # Look for alpha
    alpha_match = re.search(r'Alpha[:\s]+([+-]?\d+\.?\d*)%', content)
    if alpha_match:
        result.alpha = float(alpha_match.group(1))
    
    # Look for win rate
    win_rate_match = re.search(r'Win Rate[:\s]+(\d+\.?\d*)%', content)
    if win_rate_match:
        result.win_rate = float(win_rate_match.group(1))
    
    # Look for total trades
    trades_match = re.search(r'Total Trades[:\s]+(\d+)', content)
    if trades_match:
        result.total_trades = int(trades_match.group(1))
    
    # Look for expectancy
    expectancy_match = re.search(r'Expectancy[:\s]+([+-]?\d+\.?\d*)%', content)
    if expectancy_match:
        result.expectancy_pct = float(expectancy_match.group(1))
    
    # Mark as parse error if we didn't find key metrics
    if result.alpha is None and result.total_trades is None:
        result.parse_error = True
    
    return result


def load_results(results_dir: Path) -> List[TestResult]:
    """Load all test results from directory."""
    results = []
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return results
    
    # Find all result files
    result_files = list(results_dir.glob("*.txt"))
    
    if not result_files:
        print(f"Warning: No result files found in {results_dir}")
        return results
    
    print(f"Loading {len(result_files)} result files...")
    
    for result_file in result_files:
        result = parse_result_file(result_file)
        results.append(result)
    
    print(f"Loaded {len(results)} results")
    valid_count = sum(1 for r in results if not r.parse_error)
    print(f"  Valid: {valid_count}")
    print(f"  Errors: {len(results) - valid_count}")
    
    return results


def results_to_dataframe(results: List[TestResult]) -> pd.DataFrame:
    """Convert results to pandas DataFrame for analysis."""
    data = []
    for r in results:
        data.append({
            'category': r.category,
            'config': r.config_name,
            'period': r.period,
            'alpha': r.alpha,
            'win_rate': r.win_rate,
            'total_trades': r.total_trades,
            'expectancy_pct': r.expectancy_pct,
            'parse_error': r.parse_error,
        })
    
    return pd.DataFrame(data)


def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """Generate summary report comparing all configurations."""
    # Filter out parse errors
    valid_df = df[~df['parse_error']].copy()
    
    if valid_df.empty:
        print("No valid results to generate report")
        return
    
    with open(output_path, 'w') as f:
        f.write("# Hypothesis Test Results Summary\n\n")
        
        # Overall best performers
        f.write("## Overall Best Performers (by Alpha)\n\n")
        best_by_alpha = valid_df.nlargest(20, 'alpha')
        f.write(best_by_alpha[['config', 'period', 'alpha', 'win_rate', 'total_trades']].to_markdown(index=False))
        f.write("\n\n")
        
        # Best by category
        f.write("## Best by Category\n\n")
        for category in sorted(valid_df['category'].unique()):
            category_df = valid_df[valid_df['category'] == category]
            if category_df.empty:
                continue
            
            f.write(f"### {category}\n\n")
            best = category_df.nlargest(10, 'alpha')
            f.write(best[['config', 'period', 'alpha', 'win_rate', 'total_trades']].to_markdown(index=False))
            f.write("\n\n")
        
        # Average performance by config (across all periods)
        f.write("## Average Performance by Configuration\n\n")
        avg_by_config = valid_df.groupby('config').agg({
            'alpha': ['mean', 'std', 'min', 'max'],
            'win_rate': 'mean',
            'total_trades': 'sum',
        }).round(2)
        avg_by_config.columns = ['_'.join(col).strip() for col in avg_by_config.columns.values]
        avg_by_config = avg_by_config.sort_values('alpha_mean', ascending=False)
        f.write(avg_by_config.head(20).to_markdown())
        f.write("\n\n")
        
        # Performance by period
        f.write("## Performance by Period\n\n")
        for period in ['quick_test', 'recent_2yr', 'bear_market_long', 'bull_market_long', 'full_period_20yr']:
            period_df = valid_df[valid_df['period'] == period]
            if period_df.empty:
                continue
            
            f.write(f"### {period}\n\n")
            best = period_df.nlargest(10, 'alpha')
            f.write(best[['config', 'alpha', 'win_rate', 'total_trades']].to_markdown(index=False))
            f.write("\n\n")
    
    print(f"Summary report saved to: {output_path}")


def generate_csv_exports(df: pd.DataFrame, output_dir: Path):
    """Generate CSV files for further analysis."""
    # Full results
    csv_path = output_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Full results CSV saved to: {csv_path}")
    
    # Valid results only
    valid_df = df[~df['parse_error']]
    valid_csv_path = output_dir / "valid_results.csv"
    valid_df.to_csv(valid_csv_path, index=False)
    print(f"Valid results CSV saved to: {valid_csv_path}")
    
    # Pivot tables
    if not valid_df.empty:
        # Alpha by config and period
        pivot_alpha = valid_df.pivot_table(
            values='alpha',
            index='config',
            columns='period',
            aggfunc='mean'
        )
        pivot_alpha_path = output_dir / "pivot_alpha_by_config_period.csv"
        pivot_alpha.to_csv(pivot_alpha_path)
        print(f"Alpha pivot table saved to: {pivot_alpha_path}")


def find_best_overall(df: pd.DataFrame) -> Tuple[str, float]:
    """Find the best overall configuration by average alpha across all periods."""
    valid_df = df[~df['parse_error']]
    
    if valid_df.empty:
        return None, None
    
    # Group by config and calculate average alpha
    avg_alpha = valid_df.groupby('config')['alpha'].mean()
    best_config = avg_alpha.idxmax()
    best_alpha = avg_alpha.max()
    
    return best_config, best_alpha


def main():
    parser = argparse.ArgumentParser(
        description="Compare hypothesis test results and generate reports"
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
    
    # Load results
    results = load_results(results_dir)
    
    if not results:
        print("No results to analyze")
        return 1
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    # Generate reports
    print("\nGenerating reports...")
    generate_summary_report(df, output_dir / "summary_report.md")
    generate_csv_exports(df, output_dir)
    
    # Find best overall
    best_config, best_alpha = find_best_overall(df)
    if best_config:
        print("\n" + "="*80)
        print("BEST OVERALL CONFIGURATION")
        print("="*80)
        print(f"Configuration: {best_config}")
        print(f"Average Alpha: {best_alpha:.2f}%")
        print("\nThis configuration should be considered for updating the baseline.")
        print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
