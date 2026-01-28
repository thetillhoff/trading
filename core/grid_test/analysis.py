"""
Analyze grid-test CSV results (single-run or multi-period).

Loads backtest_results_*.csv from a directory (either directly for a single run,
or from period subdirs for hypothesis/multi-period), computes alpha/metrics,
and writes analysis_report.md, all_results_combined.csv, alpha_pivot_by_config_period.csv.
"""
from pathlib import Path
from typing import Optional

import pandas as pd


def load_results(results_dir: Path) -> pd.DataFrame:
    """
    Load all backtest_results_*.csv from results_dir.

    Single-run: results_dir has backtest_results_*.csv directly (e.g. one grid run)
    → one row per config, column 'period' = "single".

    Multi-period: results_dir has no top-level CSV, only subdirs with backtest_results_*.csv
    → one row per config per period, column 'period' = subdir name (e.g. hypothesis).
    """
    results_dir = Path(results_dir)
    all_results: list[pd.DataFrame] = []

    top_level_csvs = list(results_dir.glob("backtest_results_*.csv"))
    if top_level_csvs:
        # Prefer top-level: one combined CSV = single run
        csv_file = max(top_level_csvs, key=lambda p: p.stat().st_mtime)
        try:
            df = pd.read_csv(csv_file)
            df["period"] = "single"
            return df
        except Exception:
            pass

    # Multi-period: load from subdirs
    period_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    for period_dir in period_dirs:
        csv_files = list(period_dir.glob("backtest_results_*.csv"))
        if not csv_files:
            continue
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        try:
            df = pd.read_csv(csv_file)
            df["period"] = period_dir.name
            all_results.append(df)
        except Exception:
            continue

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def calculate_alpha_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add alpha (=outperformance) and expectancy."""
    df = df.copy()
    df["alpha"] = df["outperformance"]
    df["expectancy"] = (
        (df["win_rate"] / 100.0 * df["average_gain"])
        - ((1 - df["win_rate"] / 100.0) * df["average_loss"].abs())
    )
    return df


def analyze_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by strategy (config)."""
    config_stats = df.groupby("strategy").agg({
        "alpha": ["mean", "std", "min", "max", "count"],
        "win_rate": "mean",
        "total_trades": "sum",
        "outperformance": "mean",
    }).round(2)
    config_stats.columns = ["_".join(col).strip() for col in config_stats.columns.values]
    config_stats = config_stats.sort_values("alpha_mean", ascending=False)
    return config_stats.reset_index()


def analyze_by_period(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by period."""
    period_stats = df.groupby("period").agg({
        "alpha": ["mean", "std", "min", "max"],
        "win_rate": "mean",
        "total_trades": "sum",
    }).round(2)
    period_stats.columns = ["_".join(col).strip() for col in period_stats.columns.values]
    period_stats = period_stats.sort_values("alpha_mean", ascending=False)
    return period_stats.reset_index()


def find_best_configs(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top N configs by mean alpha."""
    config_avg = df.groupby("strategy").agg({
        "alpha": "mean",
        "win_rate": "mean",
        "total_trades": "sum",
    }).round(2)
    config_avg = config_avg.sort_values("alpha", ascending=False).head(top_n)
    return config_avg.reset_index()


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    lines = []
    headers = list(df.columns)
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in df.iterrows():
        values = [str(v) if pd.notna(v) else "" for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _generate_summary_report(df: pd.DataFrame, output_path: Path) -> None:
    with open(output_path, "w") as f:
        f.write("# Grid Test Results Analysis\n\n")
        f.write(f"**Total Configurations:** {df['strategy'].nunique()}\n")
        f.write(f"**Total Periods:** {df['period'].nunique()}\n")
        f.write(f"**Total Evaluations:** {len(df)}\n\n")

        f.write("## Top 10 Configurations (by Average Alpha)\n\n")
        best = find_best_configs(df, top_n=10)
        f.write(_df_to_markdown_table(best))
        f.write("\n\n")

        f.write("## Performance by Period\n\n")
        period_stats = analyze_by_period(df)
        f.write(_df_to_markdown_table(period_stats))
        f.write("\n\n")

        f.write("## Best Configuration per Period\n\n")
        for period in sorted(df["period"].unique()):
            period_df = df[df["period"] == period]
            best_per = period_df.nlargest(1, "alpha")[["strategy", "alpha", "win_rate", "total_trades"]]
            f.write(f"### {period}\n\n")
            f.write(_df_to_markdown_table(best_per))
            f.write("\n\n")

        f.write("## Detailed Configuration Statistics\n\n")
        config_stats = analyze_by_config(df)
        f.write(_df_to_markdown_table(config_stats))
        f.write("\n\n")


def analyze_results_dir(
    results_dir: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> bool:
    """
    Load CSVs from results_dir, compute metrics, write reports under output_dir.

    output_dir defaults to results_dir. Writes:
      - analysis_report.md
      - all_results_combined.csv
      - alpha_pivot_by_config_period.csv

    If verbose, prints top 5 configs to stdout.

    Returns True if analysis was run and reports written, False if no data.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir

    df = load_results(results_dir)
    if df.empty:
        return False

    df = calculate_alpha_metrics(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "analysis_report.md"
    _generate_summary_report(df, report_path)
    if verbose:
        print(f"Analysis report: {report_path}")

    combined_path = output_dir / "all_results_combined.csv"
    df.to_csv(combined_path, index=False)
    if verbose:
        print(f"Combined CSV: {combined_path}")

    pivot = df.pivot_table(values="alpha", index="strategy", columns="period", aggfunc="mean").round(2)
    pivot_path = output_dir / "alpha_pivot_by_config_period.csv"
    pivot.to_csv(pivot_path)
    if verbose:
        print(f"Alpha pivot: {pivot_path}")

    if verbose:
        best = find_best_configs(df, top_n=5)
        print("\n" + "=" * 80)
        print("TOP 5 CONFIGURATIONS (by Average Alpha)")
        print("=" * 80)
        for _, row in best.iterrows():
            print(f"{row['strategy']:30s} | Alpha: {row['alpha']:7.2f}% | Win Rate: {row['win_rate']:5.2f}% | Trades: {int(row['total_trades'])}")
        print("=" * 80)

    return True
