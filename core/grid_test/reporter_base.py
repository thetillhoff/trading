"""
Base reporter: ComparisonReporter with init, paths, print, save, and analysis report.

Chart methods come from ReporterChartsMixin (reporter_charts.py).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import List, Optional

from ..evaluation.walk_forward import WalkForwardResult
from ..signals.config import StrategyConfig

from .reporter_utils import trades_to_dataframe
from .reporter_charts import ReporterChartsMixin


class ComparisonReporter(ReporterChartsMixin):
    """Generates comparison reports and visualizations for backtesting results."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the reporter.
        
        Args:
            output_dir: Directory for output files (default: current directory)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_result_path(
        self,
        config: StrategyConfig,
        filename: str
    ) -> Path:
        """
        Generate hierarchical result path matching configs/ directory structure.
        
        Format: results/{relative_path_from_configs}/{filename}
        Example: configs/optimization/ew_all_indicators_wave_001.yaml 
                 → results/optimization/ew_all_indicators_wave_001/backtest_results.csv
        
        Args:
            config: Strategy configuration
            filename: Name of the file to save
            
        Returns:
            Full path for the result file
        """
        # Use stored source path if available (from config file location)
        if hasattr(config, '_source_path') and config._source_path:
            result_dir = self.output_dir / config._source_path
        else:
            # Fallback to old structure: config_name/instrument/date_range
            instrument_label = "unknown"
            if hasattr(config, 'instruments') and config.instruments:
                if len(config.instruments) == 1:
                    instrument_label = config.instruments[0]
                else:
                    instrument_label = "unified"
            
            date_range = "full"
            if hasattr(config, 'start_date') and hasattr(config, 'end_date') and config.start_date and config.end_date:
                start = config.start_date[:4]
                end = config.end_date[:4]
                date_range = f"{start}-{end}"
            
            result_dir = self.output_dir / config.name / instrument_label / date_range
        
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir / filename
    
    def _format_config_metadata(self, config: StrategyConfig) -> List[str]:
        """
        Format strategy configuration as CSV comment lines.
        
        Args:
            config: Strategy configuration to format
            
        Returns:
            List of comment lines with configuration metadata
        """
        lines = ["# Strategy Configuration"]
        lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("#")
        
        # Convert dataclass to dict and format each field
        config_dict = asdict(config)
        for key, value in sorted(config_dict.items()):
            # Format value for readability
            if isinstance(value, dict):
                value_str = str(value)
            elif isinstance(value, (list, tuple)):
                value_str = str(value)
            else:
                value_str = str(value)
            lines.append(f"# {key}: {value_str}")
        
        lines.append("#")
        return lines
    
    def print_summary(self, results: List[WalkForwardResult], top_n: int = 10) -> None:
        """Print a summary comparison of all results."""
        if not results:
            print("No results to report.")
            return
        
        # Print header
        print("\n" + "=" * 80)
        print("DOES ACTIVE TRADING BEAT BUY-AND-HOLD?")
        print("=" * 80)
        
        first_result = results[0]
        print(f"\nEvaluation Period: {first_result.evaluation_start_date.date()} to {first_result.evaluation_end_date.date()}")
        print(f"Buy-and-Hold Return: {first_result.buy_and_hold_gain:.2f}%")
        print(f"Total strategies tested: {len(results)}")
        
        # Sort by alpha (hybrid vs buy-and-hold - the key metric)
        sorted_results = sorted(results, key=lambda r: getattr(r, 'active_alpha', r.outperformance), reverse=True)
        
        # Show top N if there are many results
        show_results = sorted_results[:top_n] if len(results) > top_n else sorted_results
        
        # Print comparison table - focused on the key question
        print("\n" + "-" * 130)
        print("Hybrid = Active trading portion + Uninvested capital in buy-and-hold")
        print("Alpha = Hybrid Return - Pure Buy-and-Hold (positive = active trading adds value)")
        print("PF = Profit Factor (Total Wins / Total Losses) - above 1.0 is profitable")
        print("E[%] = Expectancy per trade (expected % return per trade)")
        print("-" * 130)
        if len(results) > top_n:
            print(f"TOP {top_n} STRATEGIES (out of {len(results)}) - sorted by Alpha")
            print("-" * 130)
        print(f"{'Strategy':<20} {'Trades':>6} {'Win%':>6} {'AvgWin':>7} {'AvgLoss':>8} {'PF':>5} {'E[%]':>6} {'Alpha':>8} {'Verdict':>12}")
        print("-" * 130)
        
        for result in show_results:
            s = result.summary
            sim = result.simulation
            name = result.config.name[:20]  # Truncate long names
            alpha = getattr(result, 'active_alpha', result.outperformance)
            verdict = "ACTIVE wins" if alpha > 0 else "HOLD wins"
            
            # Get risk/reward metrics
            avg_win = sim.avg_win_pct if hasattr(sim, 'avg_win_pct') else 0.0
            avg_loss = sim.avg_loss_pct if hasattr(sim, 'avg_loss_pct') else 0.0
            pf = sim.profit_factor if hasattr(sim, 'profit_factor') else 0.0
            expectancy = sim.expectancy_pct if hasattr(sim, 'expectancy_pct') else 0.0
            
            # Format profit factor (handle infinity)
            pf_str = "inf" if pf == float('inf') else f"{pf:.2f}"
            
            print(f"{name:<20} "
                  f"{s.total_trades:>6} "
                  f"{s.win_rate:>5.1f}% "
                  f"{avg_win:>+6.1f}% "
                  f"{avg_loss:>+7.1f}% "
                  f"{pf_str:>5} "
                  f"{expectancy:>+5.2f}% "
                  f"{alpha:>+7.1f}% "
                  f"{verdict:>12}")
        
        print("-" * 130)
        
        # Bottom line summary
        self._print_bottom_line(results, sorted_results)
    
    def _print_bottom_line(self, results: List[WalkForwardResult], sorted_results: List[WalkForwardResult]) -> None:
        """Print the bottom line summary answering: does active trading beat buy-and-hold?"""
        # Count strategies that beat buy-and-hold
        beat_hold = sum(1 for r in results if getattr(r, 'active_alpha', r.outperformance) > 0)
        total = len(results)
        
        # Find best and worst alpha
        best = sorted_results[0]
        worst = sorted_results[-1]
        best_alpha = getattr(best, 'active_alpha', best.outperformance)
        worst_alpha = getattr(worst, 'active_alpha', worst.outperformance)
        
        print("\n" + "=" * 80)
        print("BOTTOM LINE")
        print("=" * 80)
        print(f"\nStrategies where active trading beats buy-and-hold: {beat_hold}/{total}")
        
        if beat_hold > 0:
            print(f"\nBest performing strategy: {best.config.name}")
            print(f"  Alpha: {best_alpha:+.2f}%")
            print(f"  Hybrid Return: {getattr(best, 'hybrid_return', best.summary.total_gain):.2f}%")
            print(f"  Buy-and-Hold: {best.buy_and_hold_gain:.2f}%")
            print(f"  Win Rate: {best.summary.win_rate:.1f}%")
            print(f"  Trades: {best.summary.total_trades}")
        else:
            print(f"\nBest Alpha: {best_alpha:+.2f}% ({best.config.name})")
            print(f"Worst Alpha: {worst_alpha:+.2f}% ({worst.config.name})")
    
    def print_detailed_report(self, result: WalkForwardResult) -> None:
        """Print a detailed report for a single result."""
        print("\n" + "=" * 80)
        print(f"DETAILED REPORT: {result.config.name}")
        print("=" * 80)
        
        # Configuration
        print("\nConfiguration:")
        print(f"  Description: {result.config.description}")
        print(f"  Min Confidence: {result.config.min_confidence}")
        print(f"  Min Wave Size: {result.config.min_wave_size}")
        print(f"  Signal Types: {result.config.signal_types}")
        print(f"  Risk/Reward: {result.config.risk_reward}")
        print(f"  Max Days: {result.config.max_days or 'None'}")
        print(f"  Hold Through Stop-Loss: {result.config.hold_through_stop_loss}")
        
        # Summary
        s = result.summary
        print("\nPerformance Summary:")
        print(f"  Total Trades: {s.total_trades}")
        print(f"  Winning Trades: {s.winning_trades}")
        print(f"  Losing Trades: {s.losing_trades}")
        print(f"  No Outcome: {s.no_outcome_trades}")
        print(f"  Win Rate: {s.win_rate:.1f}%")
        print(f"  Average Gain (winners): {s.average_gain:.2f}%")
        print(f"  Average Loss (losers): {s.average_loss:.2f}%")
        print(f"  Total Gain: {s.total_gain:.2f}%")
        print(f"  Buy-and-Hold: {result.buy_and_hold_gain:.2f}%")
        print(f"  Outperformance: {result.outperformance:.2f}%")
        
        if s.average_days_held:
            print(f"  Average Days Held: {s.average_days_held:.1f}")
        
        # Best and worst trades
        if s.best_trade:
            print(f"\nBest Trade:")
            print(f"  Date: {s.best_trade.signal.timestamp.date()}")
            print(f"  Type: {s.best_trade.signal.signal_type.value}")
            print(f"  Gain: {s.best_trade.gain_percentage:.2f}%")
        
        if s.worst_trade:
            print(f"\nWorst Trade:")
            print(f"  Date: {s.worst_trade.signal.timestamp.date()}")
            print(f"  Type: {s.worst_trade.signal.signal_type.value}")
            print(f"  Loss: {s.worst_trade.gain_percentage:.2f}%")
    
    def print_parameter_sensitivity(self, results: List[WalkForwardResult]) -> None:
        """
        Print parameter sensitivity analysis to console.
        
        Shows how each parameter value affects Alpha and Expectancy,
        averaged across all other parameter combinations.
        
        Args:
            results: List of walk-forward results from grid search
        """
        if not results or len(results) < 2:
            return
        
        # Extract all numeric parameters from configs
        data = []
        for r in results:
            config = r.config
            sim = r.simulation
            
            row = {
                'alpha': getattr(r, 'active_alpha', r.outperformance),
                'expectancy': getattr(sim, 'expectancy_pct', 0.0),
                'trades': r.summary.total_trades,
                'min_confidence': config.min_confidence if config.use_elliott_wave else None,
                'min_wave_size': config.min_wave_size if config.use_elliott_wave else None,
                'rsi_period': config.rsi_period if config.use_rsi else None,
                'rsi_oversold': config.rsi_oversold if config.use_rsi else None,
                'rsi_overbought': config.rsi_overbought if config.use_rsi else None,
                'ema_short_period': config.ema_short_period if config.use_ema else None,
                'ema_long_period': config.ema_long_period if config.use_ema else None,
                'macd_fast': config.macd_fast if config.use_macd else None,
                'macd_slow': config.macd_slow if config.use_macd else None,
                'macd_signal': config.macd_signal if config.use_macd else None,
                # Risk management parameters (always present)
                'risk_reward': config.risk_reward,
                'position_size_pct': config.position_size_pct,
                'max_positions': config.max_positions,
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Define parameter groups
        param_groups = [
            ('Elliott Wave', [
                ('min_confidence', 'Confidence'),
                ('min_wave_size', 'Wave Size'),
            ]),
            ('RSI', [
                ('rsi_period', 'Period'),
                ('rsi_oversold', 'Oversold'),
                ('rsi_overbought', 'Overbought'),
            ]),
            ('EMA', [
                ('ema_short_period', 'Short Period'),
                ('ema_long_period', 'Long Period'),
            ]),
            ('MACD', [
                ('macd_fast', 'Fast'),
                ('macd_slow', 'Slow'),
                ('macd_signal', 'Signal'),
            ]),
            ('Risk Management', [
                ('risk_reward', 'Risk/Reward Ratio'),
                ('position_size_pct', 'Position Size %'),
                ('max_positions', 'Max Positions'),
            ]),
        ]
        
        # Filter to parameters with multiple values
        valid_params = []
        for group_name, params in param_groups:
            group_params = []
            for col, label in params:
                values = df[col].dropna()
                if len(values) > 0 and values.nunique() > 1:
                    group_params.append((col, label))
            if group_params:
                valid_params.append((group_name, group_params))
        
        if not valid_params:
            return  # No parameter variation
        
        print("\n" + "=" * 80)
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 80)
        print("\nShows how each parameter value affects performance (averaged across all other settings)")
        print("Alpha = Hybrid Return - Buy-and-Hold (positive = beats market)")
        print("Expectancy = Expected % return per trade\n")
        
        for group_name, params in valid_params:
            print(f"\n{group_name}:")
            print("-" * 80)
            print(f"{'Parameter':<20} {'Value':<12} {'Alpha':>10} {'Expectancy':>12} {'Trades':>8} {'Count':>8}")
            print("-" * 80)
            
            for col, param_label in params:
                # Group by parameter value
                param_df = df[df[col].notna()].copy()
                grouped = param_df.groupby(col).agg({
                    'alpha': ['mean', 'min', 'max', 'count'],
                    'expectancy': ['mean', 'min', 'max'],
                    'trades': 'mean',
                }).reset_index()
                
                grouped.columns = [col, 'alpha_mean', 'alpha_min', 'alpha_max', 'count',
                                  'exp_mean', 'exp_min', 'exp_max', 'trades_mean']
                grouped = grouped.sort_values(col)
                
                for _, row in grouped.iterrows():
                    value = row[col]
                    value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                    alpha_mean = row['alpha_mean']
                    exp_mean = row['exp_mean']
                    trades = row['trades_mean']
                    count = int(row['count'])
                    
                    # Highlight best values
                    alpha_best = grouped['alpha_mean'].max()
                    exp_best = grouped['exp_mean'].max()
                    alpha_marker = "★" if alpha_mean == alpha_best else " "
                    exp_marker = "★" if exp_mean == exp_best else " "
                    
                    print(f"{param_label:<20} {value_str:<12} "
                          f"{alpha_mean:>+9.2f}%{alpha_marker} "
                          f"{exp_mean:>+11.2f}%{exp_marker} "
                          f"{trades:>7.0f} "
                          f"{count:>7}")
        
        print("\n" + "=" * 80)
        print("★ = Best value for that metric")
        print("=" * 80 + "\n")
    
    def generate_analysis_report(
        self,
        results: List[WalkForwardResult],
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate markdown analysis report with best performers, metrics, and recommendations.
        
        Args:
            results: List of walk-forward results
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to the generated report
        """
        if not results:
            return ""
        
        # Sort by alpha
        sorted_results = sorted(results, key=lambda r: getattr(r, 'active_alpha', r.outperformance), reverse=True)
        
        # Generate report
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"analysis_report_{timestamp}.md"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write("# Grid Search Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Configurations Tested:** {len(results)}\n\n")
            
            # Top performers
            f.write("## Top 10 Performers (by Alpha)\n\n")
            f.write("| Rank | Strategy | Alpha (%) | Win Rate (%) | Trades | Expectancy (%) |\n")
            f.write("|------|----------|-----------|--------------|--------|----------------|\n")
            for i, result in enumerate(sorted_results[:10], 1):
                alpha = getattr(result, 'active_alpha', result.outperformance)
                f.write(f"| {i} | {result.config.name} | {alpha:.2f} | {result.simulation.win_rate:.1f} | "
                       f"{result.simulation.total_trades} | {result.simulation.expectancy_pct:.2f} |\n")
            f.write("\n")
            
            # Summary statistics
            alphas = [getattr(r, 'active_alpha', r.outperformance) for r in results]
            win_rates = [r.simulation.win_rate for r in results]
            trades = [r.simulation.total_trades for r in results]
            
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Configs with Positive Alpha:** {sum(1 for a in alphas if a > 0)} ({sum(1 for a in alphas if a > 0)*100//len(alphas)}%)\n")
            f.write(f"- **Average Alpha:** {np.mean(alphas):.2f}%\n")
            f.write(f"- **Best Alpha:** {max(alphas):.2f}%\n")
            f.write(f"- **Worst Alpha:** {min(alphas):.2f}%\n")
            f.write(f"- **Average Win Rate:** {np.mean(win_rates):.1f}%\n")
            f.write(f"- **Total Trades:** {sum(trades)}\n\n")
            
            # Best by category (if we can infer categories from names)
            f.write("## Recommendations\n\n")
            best = sorted_results[0]
            best_alpha = getattr(best, 'active_alpha', best.outperformance)
            f.write(f"**Best Overall Strategy:** `{best.config.name}`\n")
            f.write(f"- Alpha: {best_alpha:.2f}%\n")
            f.write(f"- Win Rate: {best.simulation.win_rate:.1f}%\n")
            f.write(f"- Trades: {best.simulation.total_trades}\n")
            f.write(f"- Expectancy: {best.simulation.expectancy_pct:.2f}%\n\n")

        return str(output_path)
    
    def save_parameter_sensitivity_csv(
        self,
        results: List[WalkForwardResult],
        filename_prefix: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save parameter sensitivity analysis to CSV.
        
        Args:
            results: List of walk-forward results from grid search
            filename_prefix: Prefix for output filename
            
        Returns:
            Path to generated CSV, or None if not enough parameter variation
        """
        if not results or len(results) < 2:
            return None
        
        # Extract all numeric parameters from configs
        data = []
        for r in results:
            config = r.config
            sim = r.simulation
            
            row = {
                'strategy': r.config.name,
                'alpha': getattr(r, 'active_alpha', r.outperformance),
                'expectancy': getattr(sim, 'expectancy_pct', 0.0),
                'trades': r.summary.total_trades,
                'win_rate': r.summary.win_rate,
                'profit_factor': sim.profit_factor if sim.profit_factor != float('inf') else 999.0,
                # Elliott Wave parameters
                'ew_enabled': config.use_elliott_wave,
                'min_confidence': config.min_confidence if config.use_elliott_wave else None,
                'min_wave_size': config.min_wave_size if config.use_elliott_wave else None,
                # RSI parameters
                'rsi_enabled': config.use_rsi,
                'rsi_period': config.rsi_period if config.use_rsi else None,
                'rsi_oversold': config.rsi_oversold if config.use_rsi else None,
                'rsi_overbought': config.rsi_overbought if config.use_rsi else None,
                # EMA parameters
                'ema_enabled': config.use_ema,
                'ema_short_period': config.ema_short_period if config.use_ema else None,
                'ema_long_period': config.ema_long_period if config.use_ema else None,
                # MACD parameters
                'macd_enabled': config.use_macd,
                'macd_fast': config.macd_fast if config.use_macd else None,
                'macd_slow': config.macd_slow if config.use_macd else None,
                'macd_signal': config.macd_signal if config.use_macd else None,
                # Risk management parameters
                'risk_reward': config.risk_reward,
                'position_size_pct': config.position_size_pct,
                'max_positions': config.max_positions,
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Create aggregated view: parameter value -> average metrics
        param_rows = []
        
        param_groups = [
            ('Elliott Wave', [
                ('min_confidence', 'Confidence'),
                ('min_wave_size', 'Wave Size'),
            ]),
            ('RSI', [
                ('rsi_period', 'Period'),
                ('rsi_oversold', 'Oversold'),
                ('rsi_overbought', 'Overbought'),
            ]),
            ('EMA', [
                ('ema_short_period', 'Short Period'),
                ('ema_long_period', 'Long Period'),
            ]),
            ('MACD', [
                ('macd_fast', 'Fast'),
                ('macd_slow', 'Slow'),
                ('macd_signal', 'Signal'),
            ]),
            ('Risk Management', [
                ('risk_reward', 'Risk/Reward Ratio'),
                ('position_size_pct', 'Position Size %'),
                ('max_positions', 'Max Positions'),
            ]),
        ]
        
        for group_name, params in param_groups:
            for col, param_label in params:
                param_df = df[df[col].notna()].copy()
                if len(param_df) == 0 or param_df[col].nunique() <= 1:
                    continue
                
                grouped = param_df.groupby(col).agg({
                    'alpha': ['mean', 'min', 'max', 'std', 'count'],
                    'expectancy': ['mean', 'min', 'max', 'std'],
                    'trades': 'mean',
                }).reset_index()
                
                grouped.columns = [col, 'alpha_mean', 'alpha_min', 'alpha_max', 'alpha_std', 'count',
                                  'exp_mean', 'exp_min', 'exp_max', 'exp_std', 'trades_mean']
                grouped = grouped.sort_values(col)
                
                for _, row in grouped.iterrows():
                    param_rows.append({
                        'indicator_group': group_name,
                        'parameter': param_label,
                        'value': row[col],
                        'alpha_mean': row['alpha_mean'],
                        'alpha_min': row['alpha_min'],
                        'alpha_max': row['alpha_max'],
                        'alpha_std': row['alpha_std'],
                        'expectancy_mean': row['exp_mean'],
                        'expectancy_min': row['exp_min'],
                        'expectancy_max': row['exp_max'],
                        'expectancy_std': row['exp_std'],
                        'avg_trades': row['trades_mean'],
                        'strategy_count': int(row['count']),
                    })
        
        if not param_rows:
            return None  # No parameter variation
        
        param_df = pd.DataFrame(param_rows)
        
        # Save CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if filename_prefix:
            filename = f"{filename_prefix}_parameter_sensitivity_{timestamp}.csv"
        else:
            filename = f"parameter_sensitivity_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        param_df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def save_results_csv(
        self,
        results: List[WalkForwardResult],
        filename: Optional[str] = None,
    ) -> str:
        """
        Save results to a CSV file.
        
        Args:
            results: List of walk-forward results
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to the generated CSV
        """
        if not results:
            return ""
        
        rows = []
        for result in results:
            s = result.summary
            rows.append({
                'strategy': result.config.name,
                'description': result.config.description,
                'total_trades': s.total_trades,
                'winning_trades': s.winning_trades,
                'losing_trades': s.losing_trades,
                'no_outcome_trades': s.no_outcome_trades,
                'win_rate': s.win_rate,
                'average_gain': s.average_gain,
                'average_loss': s.average_loss,
                'total_gain': s.total_gain,
                'buy_and_hold_gain': result.buy_and_hold_gain,
                'outperformance': result.outperformance,
                'average_days_held': s.average_days_held,
                'min_confidence': result.config.min_confidence,
                'min_wave_size': result.config.min_wave_size,
                'risk_reward': result.config.risk_reward,
            })
        
        df = pd.DataFrame(rows)
        
        # Save CSV
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_results_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def save_trades_csv(
        self,
        result: WalkForwardResult,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save individual trades to a CSV file for later timeline visualization.

        Includes basic trade data plus indicator values that triggered each trade:
        - rsi_value: RSI value at entry
        - ema_short/ema_long: EMA values at entry
        - macd_value/macd_signal/macd_histogram: MACD values at entry
        - indicator_confirmations: Number of indicators that confirmed the signal

        Args:
            result: Walk-forward evaluation result
            filename: Output filename (default: auto-generated)

        Returns:
            Path to the generated CSV
        """
        df = trades_to_dataframe(result)

        # Save CSV (always write, even when empty, so the file exists and lists 0 trades)
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trades_{result.config.name}_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        # Write configuration metadata as comment lines, then the CSV data
        with open(output_path, 'w') as f:
            # Write configuration metadata
            config_lines = self._format_config_metadata(result.config)
            for line in config_lines:
                f.write(line + '\n')
            
            # Write CSV data
            df.to_csv(f, index=False)
        
        return str(output_path)
