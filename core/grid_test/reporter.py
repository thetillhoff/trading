"""
Comparison reporter for backtesting results.

Generates comparison reports and visualizations for walk-forward evaluation results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from core.evaluation.walk_forward import WalkForwardResult
from core.signals.config import StrategyConfig
from core.evaluation.portfolio import PositionStatus


class ComparisonReporter:
    """Generates comparison reports and visualizations for backtesting results."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the reporter.
        
        Args:
            output_dir: Directory for output files (default: current directory)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
            print(f"\n  --> RECOMMENDATION: {best.config.name} adds value over buy-and-hold")
        else:
            print(f"\nBest Alpha: {best_alpha:+.2f}% ({best.config.name})")
            print(f"Worst Alpha: {worst_alpha:+.2f}% ({worst.config.name})")
            print(f"\n  --> RECOMMENDATION: None of the tested strategies beat pure buy-and-hold.")
            print(f"      Consider: Just buy-and-hold the index.")
    
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
    
    def generate_comparison_chart(
        self,
        results: List[WalkForwardResult],
        filename: Optional[str] = None,
        top_n: int = 15,
    ) -> str:
        """
        Generate a comparison chart for multiple results.
        
        Args:
            results: List of walk-forward results to compare
            filename: Output filename (default: auto-generated)
            top_n: Maximum number of strategies to show (default: 15)
            
        Returns:
            Path to the generated chart
        """
        if not results:
            return ""
        
        # Sort and limit to top N
        sorted_results = sorted(results, key=lambda r: r.summary.total_gain, reverse=True)
        if len(sorted_results) > top_n:
            sorted_results = sorted_results[:top_n]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        title = 'Walk-Forward Evaluation Comparison'
        if len(results) > top_n:
            title += f' (Top {top_n} of {len(results)})'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        names = [r.config.name[:20] for r in sorted_results]  # Truncate long names
        
        # 1. Alpha: Does active trading beat buy-and-hold? (THE KEY METRIC)
        ax1 = axes[0, 0]
        alphas = [getattr(r, 'active_alpha', r.outperformance) for r in sorted_results]
        colors = ['green' if a > 0 else 'indianred' for a in alphas]
        bars = ax1.bar(names, alphas, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Buy-and-Hold baseline')
        ax1.set_ylabel('Alpha (%)')
        ax1.set_title('Alpha vs Buy-and-Hold (Positive = Active Trading Wins)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Win Rate Comparison
        ax2 = axes[0, 1]
        win_rates = [r.summary.win_rate for r in sorted_results]
        ax2.bar(names, win_rates, color='steelblue', alpha=0.7)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Win Rate by Strategy')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Trade Count Comparison
        ax3 = axes[1, 0]
        winning = [r.summary.winning_trades for r in sorted_results]
        losing = [r.summary.losing_trades for r in sorted_results]
        no_outcome = [r.summary.no_outcome_trades for r in sorted_results]
        
        x = np.arange(len(names))
        width = 0.25
        ax3.bar(x - width, winning, width, label='Winning', color='green', alpha=0.7)
        ax3.bar(x, losing, width, label='Losing', color='red', alpha=0.7)
        ax3.bar(x + width, no_outcome, width, label='No Outcome', color='gray', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45)
        ax3.set_ylabel('Number of Trades')
        ax3.set_title('Trade Outcomes by Strategy')
        ax3.legend()
        
        # 4. Hybrid Return vs Buy-and-Hold
        ax4 = axes[1, 1]
        hybrid_returns = [getattr(r, 'hybrid_return', r.summary.total_gain) for r in sorted_results]
        bh_return = sorted_results[0].buy_and_hold_gain
        
        x = np.arange(len(names))
        width = 0.35
        ax4.bar(x - width/2, hybrid_returns, width, label='Hybrid Return', color='steelblue', alpha=0.7)
        ax4.axhline(y=bh_return, color='black', linestyle='--', linewidth=2, 
                    label=f'Buy-and-Hold ({bh_return:.1f}%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=45)
        ax4.set_ylabel('Return (%)')
        ax4.set_title('Hybrid Return vs Buy-and-Hold')
        ax4.legend()
        
        # Add explanations below charts
        explanation = (
            "Chart Guide: • Alpha = strategy return minus buy-and-hold (positive = beats market) "
            "• Win Rate = % of trades that hit target (high rate doesn't guarantee profit if losses are larger!) "
            "• Hybrid Return = what you'd earn if idle cash was in the market"
        )
        fig.text(0.5, -0.02, explanation, ha='center', fontsize=8, wrap=True,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save chart
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_comparison_{timestamp}.png"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_trade_timeline(
        self,
        ax,
        price_data: pd.Series,
        positions: List,
        title: str = "Trade Timeline",
        show_annotations: bool = False,
        annotation_threshold_pct: float = 0.0,
        annotation_top_n: Optional[int] = None,
    ) -> None:
        """
        Internal method to plot trades on a timeline (shared by both direct and CSV methods).
        
        Args:
            ax: Matplotlib axes to plot on
            price_data: Historical price data
            positions: List of Position objects or dict-like objects with trade data
            title: Chart title
            show_annotations: Show P&L annotations (default: False)
            annotation_threshold_pct: Only show annotations for trades above this % threshold (default: 0.0 = show all)
            annotation_top_n: Only show annotations for top N winning and bottom N losing trades (default: None = show all)
        """
        # Plot price line
        ax.plot(price_data.index, price_data.values, linewidth=1, color='gray', alpha=0.7, label='Price')
        
        # Convert positions to a common format (handle both Position objects and dicts)
        trades = []
        for pos in positions:
            if hasattr(pos, 'entry_timestamp'):
                # Position object
                cost_basis = pos.cost_basis
                pnl = pos.pnl
                pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
                trades.append({
                    'entry_date': pos.entry_timestamp,
                    'entry_price': pos.entry_price,
                    'exit_date': pos.exit_timestamp,
                    'exit_price': pos.exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'cost_basis': cost_basis,
                    'status': pos.status.value if hasattr(pos.status, 'value') else str(pos.status),
                })
            else:
                # Dict-like (from CSV)
                cost_basis = pos.get('cost_basis', 1)
                pnl = pos.get('pnl', 0)
                pnl_pct = pos.get('pnl_pct', (pnl / cost_basis * 100) if cost_basis > 0 else 0.0)
                # Ensure pnl_pct is in the dict
                if 'pnl_pct' not in pos or pos.get('pnl_pct') is None:
                    pos = dict(pos)  # Make a copy if needed
                    pos['pnl_pct'] = pnl_pct
                trades.append(pos)
        
        # Categorize trades
        closed_target = [t for t in trades if t.get('status') == 'closed_target' and t.get('exit_date')]
        closed_stop = [t for t in trades if t.get('status') == 'closed_stop' and t.get('exit_date')]
        closed_timeout = [t for t in trades if t.get('status') == 'closed_timeout' and t.get('exit_date')]
        closed_end = [t for t in trades if t.get('status') == 'closed_end' and t.get('exit_date')]
        winning = [t for t in trades if t.get('pnl', 0) > 0 and t.get('exit_date')]
        losing = [t for t in trades if t.get('pnl', 0) <= 0 and t.get('exit_date') and t.get('pnl', 0) < 0]
        
        # Determine which trades to annotate (filter by significance)
        # Use a tuple of (entry_date, entry_price, exit_date) as unique identifier
        trades_to_annotate = set()
        if show_annotations:
            if annotation_top_n is not None and annotation_top_n > 0:
                # Show only top N winners and bottom N losers
                winning_sorted = sorted(winning, key=lambda t: t.get('pnl_pct', 0), reverse=True)
                losing_sorted = sorted(losing, key=lambda t: t.get('pnl_pct', 0))
                
                top_winners = winning_sorted[:annotation_top_n]
                bottom_losers = losing_sorted[:annotation_top_n]
                
                for t in top_winners + bottom_losers:
                    key = (t.get('entry_date'), t.get('entry_price'), t.get('exit_date'))
                    if all(k is not None for k in key):
                        trades_to_annotate.add(key)
            elif annotation_threshold_pct > 0:
                # Show only trades above threshold
                for t in winning + losing:
                    pnl_pct = abs(t.get('pnl_pct', 0))
                    if pnl_pct >= annotation_threshold_pct:
                        key = (t.get('entry_date'), t.get('entry_price'), t.get('exit_date'))
                        if all(k is not None for k in key):
                            trades_to_annotate.add(key)
            else:
                # Show all trades
                for t in winning + losing:
                    key = (t.get('entry_date'), t.get('entry_price'), t.get('exit_date'))
                    if all(k is not None for k in key):
                        trades_to_annotate.add(key)
        
        # Plot trades by exit type
        trade_groups = [
            (closed_target, 'lightgreen', 'o', 'Target Hit'),
            (closed_stop, 'lightcoral', 'x', 'Stop Loss'),
            (closed_timeout, 'orange', 's', 'Timeout'),
            (closed_end, 'gray', 'D', 'Period End'),
        ]

        for trades_list, exit_color, exit_marker, exit_type in trade_groups:
            for trade in trades_list:
                entry_date = trade.get('entry_date')
                entry_price = trade.get('entry_price')
                exit_date = trade.get('exit_date')
                exit_price = trade.get('exit_price')
                pnl_pct = trade.get('pnl_pct', 0)

                # Determine entry color based on P&L
                entry_color = 'green' if pnl_pct > 0 else 'red'

                # Plot entry point
                if entry_date and entry_price:
                    ax.scatter(entry_date, entry_price,
                              color=entry_color, marker='^', s=30, zorder=5,
                              edgecolors=f'dark{entry_color}', linewidths=0.5, alpha=0.7)

                # Plot exit point with type-specific styling
                if exit_date and exit_price:
                    ax.scatter(exit_date, exit_price,
                              color=exit_color, marker=exit_marker, s=25, zorder=5,
                              edgecolors=exit_color.replace('light', 'dark'), linewidths=0.5, alpha=0.7)

                    # Connection line - dotted for period-end closes
                    if entry_date and entry_price:
                        linestyle = ':' if exit_type == 'Period End' else '--'
                        alpha = 0.3 if exit_type == 'Period End' else 0.4
                        ax.plot([entry_date, exit_date],
                               [entry_price, exit_price],
                               color=entry_color, linestyle=linestyle, alpha=alpha, linewidth=0.8)

                    # Annotation with P&L (only for significant trades)
                    trade_key = (entry_date, entry_price, exit_date)
                    if show_annotations and trade_key in trades_to_annotate:
                        pnl_abs = trade.get('pnl', 0)
                        # Show both percentage and absolute if significant
                        if abs(pnl_pct) >= 5.0:  # For big trades, show both
                            annotation = f'{pnl_pct:+.1f}%\n${pnl_abs:+.2f}'
                        else:
                            annotation = f'{pnl_pct:+.1f}%'
                        ax.annotate(annotation,
                                   xy=(exit_date, exit_price),
                                   xytext=(2, 2), textcoords='offset points',
                                   fontsize=5, color=entry_color, fontweight='bold', alpha=0.8,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=exit_color, alpha=0.7, edgecolor=entry_color, linewidth=0.5))
        
        
        # Add legend (small font)
        legend_elements = [
            plt.Line2D([0], [0], color='gray', linewidth=1, label='Price'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
                      markersize=4, markeredgecolor='darkgreen', markeredgewidth=0.5, label='Entry (Profit)'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                      markersize=4, markeredgecolor='darkred', markeredgewidth=0.5, label='Entry (Loss)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                      markersize=3, markeredgecolor='green', markeredgewidth=0.5, label='Target Hit'),
            plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='lightcoral',
                      markersize=3, markeredgecolor='lightcoral', markeredgewidth=0.5, label='Stop Loss'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange',
                      markersize=3, markeredgecolor='orange', markeredgewidth=0.5, label='Timeout'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                      markersize=3, markeredgecolor='gray', markeredgewidth=0.5, label='Period End'),
            plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, alpha=0.4, label='Profit Trade'),
            plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, alpha=0.4, label='Loss Trade'),
            plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=1, alpha=0.3, label='Period End'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=6)
        
        # Styling with small fonts
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel('Price', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=5)
        ax.grid(True, alpha=0.2, linewidth=0.5)
    
    def generate_trade_timeline(
        self,
        result: WalkForwardResult,
        price_data: pd.Series,
        filename: Optional[str] = None,
        show_annotations: bool = True,
        annotation_threshold_pct: float = 0.0,
        annotation_top_n: Optional[int] = None,
        max_trades: Optional[int] = 100,
    ) -> str:
        """
        Generate a timeline chart showing trades with entry/exit points.

        Args:
            result: Walk-forward evaluation result
            price_data: Historical price data for the evaluation period
            filename: Output filename (default: auto-generated)
            show_annotations: Show P&L annotations on trades (default: True)
            annotation_threshold_pct: Only show annotations for trades above this % threshold
            annotation_top_n: Only show annotations for top N winning and bottom N losing trades
            max_trades: Maximum number of trades to display on timeline (default: 100)

        Returns:
            Path to the generated chart
        """
        if not result or not result.simulation.positions:
            return ""
        
        positions = result.simulation.positions

        # Limit number of trades displayed for readability
        if max_trades is not None and len(positions) > max_trades:
            # Sort by P&L to show best and worst trades, not just most recent
            positions_sorted = sorted(positions, key=lambda p: getattr(p, 'pnl', 0), reverse=True)
            # Take top half from best performers and bottom half from worst performers
            top_count = max_trades // 2
            bottom_count = max_trades - top_count
            positions = positions_sorted[:top_count] + positions_sorted[-bottom_count:]

        # Filter price data to evaluation period
        eval_data = price_data[
            (price_data.index >= result.evaluation_start_date) &
            (price_data.index <= result.evaluation_end_date)
        ]

        if len(eval_data) == 0:
            return ""

        # Create high-resolution figure (larger size for zooming)
        fig, ax = plt.subplots(figsize=(24, 14))

        # Categorize for summary
        all_closed = [p for p in positions if getattr(p, 'exit_timestamp', None)]
        winning = [p for p in all_closed if getattr(p, 'pnl', 0) > 0]
        losing = [p for p in all_closed if getattr(p, 'pnl', 0) <= 0]
        total_trades = len(all_closed)
        
        # Use shared plotting method
        original_total = len(result.simulation.positions)
        if max_trades and original_total > max_trades:
            title_suffix = f" (showing best/worst {total_trades} of {original_total} total trades)"
        else:
            title_suffix = ""
        title = f'{result.config.name} | {result.evaluation_start_date.date()} to {result.evaluation_end_date.date()} | {total_trades} trades{title_suffix}'
        self._plot_trade_timeline(
            ax, eval_data, positions,
            title=title,
            show_annotations=show_annotations,
            annotation_threshold_pct=annotation_threshold_pct,
            annotation_top_n=annotation_top_n,
        )
        
        # Add summary text box (small font)
        summary_text = (
            f"Total: {total_trades} | "
            f"Wins: {len(winning)} ({len(winning)/total_trades*100:.1f}%) | "
            f"Losses: {len(losing)} ({len(losing)/total_trades*100:.1f}%)\n"
            f"Return: {result.simulation.total_return_pct:.2f}% | "
            f"Alpha: {getattr(result, 'active_alpha', result.outperformance):.2f}%"
        )
        ax.text(0.02, 0.02, summary_text, transform=ax.transAxes,
               fontsize=7, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.tight_layout()
        
        # Save chart with high DPI for zooming
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trade_timeline_{timestamp}.png"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=600, bbox_inches='tight')  # High DPI for zooming
        plt.close()
        
        return str(output_path)
    
    def generate_trade_scatter_plots(
        self,
        result: WalkForwardResult,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate scatter plots showing trade characteristics by buy/sell.

        Creates multiple scatter plots colored by trade direction (buy=blue, sell=orange):
        - Entry Price vs Exit Price: Shows price movement patterns
        - P&L vs Duration: Shows timing efficiency
        - P&L vs Timeline: Shows performance over time
        - P&L % vs Timeline: Shows percentage returns over time

        Args:
            result: Walk-forward evaluation result
            filename: Output filename (default: auto-generated)

        Returns:
            Path to the generated chart
        """
        if not result or not result.simulation.positions:
            return ""
        
        positions = result.simulation.positions
        closed_positions = [p for p in positions if p.exit_timestamp and p.exit_timestamp is not None]
        
        if len(closed_positions) == 0:
            return ""
        
        # Separate trades by original signal type (buy/sell) instead of win/loss
        # Handle both Position objects and CSV dict data
        buy_trades = []
        sell_trades = []

        for p in closed_positions:
            if hasattr(p, 'original_signal_type'):
                # Position object
                signal_type = p.original_signal_type
            else:
                # CSV dict - check for original_signal_type field
                signal_type = p.get('original_signal_type', p.get('signal_type', 'buy'))

            if signal_type == 'buy':
                buy_trades.append(p)
            elif signal_type == 'sell':
                sell_trades.append(p)

        # Calculate metrics for each trade type
        def get_trade_data(pos_list):
            durations = []
            pnls = []
            pnl_pcts = []
            entry_prices = []
            exit_prices = []
            entry_dates = []

            for pos in pos_list:
                if pos.exit_timestamp and pos.entry_timestamp:
                    duration = (pos.exit_timestamp - pos.entry_timestamp).days
                    durations.append(duration)
                    pnls.append(pos.pnl)
                    pnl_pct = (pos.pnl / pos.cost_basis * 100) if pos.cost_basis > 0 else 0
                    pnl_pcts.append(pnl_pct)
                    entry_prices.append(pos.entry_price)
                    exit_prices.append(pos.exit_price)
                    entry_dates.append(pos.entry_timestamp)

            return durations, pnls, pnl_pcts, entry_prices, exit_prices, entry_dates

        buy_durations, buy_pnls, buy_pnl_pcts, buy_entry_prices, buy_exit_prices, buy_entry_dates = get_trade_data(buy_trades)
        sell_durations, sell_pnls, sell_pnl_pcts, sell_entry_prices, sell_exit_prices, sell_entry_dates = get_trade_data(sell_trades)
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Build title with signal type breakdown
        signal_breakdown = []
        if buy_trades:
            signal_breakdown.append(f'{len(buy_trades)} BUY')
        if sell_trades:
            signal_breakdown.append(f'{len(sell_trades)} SELL')
        breakdown_str = ' + '.join(signal_breakdown) if signal_breakdown else 'No signals'
        
        fig.suptitle(
            f'Trade Analysis: {result.config.name} | {len(closed_positions)} closed trades ({breakdown_str})',
            fontsize=14, fontweight='bold'
        )
        
        # Plot 1: Entry Price vs Exit Price
        ax1 = axes[0, 0]
        if buy_entry_prices:
            ax1.scatter(buy_entry_prices, buy_exit_prices, alpha=0.6, color='green', s=50, label=f'Buy ({len(buy_trades)})', edgecolors='darkgreen', linewidths=0.5)
        if sell_entry_prices:
            ax1.scatter(sell_entry_prices, sell_exit_prices, alpha=0.6, color='red', s=50, label=f'Sell ({len(sell_trades)})', edgecolors='darkred', linewidths=0.5)
        # Add diagonal line (y=x) for reference
        if buy_entry_prices or sell_entry_prices:
            all_entries = (buy_entry_prices or []) + (sell_entry_prices or [])
            min_price = min(all_entries)
            max_price = max(all_entries)
            ax1.plot([min_price, max_price], [min_price, max_price], 'k--', linewidth=1, alpha=0.3, label='Break-even')
        ax1.set_xlabel('Entry Price ($)', fontsize=10)
        ax1.set_ylabel('Exit Price ($)', fontsize=10)
        ax1.set_title('Entry vs Exit Price\n\nAbove diagonal = profitable\nBelow diagonal = losses', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Add note if only one signal type present
        if not buy_trades and sell_trades:
            ax1.text(0.5, 0.95, 'Note: Only SELL signals detected',
                    transform=ax1.transAxes, fontsize=9, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif buy_trades and not sell_trades:
            ax1.text(0.5, 0.95, 'Note: Only BUY signals detected (no short positions)',
                    transform=ax1.transAxes, fontsize=9, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Plot 2: P&L vs Duration
        ax2 = axes[0, 1]
        if buy_durations:
            ax2.scatter(buy_durations, buy_pnls, alpha=0.6, color='green', s=50, label=f'Buy ({len(buy_trades)})', edgecolors='darkgreen', linewidths=0.5)
        if sell_durations:
            ax2.scatter(sell_durations, sell_pnls, alpha=0.6, color='red', s=50, label=f'Sell ({len(sell_trades)})', edgecolors='darkred', linewidths=0.5)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax2.set_xlabel('Duration (days)', fontsize=10)
        ax2.set_ylabel('P&L ($)', fontsize=10)
        ax2.set_title('P&L vs Duration\n\nQuick profits = efficient timing\nLong holds = momentum trades', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: P&L vs Timeline
        ax3 = axes[1, 0]
        if buy_entry_dates:
            ax3.scatter(buy_entry_dates, buy_pnls, alpha=0.6, color='green', s=50, label=f'Buy ({len(buy_trades)})', edgecolors='darkgreen', linewidths=0.5)
        if sell_entry_dates:
            ax3.scatter(sell_entry_dates, sell_pnls, alpha=0.6, color='red', s=50, label=f'Sell ({len(sell_trades)})', edgecolors='darkred', linewidths=0.5)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax3.set_xlabel('Entry Date', fontsize=10)
        ax3.set_ylabel('P&L ($)', fontsize=10)
        ax3.set_title('P&L vs Timeline\n\nPerformance over time\nClusters show market conditions', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        # Format x-axis as dates
        import matplotlib.dates as mdates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax3.xaxis.set_major_locator(mdates.YearLocator())
        
        # Plot 4: P&L % vs Timeline
        ax4 = axes[1, 1]
        if buy_entry_dates:
            ax4.scatter(buy_entry_dates, buy_pnl_pcts, alpha=0.6, color='green', s=50, label=f'Buy ({len(buy_trades)})', edgecolors='darkgreen', linewidths=0.5)
        if sell_entry_dates:
            ax4.scatter(sell_entry_dates, sell_pnl_pcts, alpha=0.6, color='red', s=50, label=f'Sell ({len(sell_trades)})', edgecolors='darkred', linewidths=0.5)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax4.set_xlabel('Entry Date', fontsize=10)
        ax4.set_ylabel('P&L (%)', fontsize=10)
        ax4.set_title('P&L % vs Timeline\n\nPercentage returns over time\nShows risk-adjusted performance', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        # Format x-axis as dates
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax4.xaxis.set_major_locator(mdates.YearLocator())
        
        plt.tight_layout()
        
        # Save chart
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trade_scatter_{timestamp}.png"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_alpha_over_time(
        self,
        result: WalkForwardResult,
        price_data: pd.Series,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a chart showing alpha over time.

        Shows three return baselines:
        - Cash Only: 2% p.a. with monthly compounding
        - Buy-and-Hold: 100% invested in market
        - Strategy: Active trading with cash earning 2% p.a.

        Alpha measures strategy performance vs buy-and-hold.

        Args:
            result: Walk-forward evaluation result
            price_data: Historical price data for the evaluation period
            filename: Output filename (default: auto-generated)

        Returns:
            Path to the generated chart
        """
        if not result or not result.simulation.wallet_history:
            return ""
        
        # Filter price data to evaluation period
        eval_data = price_data[
            (price_data.index >= result.evaluation_start_date) &
            (price_data.index <= result.evaluation_end_date)
        ]
        
        if len(eval_data) == 0:
            return ""
        
        # Calculate baseline returns
        initial_price = eval_data.iloc[0]
        initial_capital = result.simulation.initial_capital

        # Buy-and-hold: invest all capital at start
        bh_shares = initial_capital / initial_price
        bh_values = eval_data * bh_shares
        bh_returns_pct = ((bh_values - initial_capital) / initial_capital) * 100

        # Cash only: all capital earns 2% p.a. with monthly compounding
        monthly_cash_rate = 0.02 / 12  # 2% annual / 12 months
        cash_only_values = []
        cash_balance = initial_capital
        prev_date = eval_data.index[0]

        for current_date in eval_data.index:
            months_elapsed = (current_date - prev_date).days / 30.44  # Average days per month
            if months_elapsed > 0:
                # Apply monthly compounding for the elapsed months
                for _ in range(int(months_elapsed)):
                    cash_balance *= (1 + monthly_cash_rate)
                # Apply partial month if needed
                partial_month = months_elapsed - int(months_elapsed)
                if partial_month > 0:
                    cash_balance *= (1 + monthly_cash_rate * partial_month)

            cash_only_values.append(cash_balance)
            prev_date = current_date

        cash_only_returns_pct = [((v - initial_capital) / initial_capital) * 100 for v in cash_only_values]

        # Strategy portfolio values over time
        wallet_history = result.simulation.wallet_history
        portfolio_values = [w.total_value for w in wallet_history]
        portfolio_dates = [w.timestamp for w in wallet_history]
        
        # Calculate strategy returns (portfolio + cash earning 2% p.a.)
        # The wallet_history already contains total_value (cash + invested_value)
        # We need to ADD interest earned on the cash portion on top of that
        
        monthly_cash_rate = 0.02 / 12  # 2% annual / 12 months
        
        # Track accumulated interest on cash over time
        accumulated_interest = 0.0
        strategy_values = []
        
        prev_date = portfolio_dates[0] if portfolio_dates else None
        
        for i, (date, wallet_state) in enumerate(zip(portfolio_dates, wallet_history)):
            # Base portfolio value (from simulation)
            base_value = wallet_state.total_value
            
            # Calculate interest on cash since last period
            if i > 0 and prev_date is not None:
                # Get cash balance from previous period
                prev_cash = wallet_history[i-1].cash
                
                # Calculate interest earned on that cash
                days_elapsed = (date - prev_date).days
                if days_elapsed > 0 and prev_cash > 0:
                    # Daily rate from 2% annual
                    daily_rate = 0.02 / 365.25
                    interest_earned = prev_cash * (daily_rate * days_elapsed)
                    accumulated_interest += interest_earned
            
            # Strategy value = portfolio value + accumulated interest
            strategy_value = base_value + accumulated_interest
            strategy_values.append(strategy_value)
            
            prev_date = date
        
        # Convert to percentage returns
        strategy_returns_pct = [((v - initial_capital) / initial_capital) * 100 for v in strategy_values]
        
        # Calculate alpha over time (strategy return - buy-and-hold return)
        alpha_over_time = []
        alpha_dates = []
        for date, strategy_return in zip(portfolio_dates, strategy_returns_pct):
            # Find corresponding buy-and-hold return at this date
            try:
                if isinstance(date, pd.Timestamp):
                    date_for_lookup = date
                else:
                    date_for_lookup = pd.Timestamp(date)

                bh_idx = eval_data.index.get_indexer([date_for_lookup], method='nearest')[0]
                if bh_idx >= 0 and bh_idx < len(bh_returns_pct):
                    bh_return_at_date = bh_returns_pct.iloc[bh_idx]
                    alpha = strategy_return - bh_return_at_date
                    alpha_over_time.append(alpha)
                    alpha_dates.append(date)
            except (IndexError, KeyError):
                # Skip if we can't find matching date
                continue
        
        if len(alpha_over_time) == 0:
            return ""
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle(
            f'Alpha Over Time: {result.config.name} | Final Alpha: {result.active_alpha:.2f}%',
            fontsize=14, fontweight='bold'
        )
        
        # Plot 1: Returns comparison
        ax1.plot(eval_data.index, cash_only_returns_pct, label='Cash Only (2% p.a.)', color='gray', linewidth=2, alpha=0.7)
        ax1.plot(eval_data.index, bh_returns_pct, label='Buy-and-Hold', color='blue', linewidth=2, alpha=0.7)
        ax1.plot(portfolio_dates, strategy_returns_pct, label='Strategy (Cash earns 2% p.a.)', color='green', linewidth=2, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Return (%)', fontsize=10)
        ax1.set_title('Cumulative Returns: Cash vs Buy-and-Hold vs Strategy', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Alpha over time
        ax2.plot(alpha_dates, alpha_over_time, label='Active Alpha', color='purple', linewidth=2, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Buy-and-Hold (α=0)')
        ax2.fill_between(alpha_dates, 0, alpha_over_time, where=(np.array(alpha_over_time) >= 0), 
                         alpha=0.3, color='green', label='Positive Alpha')
        ax2.fill_between(alpha_dates, 0, alpha_over_time, where=(np.array(alpha_over_time) < 0), 
                         alpha=0.3, color='red', label='Negative Alpha')
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Alpha (%)', fontsize=10)
        ax2.set_title('Active Alpha: Strategy vs Buy-and-Hold', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Add summary text
        final_alpha = alpha_over_time[-1] if alpha_over_time else 0
        max_alpha = max(alpha_over_time) if alpha_over_time else 0
        min_alpha = min(alpha_over_time) if alpha_over_time else 0
        summary_text = (
            f"Final Alpha: {final_alpha:.2f}% | "
            f"Max: {max_alpha:.2f}% | "
            f"Min: {min_alpha:.2f}%"
        )
        ax2.text(0.02, 0.02, summary_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.tight_layout()
        
        # Save chart
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"alpha_over_time_{timestamp}.png"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_equity_curve(
        self,
        results: List[WalkForwardResult],
        price_data: Optional[pd.Series] = None,
        filename: Optional[str] = None,
        top_n: int = 10,
    ) -> str:
        """
        Generate comparison chart showing hybrid returns vs buy-and-hold.
        
        Directly compares what you'd earn with each strategy (including idle cash
        earning market returns) vs pure buy-and-hold.
        
        Args:
            results: List of walk-forward results
            price_data: Price data for market calculation (required)
            filename: Output filename (default: auto-generated)
            top_n: Maximum number of strategies to show (default: 10)
            
        Returns:
            Path to the generated chart
        """
        if not results or price_data is None:
            return ""
        
        # Sort by alpha and limit to top N
        sorted_results = sorted(results, key=lambda r: getattr(r, 'active_alpha', r.outperformance), reverse=True)
        if len(sorted_results) > top_n:
            display_results = sorted_results[:top_n]
        else:
            display_results = sorted_results
        
        first_result = results[0]
        bh_return = first_result.buy_and_hold_gain
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        names = [r.config.name[:18] for r in display_results]
        hybrid_returns = [getattr(r, 'hybrid_return', r.summary.total_gain) for r in display_results]
        alphas = [getattr(r, 'active_alpha', r.outperformance) for r in display_results]
        expectancies = [r.simulation.expectancy_pct if hasattr(r.simulation, 'expectancy_pct') else 0 for r in display_results]
        profit_factors = [r.simulation.profit_factor if hasattr(r.simulation, 'profit_factor') else 0 for r in display_results]
        
        # LEFT CHART: Hybrid Return vs Buy-and-Hold
        x = np.arange(len(names))
        width = 0.35
        
        # Bars for hybrid return
        bars = ax1.bar(x, hybrid_returns, width, label='Hybrid Strategy', color='steelblue', alpha=0.8)
        
        # Buy-and-hold reference line
        ax1.axhline(y=bh_return, color='black', linestyle='--', linewidth=2.5, 
                    label=f'Buy-and-Hold ({bh_return:.1f}%)')
        
        # Color bars: green if beats market, red if not
        for bar, hybrid in zip(bars, hybrid_returns):
            bar.set_color('green' if hybrid > bh_return else 'indianred')
            bar.set_alpha(0.8)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_ylabel('Total Return (%)')
        ax1.set_title('Hybrid Strategy Return vs Buy-and-Hold\n(Green = Beats Market)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # RIGHT CHART: Alpha and Expectancy (the quality metrics)
        x = np.arange(len(names))
        width = 0.35
        
        # Alpha bars
        alpha_bars = ax2.bar(x - width/2, alphas, width, label='Alpha (%)', alpha=0.8)
        for bar, alpha in zip(alpha_bars, alphas):
            bar.set_color('green' if alpha > 0 else 'indianred')
        
        # Expectancy bars (scaled for visibility)
        exp_scale = 10  # Scale up expectancy for visibility
        exp_bars = ax2.bar(x + width/2, [e * exp_scale for e in expectancies], width, 
                          label=f'Expectancy (×{exp_scale})', color='teal', alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Alpha (%) / Scaled Expectancy')
        ax2.set_title('Strategy Quality Metrics')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add explanatory text box for right chart
        explanation = (
            "Alpha: How much strategy beats buy-and-hold\n"
            "  • Positive = strategy adds value\n"
            "  • Negative = just buy-and-hold instead\n\n"
            f"Expectancy (×{exp_scale}): Average return per trade\n"
            "  • Combines win rate AND win/loss sizes\n"
            "  • Positive = profitable trades on average"
        )
        ax2.text(0.02, 0.98, explanation, transform=ax2.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Add summary text
        beats_market = sum(1 for a in alphas if a > 0)
        fig.suptitle(f'Strategy Comparison: {beats_market}/{len(display_results)} strategies beat buy-and-hold', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_comparison_{timestamp}.png"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _build_time_based_returns(
        self,
        result: WalkForwardResult,
        date_index: pd.DatetimeIndex,
        start_price: float,
    ) -> Optional[pd.Series]:
        """
        Build a time series of cumulative returns from portfolio simulation.
        
        Uses the wallet history which tracks actual portfolio value over time.
        """
        # Check if we have the new simulation format
        if hasattr(result, 'simulation') and result.simulation.wallet_history:
            wallet_history = result.simulation.wallet_history
            
            # Create series from wallet history
            returns_by_date = {
                state.timestamp: state.return_pct
                for state in wallet_history
            }
            
            # Align with the date index
            returns_series = pd.Series(index=date_index, dtype=float)
            returns_series.iloc[0] = 0.0
            
            current_return = 0.0
            for date in date_index:
                if date in returns_by_date:
                    current_return = returns_by_date[date]
                returns_series[date] = current_return
            
            return returns_series
        
        # Fallback for old format (backward compatibility)
        if not hasattr(result, 'evaluations') or not result.evaluations:
            return None
        
        # Create a series of cumulative gains indexed by date
        cumulative_gain = 0.0
        gains_by_date = {}
        
        # Sort evaluations by exit timestamp
        sorted_evals = sorted(
            [e for e in result.evaluations if e.exit_timestamp is not None],
            key=lambda e: e.exit_timestamp
        )
        
        # Record cumulative gain at each exit point
        for eval in sorted_evals:
            cumulative_gain += eval.gain_percentage
            gains_by_date[eval.exit_timestamp] = cumulative_gain
        
        if not gains_by_date:
            return None
        
        # Create a full time series, forward-filling the cumulative gain
        returns_series = pd.Series(index=date_index, dtype=float)
        returns_series.iloc[0] = 0.0  # Start at 0
        
        current_cumulative = 0.0
        for date in date_index:
            if date in gains_by_date:
                current_cumulative = gains_by_date[date]
            returns_series[date] = current_cumulative
        
        return returns_series
    
    def generate_dimension_charts(
        self,
        results: List[WalkForwardResult],
        filename_prefix: Optional[str] = None,
    ) -> List[str]:
        """
        Generate charts showing performance by each grid dimension.
        
        Creates one chart per parameter dimension, showing how different
        values of that parameter affect performance (averaged across all
        other parameter combinations).
        
        Args:
            results: List of walk-forward results from grid search
            filename_prefix: Prefix for output filenames
            
        Returns:
            List of paths to generated charts
        """
        if not results or len(results) < 2:
            return []
        
        # Extract parameter values from config names or configs
        data = []
        for r in results:
            config = r.config
            sim = r.simulation
            data.append({
                'hybrid_return': getattr(r, 'hybrid_return', r.summary.total_gain),
                'alpha': getattr(r, 'active_alpha', r.outperformance),
                'profit_factor': getattr(sim, 'profit_factor', 0.0) if sim.profit_factor != float('inf') else 3.0,
                'expectancy': getattr(sim, 'expectancy_pct', 0.0),
                'trades': r.summary.total_trades,
                'elliott_wave': 'yes' if getattr(config, 'use_elliott_wave', True) else 'no',
                'rsi': 'yes' if getattr(config, 'use_rsi', False) else 'no',
                'ema': 'yes' if getattr(config, 'use_ema', False) else 'no',
                'macd': 'yes' if getattr(config, 'use_macd', False) else 'no',
                'signal_types': config.signal_types,
            })
        
        df = pd.DataFrame(data)
        
        # Define dimensions to analyze
        dimensions = [
            ('elliott_wave', 'Elliott Wave'),
            ('rsi', 'RSI'),
            ('ema', 'EMA'),
            ('macd', 'MACD'),
            ('signal_types', 'Signal Types'),
        ]
        
        # Filter to dimensions with multiple values
        dimensions = [
            (col, label) for col, label in dimensions
            if df[col].nunique() > 1
        ]
        
        if not dimensions:
            return []
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = filename_prefix or f"grid_dimension_{timestamp}"
        
        output_paths = []
        market_gain = results[0].buy_and_hold_gain
        
        # Create a single figure with subplots for all dimensions
        n_dims = len(dimensions)
        fig, axes = plt.subplots(2, n_dims, figsize=(5 * n_dims, 10))
        
        if n_dims == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Grid Search Analysis by Indicator\nBuy-and-Hold: {market_gain:.1f}% | Alpha = 0 means same as buy-and-hold', 
                     fontsize=14, fontweight='bold')
        
        for idx, (col, label) in enumerate(dimensions):
            # Group by this dimension and calculate stats including min/max for range
            grouped = df.groupby(col).agg({
                'alpha': ['mean', 'min', 'max', 'count'],
                'expectancy': ['mean', 'min', 'max'],
            }).reset_index()
            
            # Flatten column names
            grouped.columns = [col, 'alpha_mean', 'alpha_min', 'alpha_max', 'count',
                              'exp_mean', 'exp_min', 'exp_max']
            
            # Sort by the parameter value
            if col != 'signal_types':
                grouped = grouped.sort_values(col)
            
            x = list(range(len(grouped)))
            x_labels = [str(v) for v in grouped[col]]
            
            # Top row: Alpha with min/max range
            ax1 = axes[0, idx]
            bars = ax1.bar(x, grouped['alpha_mean'], color='steelblue', alpha=0.7)
            
            # Add error bars showing min/max range
            yerr_lower = grouped['alpha_mean'] - grouped['alpha_min']
            yerr_upper = grouped['alpha_max'] - grouped['alpha_mean']
            ax1.errorbar(x, grouped['alpha_mean'], 
                        yerr=[yerr_lower, yerr_upper],
                        fmt='none', color='black', capsize=5, capthick=1.5, linewidth=1.5)
            
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, 
                       label='Buy-and-Hold (α=0)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(x_labels, rotation=45 if col == 'signal_types' else 0)
            ax1.set_xlabel(label)
            ax1.set_ylabel('Alpha (%)')
            ax1.set_title(f'Alpha by {label} (avg ± range)')
            ax1.legend(loc='best', fontsize=8)
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, grouped['count'])):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'n={int(count)}', ha='center', va='bottom', fontsize=8)
            
            # Color bars based on whether alpha is positive
            for bar, alpha in zip(bars, grouped['alpha_mean']):
                bar.set_color('green' if alpha > 0 else 'indianred')
                bar.set_alpha(0.7)
            
            # Bottom row: Expectancy with min/max range
            ax2 = axes[1, idx]
            bars2 = ax2.bar(x, grouped['exp_mean'], color='steelblue', alpha=0.7)
            
            # Add error bars showing min/max range
            yerr_lower = grouped['exp_mean'] - grouped['exp_min']
            yerr_upper = grouped['exp_max'] - grouped['exp_mean']
            ax2.errorbar(x, grouped['exp_mean'], 
                        yerr=[yerr_lower, yerr_upper],
                        fmt='none', color='black', capsize=5, capthick=1.5, linewidth=1.5)
            
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Break-even')
            ax2.set_xticks(x)
            ax2.set_xticklabels(x_labels, rotation=45 if col == 'signal_types' else 0)
            ax2.set_xlabel(label)
            ax2.set_ylabel('Expectancy (% per trade)')
            ax2.set_title(f'Expectancy by {label} (avg ± range)')
            ax2.legend(loc='best', fontsize=8)
            
            # Color bars based on expectancy
            for bar, exp in zip(bars2, grouped['exp_mean']):
                bar.set_color('green' if exp > 0 else 'indianred')
                bar.set_alpha(0.7)
        
        # Add explanatory text at the bottom of the figure
        explanation = (
            "How to read this chart:\n"
            "• Bars show AVERAGE across all strategy combinations with that setting. Error bars show min→max RANGE.\n"
            "• Alpha (top): How much the setting beats buy-and-hold. Green = better than market, Red = worse. n=number of strategies averaged.\n"
            "• Expectancy (bottom): Average % return per trade. Higher = more profitable trades on average."
        )
        fig.text(0.5, -0.02, explanation, ha='center', fontsize=9, 
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Make room for explanation
        
        output_path = self.output_dir / f"{prefix}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        output_paths.append(str(output_path))
        
        # Generate parameter sensitivity charts and reports
        param_chart = self.generate_parameter_sensitivity_chart(results, filename_prefix)
        if param_chart:
            output_paths.append(param_chart)
            # Also print and save parameter sensitivity analysis
            self.print_parameter_sensitivity(results)
            param_csv = self.save_parameter_sensitivity_csv(results, filename_prefix)
            if param_csv:
                output_paths.append(param_csv)
        
        return output_paths
    
    def generate_parameter_sensitivity_chart(
        self,
        results: List[WalkForwardResult],
        filename_prefix: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate chart showing how each numeric parameter value affects performance.
        
        This shows the impact of RSI period, EMA periods, MACD settings, etc.
        Each parameter is analyzed independently, averaging results across all
        other parameter combinations.
        
        Args:
            results: List of walk-forward results from grid search
            filename_prefix: Prefix for output filenames
            
        Returns:
            Path to generated chart, or None if not enough parameter variation
        """
        if not results or len(results) < 2:
            return None
        
        # Extract all numeric parameters from configs
        data = []
        for r in results:
            config = r.config
            sim = r.simulation
            
            row = {
                'alpha': getattr(r, 'active_alpha', r.outperformance),
                'expectancy': getattr(sim, 'expectancy_pct', 0.0),
                'trades': r.summary.total_trades,
                # Elliott Wave parameters (only if EW is enabled)
                'min_confidence': config.min_confidence if config.use_elliott_wave else None,
                'min_wave_size': config.min_wave_size if config.use_elliott_wave else None,
                # RSI parameters (only if RSI is enabled)
                'rsi_period': config.rsi_period if config.use_rsi else None,
                'rsi_oversold': config.rsi_oversold if config.use_rsi else None,
                'rsi_overbought': config.rsi_overbought if config.use_rsi else None,
                # EMA parameters (only if EMA is enabled)
                'ema_short_period': config.ema_short_period if config.use_ema else None,
                'ema_long_period': config.ema_long_period if config.use_ema else None,
                # MACD parameters (only if MACD is enabled)
                'macd_fast': config.macd_fast if config.use_macd else None,
                'macd_slow': config.macd_slow if config.use_macd else None,
                'macd_signal': config.macd_signal if config.use_macd else None,
                # Risk management parameters (always present)
                'risk_reward': config.risk_reward,
                'position_size_pct': config.position_size_pct,
                'max_positions': config.max_positions,
                'confidence_size_multiplier': config.confidence_size_multiplier,
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Define parameter groups for analysis
        param_groups = [
            # Elliott Wave parameters
            ('Elliott Wave', [
                ('min_confidence', 'Confidence'),
                ('min_wave_size', 'Wave Size'),
            ]),
            # RSI parameters
            ('RSI', [
                ('rsi_period', 'Period'),
                ('rsi_oversold', 'Oversold'),
                ('rsi_overbought', 'Overbought'),
            ]),
            # EMA parameters
            ('EMA', [
                ('ema_short_period', 'Short'),
                ('ema_long_period', 'Long'),
            ]),
            # MACD parameters
            ('MACD', [
                ('macd_fast', 'Fast'),
                ('macd_slow', 'Slow'),
                ('macd_signal', 'Signal'),
            ]),
            # Risk management parameters
            ('Risk Management', [
                ('risk_reward', 'Risk/Reward Ratio'),
                ('position_size_pct', 'Position Size %'),
                ('max_positions', 'Max Positions'),
                ('confidence_size_multiplier', 'Confidence Multiplier'),
            ]),
        ]
        
        # Filter to parameters that have multiple values
        valid_params = []
        for group_name, params in param_groups:
            group_params = []
            for col, label in params:
                # Get non-null values
                values = df[col].dropna()
                if len(values) > 0 and values.nunique() > 1:
                    group_params.append((col, label))
            if group_params:
                valid_params.append((group_name, group_params))
        
        if not valid_params:
            return None  # No parameter variation to analyze
        
        # Calculate total subplots needed
        total_params = sum(len(params) for _, params in valid_params)
        
        # Create figure: 2 rows (Alpha and Expectancy) × N parameter columns
        # Group by indicator for better organization
        n_cols = min(total_params, 6)  # Max 6 columns per row
        n_rows = 2 * ((total_params + n_cols - 1) // n_cols)  # 2 metrics per parameter row
        
        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = filename_prefix or f"grid_dimension_{timestamp}"
        market_gain = results[0].buy_and_hold_gain
        
        fig.suptitle(f'Parameter Sensitivity Analysis\nBuy-and-Hold: {market_gain:.1f}% | Shows how each parameter value affects results', 
                     fontsize=14, fontweight='bold')
        
        plot_idx = 0
        for group_name, params in valid_params:
            for col, param_label in params:
                # Filter to rows where this parameter is set (indicator enabled)
                param_df = df[df[col].notna()].copy()
                
                if len(param_df) == 0:
                    continue
                
                # Group by parameter value
                grouped = param_df.groupby(col).agg({
                    'alpha': ['mean', 'min', 'max', 'count'],
                    'expectancy': ['mean', 'min', 'max'],
                }).reset_index()
                
                grouped.columns = [col, 'alpha_mean', 'alpha_min', 'alpha_max', 'count',
                                  'exp_mean', 'exp_min', 'exp_max']
                grouped = grouped.sort_values(col)
                
                x = list(range(len(grouped)))
                x_labels = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in grouped[col]]
                
                # Alpha subplot
                ax1 = fig.add_subplot(n_rows, n_cols, plot_idx * 2 + 1)
                bars = ax1.bar(x, grouped['alpha_mean'], alpha=0.7)
                
                # Error bars
                yerr_lower = grouped['alpha_mean'] - grouped['alpha_min']
                yerr_upper = grouped['alpha_max'] - grouped['alpha_mean']
                ax1.errorbar(x, grouped['alpha_mean'], 
                            yerr=[yerr_lower, yerr_upper],
                            fmt='none', color='black', capsize=3, linewidth=1)
                
                ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
                ax1.set_xticks(x)
                ax1.set_xticklabels(x_labels, rotation=45 if len(x_labels[0]) > 4 else 0, fontsize=8)
                ax1.set_xlabel(f'{group_name}: {param_label}', fontsize=9)
                ax1.set_ylabel('Alpha (%)', fontsize=8)
                ax1.set_title(f'Alpha', fontsize=10)
                
                # Add count on bars
                for bar, count in zip(bars, grouped['count']):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                            f'n={int(count)}', ha='center', va='bottom', fontsize=7)
                
                # Color bars
                for bar, alpha in zip(bars, grouped['alpha_mean']):
                    bar.set_color('green' if alpha > 0 else 'indianred')
                    bar.set_alpha(0.7)
                
                # Expectancy subplot
                ax2 = fig.add_subplot(n_rows, n_cols, plot_idx * 2 + 2)
                bars2 = ax2.bar(x, grouped['exp_mean'], alpha=0.7)
                
                yerr_lower = grouped['exp_mean'] - grouped['exp_min']
                yerr_upper = grouped['exp_max'] - grouped['exp_mean']
                ax2.errorbar(x, grouped['exp_mean'], 
                            yerr=[yerr_lower, yerr_upper],
                            fmt='none', color='black', capsize=3, linewidth=1)
                
                ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
                ax2.set_xticks(x)
                ax2.set_xticklabels(x_labels, rotation=45 if len(x_labels[0]) > 4 else 0, fontsize=8)
                ax2.set_xlabel(f'{group_name}: {param_label}', fontsize=9)
                ax2.set_ylabel('Expectancy (%)', fontsize=8)
                ax2.set_title(f'Expectancy', fontsize=10)
                
                # Color bars
                for bar, exp in zip(bars2, grouped['exp_mean']):
                    bar.set_color('green' if exp > 0 else 'indianred')
                    bar.set_alpha(0.7)
                
                plot_idx += 1
        
        # Add explanation
        explanation = (
            "How to read: Each pair shows how one parameter value affects Alpha (vs buy-and-hold) and Expectancy (avg return per trade).\n"
            "Bars show averages across all strategies using that value. Error bars show min→max range. n=number of strategies.\n"
            "Parameters only shown when their indicator is enabled. Higher Alpha = better than market. Higher Expectancy = better trades."
        )
        fig.text(0.5, 0.01, explanation, ha='center', fontsize=9, 
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        output_path = self.output_dir / f"{prefix}_params.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(output_path)
    
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
                'confidence_size_multiplier': config.confidence_size_multiplier,
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
                ('confidence_size_multiplier', 'Confidence Multiplier'),
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
                'confidence_size_multiplier': config.confidence_size_multiplier,
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
                ('confidence_size_multiplier', 'Confidence Multiplier'),
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
        if not result or not result.simulation.positions:
            return ""
        
        rows = []
        for pos in result.simulation.positions:
            pnl_pct = (pos.pnl / pos.cost_basis * 100) if pos.cost_basis > 0 else 0.0
            days_held = (pos.exit_timestamp - pos.entry_timestamp).days if pos.exit_timestamp and pos.entry_timestamp else None
            
            rows.append({
                'entry_date': pos.entry_timestamp.strftime('%Y-%m-%d') if pos.entry_timestamp else '',
                'entry_price': pos.entry_price,
                'exit_date': pos.exit_timestamp.strftime('%Y-%m-%d') if pos.exit_timestamp else '',
                'exit_price': pos.exit_price if pos.exit_price else '',
                'signal_type': pos.signal_type,
                'shares': pos.shares,
                'cost_basis': pos.cost_basis,
                'pnl': pos.pnl,
                'pnl_pct': pnl_pct,
                'status': pos.status.value if hasattr(pos.status, 'value') else str(pos.status),
                'target_price': pos.target_price if pos.target_price else '',
                'stop_loss': pos.stop_loss if pos.stop_loss else '',
                'days_held': days_held if days_held else '',
                # Indicator values
                'rsi_value': pos.rsi_value if pos.rsi_value is not None else '',
                'ema_short': pos.ema_short if pos.ema_short is not None else '',
                'ema_long': pos.ema_long if pos.ema_long is not None else '',
                'macd_value': pos.macd_value if pos.macd_value is not None else '',
                'macd_signal': pos.macd_signal if pos.macd_signal is not None else '',
                'macd_histogram': pos.macd_histogram if pos.macd_histogram is not None else '',
                'indicator_confirmations': pos.indicator_confirmations,
                'original_signal_type': pos.original_signal_type,
                # New enhanced trade metadata
                'certainty': pos.certainty,
                'risk_amount': pos.risk_amount,
                'risk_reward_ratio': pos.risk_reward_ratio,
                'projection_price': pos.projection_price if pos.projection_price else '',
                # New feature tracking
                'position_size_method': pos.position_size_method,
                'trend_filter_active': pos.trend_filter_active,
                'trend_direction': pos.trend_direction,
            })
        
        df = pd.DataFrame(rows)
        
        # Save CSV
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
