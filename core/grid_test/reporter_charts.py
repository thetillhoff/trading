"""
Chart generation methods for backtesting results.

Mixin class used by ComparisonReporter. All generate_* and _plot_* methods.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple, Callable
from pathlib import Path
from datetime import datetime

from ..evaluation.walk_forward import WalkForwardResult
from ..signals.config import StrategyConfig
from ..evaluation.portfolio import PositionStatus

from .reporter_utils import MAX_LEGEND_INSTRUMENTS, _daily_rate_from_pa, _is_new_month
from .reporter_analysis import compute_alpha_over_time_series


class ReporterChartsMixin:
    """Mixin providing all chart generation methods for ComparisonReporter."""

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
        price_data_by_instrument: Optional[Dict[str, pd.Series]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a chart showing alpha over time.

        Top: Cumulative returns (Cash only, buy-and-hold per instrument, Strategy).
        Middle: Market exposure (% of portfolio invested).
        Bottom: Active alpha (strategy vs first instrument buy-and-hold).

        Args:
            result: Walk-forward evaluation result
            price_data: Historical price data for the evaluation period (first instrument when multi)
            price_data_by_instrument: Optional dict instrument -> price series for B&H per instrument
            filename: Output filename (default: auto-generated)

        Returns:
            Path to the generated chart
        """
        series = compute_alpha_over_time_series(result, price_data, price_data_by_instrument)
        if series is None:
            return ""

        common_index = series.common_index
        cash_only_returns_pct = series.cash_only_returns_pct
        bh_series_by_inst = series.bh_series_by_inst
        strategy_returns_pct_aligned = series.strategy_returns_pct_aligned
        alpha_dates = series.alpha_dates
        alpha_over_time = series.alpha_over_time
        interest_rate_pa = series.interest_rate_pa

        wallet_history = result.simulation.wallet_history
        exposure_dates = [w.timestamp for w in wallet_history]
        exposure_pct = [
            (w.invested_value / w.total_value * 100) if w.total_value and w.total_value > 0 else 0.0
            for w in wallet_history
        ]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14))
        fig.suptitle(
            f'Alpha Over Time: {result.config.name} | Final Alpha: {result.active_alpha:.2f}%',
            fontsize=14, fontweight='bold'
        )

        # Plot 1: Cash only, buy-and-hold per instrument, strategy
        cash_label = f'Cash Only ({interest_rate_pa * 100:.1f}% p.a.)'
        strategy_label = f'Strategy (Cash earns {interest_rate_pa * 100:.1f}% p.a.)'
        ax1.plot(common_index, cash_only_returns_pct, label=cash_label, color='gray', linewidth=2, alpha=0.7)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(bh_series_by_inst), 1)))
        for idx, (inst, bh_series) in enumerate(bh_series_by_inst.items()):
            ax1.plot(common_index, bh_series.values, label=f'B&H {inst}', color=colors[idx % len(colors)], linewidth=2, alpha=0.7)
        ax1.plot(common_index, strategy_returns_pct_aligned, label=strategy_label, color='green', linewidth=2, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Return (%)', fontsize=10)
        ax1.set_title('Cumulative Returns: Cash vs Buy-and-Hold vs Strategy', fontsize=12, fontweight='bold')
        if len(bh_series_by_inst) > MAX_LEGEND_INSTRUMENTS:
            handles, labels = ax1.get_legend_handles_labels()
            # Order: cash, B&H per inst (same as bh_series_by_inst), strategy
            inst_list = list(bh_series_by_inst.keys())
            best_10 = sorted(
                inst_list,
                key=lambda inst: float(bh_series_by_inst[inst].iloc[-1]) if len(bh_series_by_inst[inst]) else float('-inf'),
                reverse=True,
            )[:MAX_LEGEND_INSTRUMENTS]
            inst_to_idx = {inst: 1 + i for i, inst in enumerate(inst_list)}
            sel_handles = [handles[0]] + [handles[inst_to_idx[inst]] for inst in best_10] + [handles[-1]]
            sel_labels = [labels[0]] + [labels[inst_to_idx[inst]] for inst in best_10] + [labels[-1]]
            ax1.legend(sel_handles, sel_labels, fontsize=9, loc='best')
        else:
            ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Market exposure (% of portfolio invested)
        ax2.plot(exposure_dates, exposure_pct, label='Market exposure', color='steelblue', linewidth=2, alpha=0.8)
        ax2.fill_between(exposure_dates, 0, exposure_pct, alpha=0.3, color='steelblue')
        ax2.set_ylim(0, max(100, max(exposure_pct) * 1.05) if exposure_pct else 100)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Exposure (%)', fontsize=10)
        ax2.set_title('Market Exposure: % of Portfolio Invested', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Alpha over time
        ax3.plot(alpha_dates, alpha_over_time, label='Active Alpha', color='purple', linewidth=2, alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Buy-and-Hold (α=0)')
        ax3.fill_between(alpha_dates, 0, alpha_over_time, where=(np.array(alpha_over_time) >= 0),
                         alpha=0.3, color='green', label='Positive Alpha')
        ax3.fill_between(alpha_dates, 0, alpha_over_time, where=(np.array(alpha_over_time) < 0),
                         alpha=0.3, color='red', label='Negative Alpha')
        ax3.set_xlabel('Date', fontsize=10)
        ax3.set_ylabel('Alpha (%)', fontsize=10)
        ax3.set_title('Active Alpha: Strategy vs Buy-and-Hold', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9, loc='best')
        ax3.grid(True, alpha=0.3)

        # Add summary text
        final_alpha = alpha_over_time[-1] if alpha_over_time else 0
        max_alpha = max(alpha_over_time) if alpha_over_time else 0
        min_alpha = min(alpha_over_time) if alpha_over_time else 0
        summary_text = (
            f"Final Alpha: {final_alpha:.2f}% | "
            f"Max: {max_alpha:.2f}% | "
            f"Min: {min_alpha:.2f}%"
        )
        ax3.text(0.02, 0.02, summary_text, transform=ax3.transAxes,
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

    def generate_market_exposure(
        self,
        result: WalkForwardResult,
        filename: Optional[str] = None,
    ) -> str:
        """
        Single chart: percentage of portfolio invested in the market over time.
        """
        if not result or not result.simulation.wallet_history:
            return ""
        wallet_history = result.simulation.wallet_history
        exposure_dates = [w.timestamp for w in wallet_history]
        exposure_pct = [
            (w.invested_value / w.total_value * 100) if w.total_value and w.total_value > 0 else 0.0
            for w in wallet_history
        ]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(exposure_dates, exposure_pct, label='Market exposure', color='steelblue', linewidth=2, alpha=0.8)
        ax.fill_between(exposure_dates, 0, exposure_pct, alpha=0.3, color='steelblue')
        ax.set_ylim(0, max(100, max(exposure_pct) * 1.05) if exposure_pct else 100)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Exposure (%)', fontsize=10)
        ax.set_title(f'Market Exposure: % of Portfolio Invested — {result.config.name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if filename is None:
            filename = f"market_exposure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def generate_value_gain_and_benchmarks(
        self,
        result: WalkForwardResult,
        price_data: Optional[pd.Series] = None,
        benchmark_series: Optional[Dict[str, pd.Series]] = None,
        price_data_by_instrument: Optional[Dict[str, pd.Series]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Top: cumulative value-gain % from conducted trades per instrument over time (one line per instrument).
        Bottom (if price_data_by_instrument): buy-and-hold return % per instrument (same as alpha_over_time), same colors.
        """
        if not result or not result.simulation.wallet_history:
            return ""
        closed = [p for p in result.simulation.positions if p.exit_timestamp and p.cost_basis > 0]
        wallet_history = result.simulation.wallet_history
        portfolio_dates = [w.timestamp for w in wallet_history]
        if not portfolio_dates:
            return ""
        by_inst: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
        for p in closed:
            inst = p.instrument if p.instrument else 'unknown'
            pnl_pct = (p.pnl / p.cost_basis * 100) if p.cost_basis > 0 else 0.0
            by_inst.setdefault(inst, []).append((p.exit_timestamp, pnl_pct))
        for inst in by_inst:
            by_inst[inst].sort(key=lambda x: x[0])
        instruments = sorted(by_inst.keys())
        if not instruments:
            return ""
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(instruments), 1)))
        has_market = bool(price_data_by_instrument)
        n_rows = 2 if has_market else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 6 * n_rows), sharex=True)
        axes = np.atleast_1d(axes)
        ax_top = axes[0]
        final_cum_by_inst: Dict[str, float] = {}
        for i, inst in enumerate(instruments):
            cum = 0.0
            cum_series = []
            idx = 0
            trades = by_inst[inst]
            for date in portfolio_dates:
                while idx < len(trades) and trades[idx][0] <= date:
                    cum += trades[idx][1]
                    idx += 1
                cum_series.append(cum)
            final_cum_by_inst[inst] = cum
            ax_top.plot(
                portfolio_dates, cum_series, label=inst, linewidth=2, alpha=0.8, color=colors[i % len(colors)]
            )
        ax_top.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax_top.set_ylabel('Cumulative value-gain % from trades')
        ax_top.set_title(f'Value-gain % from trades per instrument: {result.config.name}')
        if len(instruments) > MAX_LEGEND_INSTRUMENTS:
            handles, labels = ax_top.get_legend_handles_labels()
            best_10 = sorted(instruments, key=lambda inst: final_cum_by_inst.get(inst, float('-inf')), reverse=True)[:MAX_LEGEND_INSTRUMENTS]
            label_to_handle = dict(zip(labels, handles))
            sel_handles = [label_to_handle[inst] for inst in best_10]
            sel_labels = best_10
            ax_top.legend(sel_handles, sel_labels, fontsize=9, loc='best')
        else:
            ax_top.legend(loc='best', fontsize=9)
        ax_top.grid(True, alpha=0.3)

        if has_market:
            ax_bottom = axes[1]
            all_dates = pd.DatetimeIndex([])
            for inst, ser in price_data_by_instrument.items():
                if ser is not None and len(ser) > 0:
                    all_dates = all_dates.union(ser.index)
            if len(all_dates) == 0:
                ax_bottom.set_visible(False)
            else:
                all_dates = all_dates.sort_values()
                final_bh_by_inst: Dict[str, float] = {}
                for i, inst in enumerate(instruments):
                    ser = price_data_by_instrument.get(inst)
                    if ser is None or len(ser) == 0:
                        continue
                    reindexed = ser.reindex(all_dates).ffill().bfill()
                    if reindexed.isna().all():
                        continue
                    base = float(reindexed.iloc[0])
                    if base <= 0:
                        continue
                    return_pct = (reindexed / base - 1) * 100
                    final_bh_by_inst[inst] = float(return_pct.iloc[-1])
                    ax_bottom.plot(
                        all_dates, return_pct.values, label=inst, linewidth=2, alpha=0.8, color=colors[i % len(colors)]
                    )
                ax_bottom.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
                ax_bottom.set_xlabel('Date')
                ax_bottom.set_ylabel('Buy-and-hold return (%)')
                ax_bottom.set_title('Market (buy-and-hold) return % per instrument')
                if len(final_bh_by_inst) > MAX_LEGEND_INSTRUMENTS:
                    handles, labels = ax_bottom.get_legend_handles_labels()
                    best_10 = sorted(final_bh_by_inst.keys(), key=lambda inst: final_bh_by_inst.get(inst, float('-inf')), reverse=True)[:MAX_LEGEND_INSTRUMENTS]
                    label_to_handle = dict(zip(labels, handles))
                    sel_handles = [label_to_handle[inst] for inst in best_10]
                    sel_labels = best_10
                    ax_bottom.legend(sel_handles, sel_labels, fontsize=9, loc='best')
                else:
                    ax_bottom.legend(loc='best', fontsize=9)
                ax_bottom.grid(True, alpha=0.3)
        else:
            axes[0].set_xlabel('Date')

        plt.tight_layout()
        if filename is None:
            filename = f"value_gain_per_instrument_over_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def generate_pnl_vs_duration_scatter(
        self,
        result: WalkForwardResult,
        filename: Optional[str] = None,
    ) -> str:
        """Scatter: pnl% vs duration of each trade (closed positions only)."""
        if not result or not result.simulation.positions:
            return ""
        closed = [p for p in result.simulation.positions if p.exit_timestamp and p.entry_timestamp]
        if not closed:
            return ""
        durations = [(p.exit_timestamp - p.entry_timestamp).days for p in closed]
        pnl_pcts = [(p.pnl / p.cost_basis * 100) if p.cost_basis > 0 else 0.0 for p in closed]
        colors = ['green' if p > 0 else 'red' for p in pnl_pcts]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(durations, pnl_pcts, c=colors, alpha=0.6, s=30, edgecolors='darkgray')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax.set_xlabel('Duration (days)')
        ax.set_ylabel('P&L %')
        ax.set_title(f'P&L % vs Duration: {result.config.name}')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if filename is None:
            filename = f"scatter_pnl_pct_vs_duration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def generate_confidence_risk_vs_pnl_scatter(
        self,
        result: WalkForwardResult,
        filename: Optional[str] = None,
    ) -> str:
        """Scatter: certainty vs pnl% and risk (e.g. risk_reward_ratio) vs pnl% (two subplots)."""
        if not result or not result.simulation.positions:
            return ""
        closed = [p for p in result.simulation.positions if p.exit_timestamp]
        if not closed:
            return ""
        pnl_pcts = [(p.pnl / p.cost_basis * 100) if p.cost_basis > 0 else 0.0 for p in closed]
        certainties = [p.certainty for p in closed]
        risks = [p.risk_reward_ratio if p.risk_reward_ratio else 0.0 for p in closed]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(certainties, pnl_pcts, alpha=0.6, s=30, c='steelblue', edgecolors='darkgray')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax1.set_xlabel('Certainty at entry')
        ax1.set_ylabel('P&L %')
        ax1.set_title('Certainty vs P&L %')
        ax1.grid(True, alpha=0.3)
        ax2.scatter(risks, pnl_pcts, alpha=0.6, s=30, c='teal', edgecolors='darkgray')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax2.set_xlabel('Risk/reward ratio')
        ax2.set_ylabel('P&L %')
        ax2.set_title('Risk vs P&L %')
        ax2.grid(True, alpha=0.3)
        fig.suptitle(f'Confidence/Risk vs P&L: {result.config.name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        if filename is None:
            filename = f"scatter_confidence_risk_vs_pnl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def generate_gain_per_instrument(
        self,
        result: WalkForwardResult,
        filename: Optional[str] = None,
        price_data_by_instrument: Optional[Dict[str, pd.Series]] = None,
    ) -> str:
        """Bar chart: strategy total gain % vs buy-and-hold return % per instrument (grouped bars)."""
        if not result or not result.simulation.positions:
            return ""
        closed = [p for p in result.simulation.positions if p.exit_timestamp and p.cost_basis > 0]
        if not closed:
            return ""
        by_inst: Dict[str, List[float]] = {}
        for p in closed:
            inst = p.instrument if p.instrument else 'unknown'
            by_inst.setdefault(inst, []).append(p.pnl / p.cost_basis * 100)
        instruments = sorted(by_inst.keys())
        strategy_gains = [sum(by_inst[inst]) for inst in instruments]
        start_date = result.evaluation_start_date
        end_date = result.evaluation_end_date
        bh_returns: Optional[List[Optional[float]]] = None
        if price_data_by_instrument and start_date is not None and end_date is not None:
            bh_returns = []
            for inst in instruments:
                series = price_data_by_instrument.get(inst)
                if series is None or len(series) < 2:
                    bh_returns.append(None)
                    continue
                mask = (series.index >= start_date) & (series.index <= end_date)
                sub = series.loc[mask]
                if len(sub) < 2:
                    bh_returns.append(None)
                    continue
                p0 = float(sub.iloc[0])
                p1 = float(sub.iloc[-1])
                if p0 <= 0:
                    bh_returns.append(None)
                    continue
                bh_returns.append((p1 - p0) / p0 * 100)
        fig, ax = plt.subplots(figsize=(max(8, len(instruments) * 1.2), 5))
        x = np.arange(len(instruments))
        width = 0.35
        ax.bar(x - width / 2, strategy_gains, width, label='Strategy', color='steelblue', alpha=0.7)
        if bh_returns is not None:
            bh_vals = [b if b is not None else np.nan for b in bh_returns]
            ax.bar(x + width / 2, bh_vals, width, label='B&H', color='orange', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(instruments, rotation=45, ha='right')
        ax.set_xlabel('Instrument')
        ax.set_ylabel('Return %')
        ax.set_title(f'Strategy vs Buy-and-Hold return % per Instrument: {result.config.name}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        if filename is None:
            filename = f"gain_per_instrument_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def generate_trades_per_instrument(
        self,
        result: WalkForwardResult,
        filename: Optional[str] = None,
    ) -> str:
        """Bar chart: total, winning, and losing trade counts per instrument (grouped bars)."""
        if not result or not result.simulation.positions:
            return ""
        closed = [p for p in result.simulation.positions if p.exit_timestamp]
        if not closed:
            return ""
        by_inst: Dict[str, Dict[str, int]] = {}
        for p in closed:
            inst = p.instrument if p.instrument else 'unknown'
            if inst not in by_inst:
                by_inst[inst] = {'total': 0, 'winning': 0, 'losing': 0}
            by_inst[inst]['total'] += 1
            if p.pnl > 0:
                by_inst[inst]['winning'] += 1
            else:
                by_inst[inst]['losing'] += 1
        instruments = sorted(by_inst.keys())
        totals = [by_inst[inst]['total'] for inst in instruments]
        winnings = [by_inst[inst]['winning'] for inst in instruments]
        losings = [by_inst[inst]['losing'] for inst in instruments]
        fig, ax = plt.subplots(figsize=(max(8, len(instruments) * 1.2), 5))
        x = np.arange(len(instruments))
        width = 0.25
        ax.bar(x - width, totals, width, label='Total', color='steelblue', alpha=0.7)
        ax.bar(x, winnings, width, label='Winning', color='green', alpha=0.7)
        ax.bar(x + width, losings, width, label='Losing', color='red', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(instruments, rotation=45, ha='right')
        ax.set_xlabel('Instrument')
        ax.set_ylabel('Trade count')
        ax.set_title(f'Trades per Instrument (total / winning / losing): {result.config.name}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        if filename is None:
            filename = f"trades_per_instrument_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def _line_panel_from_continuous(
        self,
        getter: Callable[[object], Optional[float]],
        closed: list,
        pnl_pcts: list,
        n_bins: int,
        value_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build binned (bin_centers, mean_pnls, counts) for a continuous indicator.
        getter(pos) returns value or None. value_range=(vmin,vmax) => equal-width bins; else percentile-based.
        Returns empty arrays if fewer than 2 valid pairs."""
        pairs: List[Tuple[float, float]] = []
        for i, p in enumerate(closed):
            if i >= len(pnl_pcts):
                break
            v = getter(p)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                continue
            try:
                pairs.append((float(v), float(pnl_pcts[i])))
            except (TypeError, ValueError, IndexError):
                pass
        if len(pairs) < 2:
            return np.array([]), np.array([]), np.array([])
        values = np.array([x[0] for x in pairs])
        pnls = np.array([x[1] for x in pairs])
        n_bins = min(n_bins, max(5, len(pairs) // 3))
        if value_range is not None:
            vmin, vmax = value_range
            edges = np.linspace(vmin, vmax, n_bins + 1)
        else:
            edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
            edges[0] = min(edges[0], values.min() - 1e-9)
            edges[-1] = max(edges[-1], values.max() + 1e-9)
        bin_idx = np.clip(np.digitize(values, edges, right=False) - 1, 0, n_bins - 1)
        centers: List[float] = []
        means: List[float] = []
        counts: List[int] = []
        for j in range(n_bins):
            mask = bin_idx == j
            if not np.any(mask):
                continue
            centers.append((float(edges[j]) + float(edges[j + 1])) / 2)
            means.append(float(np.mean(pnls[mask])))
            counts.append(int(np.sum(mask)))
        return np.array(centers), np.array(means), np.array(counts)

    def _add_line_panel(
        self,
        ax,
        bin_centers: np.ndarray,
        mean_pnls: np.ndarray,
        counts: np.ndarray,
        xlabel: str,
        title: str,
    ) -> None:
        """Plot mean P&L vs bin center as line, zero line, labels. Title can include N trades."""
        if len(bin_centers) == 0:
            return
        ax.plot(bin_centers, mean_pnls, marker='o', linestyle='-', color='steelblue', markersize=4)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean P&L % at exit')
        n_trades = int(counts.sum()) if len(counts) else 0
        ax.set_title(f'{title} (N={n_trades})' if n_trades else title)
        ax.grid(True, alpha=0.3, axis='y')

    def generate_indicator_best_worst_overview(
        self,
        result: WalkForwardResult,
        filename: Optional[str] = None,
    ) -> str:
        """Panels: (1) mean P&L % per instrument; (2) by confirmation count (bars);
        (3–8) by certainty, RSI, EMA short, EMA long, MACD hist, MACD signal as line (binned mean P&L vs value)."""
        if not result or not result.simulation.positions:
            return ""
        closed = [p for p in result.simulation.positions if p.exit_timestamp and p.cost_basis > 0]
        if not closed:
            return ""
        pnl_pcts = [(p.pnl / p.cost_basis * 100) if p.cost_basis > 0 else 0.0 for p in closed]
        by_inst: Dict[str, List[float]] = {}
        for p, pct in zip(closed, pnl_pcts):
            inst = p.instrument if p.instrument else 'unknown'
            by_inst.setdefault(inst, []).append(pct)
        instruments = sorted(by_inst.keys())
        mean_pnls_inst = [float(np.mean(by_inst[inst])) for inst in instruments]

        # By confirmation count (1, 2, 3)
        by_conf: Dict[int, List[float]] = {}
        for p, pct in zip(closed, pnl_pcts):
            c = getattr(p, 'indicator_confirmations', None)
            if c is not None:
                try:
                    k = int(c)
                    by_conf.setdefault(k, []).append(pct)
                except (TypeError, ValueError):
                    pass
        conf_counts = sorted(by_conf.keys())
        mean_pnls_conf = [float(np.mean(by_conf[k])) for k in conf_counts]
        conf_labels = [str(k) for k in conf_counts]

        # Line panels: certainty, RSI, EMA short, EMA long, MACD hist, MACD signal (binned mean P&L vs value)
        cert_centers, cert_means, cert_counts = self._line_panel_from_continuous(
            lambda p: getattr(p, 'certainty', None), closed, pnl_pcts, n_bins=18, value_range=(0.0, 1.0)
        )
        rsi_centers, rsi_means, rsi_counts = self._line_panel_from_continuous(
            lambda p: getattr(p, 'rsi_value', None), closed, pnl_pcts, n_bins=12, value_range=(0.0, 100.0)
        )
        ema_short_centers, ema_short_means, ema_short_counts = self._line_panel_from_continuous(
            lambda p: getattr(p, 'ema_short', None), closed, pnl_pcts, n_bins=12, value_range=None
        )
        ema_long_centers, ema_long_means, ema_long_counts = self._line_panel_from_continuous(
            lambda p: getattr(p, 'ema_long', None), closed, pnl_pcts, n_bins=12, value_range=None
        )
        macd_hist_centers, macd_hist_means, macd_hist_counts = self._line_panel_from_continuous(
            lambda p: getattr(p, 'macd_histogram', None), closed, pnl_pcts, n_bins=10, value_range=None
        )
        macd_sig_centers, macd_sig_means, macd_sig_counts = self._line_panel_from_continuous(
            lambda p: getattr(p, 'macd_signal', None), closed, pnl_pcts, n_bins=10, value_range=None
        )

        n_rows = 1
        n_rows += 1 if conf_labels else 0
        n_rows += 1 if len(cert_centers) else 0
        n_rows += 1 if len(rsi_centers) else 0
        n_rows += 1 if len(ema_short_centers) else 0
        n_rows += 1 if len(ema_long_centers) else 0
        n_rows += 1 if len(macd_hist_centers) else 0
        n_rows += 1 if len(macd_sig_centers) else 0
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(max(8, len(instruments) * 1.2), 4 * n_rows),
        )
        axes = np.atleast_1d(axes)
        ax_idx = 0

        def _add_bar_panel(ax, labels: List[str], means: List[float], title: str):
            x = np.arange(len(labels))
            colors = ['green' if m >= 0 else 'red' for m in means]
            ax.bar(x, means, color=colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha='right')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_ylabel('Mean P&L % at exit')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='y')

        x1 = np.arange(len(instruments))
        colors1 = ['green' if m >= 0 else 'red' for m in mean_pnls_inst]
        axes[ax_idx].bar(x1, mean_pnls_inst, color=colors1, alpha=0.7)
        axes[ax_idx].set_xticks(x1)
        axes[ax_idx].set_xticklabels(instruments, rotation=45, ha='right')
        axes[ax_idx].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[ax_idx].set_ylabel('Mean P&L % at exit')
        axes[ax_idx].set_title('Mean P&L % at exit per Instrument')
        axes[ax_idx].grid(True, alpha=0.3, axis='y')
        ax_idx += 1

        if conf_labels:
            _add_bar_panel(axes[ax_idx], [f'{lb} conf' for lb in conf_labels], mean_pnls_conf, 'Mean P&L % by Confirmation Count (what confirmations suggested)')
            ax_idx += 1

        if len(cert_centers):
            self._add_line_panel(axes[ax_idx], cert_centers, cert_means, cert_counts, 'Certainty', 'Mean P&L % by Certainty (what certainty suggested)')
            ax_idx += 1
        if len(rsi_centers):
            self._add_line_panel(axes[ax_idx], rsi_centers, rsi_means, rsi_counts, 'RSI at entry', 'Mean P&L % by RSI at entry (what RSI suggested)')
            ax_idx += 1
        if len(ema_short_centers):
            self._add_line_panel(axes[ax_idx], ema_short_centers, ema_short_means, ema_short_counts, 'EMA short at entry', 'Mean P&L % by EMA short at entry (what EMA short suggested)')
            ax_idx += 1
        if len(ema_long_centers):
            self._add_line_panel(axes[ax_idx], ema_long_centers, ema_long_means, ema_long_counts, 'EMA long at entry', 'Mean P&L % by EMA long at entry (what EMA long suggested)')
            ax_idx += 1
        if len(macd_hist_centers):
            self._add_line_panel(axes[ax_idx], macd_hist_centers, macd_hist_means, macd_hist_counts, 'MACD histogram at entry', 'Mean P&L % by MACD histogram at entry (what MACD hist suggested)')
            ax_idx += 1
        if len(macd_sig_centers):
            self._add_line_panel(axes[ax_idx], macd_sig_centers, macd_sig_means, macd_sig_counts, 'MACD signal at entry', 'Mean P&L % by MACD signal at entry (what MACD signal suggested)')
            ax_idx += 1

        fig.suptitle(f'{result.config.name} — Actual trade success vs what indicators suggested', fontsize=12, fontweight='bold')
        plt.tight_layout()
        if filename is None:
            filename = f"indicator_best_worst_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def generate_performance_timings_chart(
        self,
        result: WalkForwardResult,
        filename: Optional[str] = None,
    ) -> str:
        """Bar chart of computation time by phase and per-indicator (seconds)."""
        timings = getattr(result, 'performance_timings', None) if result else None
        if not timings:
            return ""
        phase_keys = ["data_load", "signal_detection", "portfolio_simulation"]
        phases = [(k, timings[k]) for k in phase_keys if k in timings]
        indicator_items = [(k.replace("indicator_", ""), timings[k]) for k in sorted(timings) if k.startswith("indicator_")]
        n_phase = len(phases)
        n_ind = len(indicator_items)
        if n_phase == 0 and n_ind == 0:
            return ""
        n_rows = (1 if n_phase else 0) + (1 if n_ind else 0)
        fig, axes = plt.subplots(n_rows, 1, figsize=(max(8, (n_phase + n_ind) * 0.8), 4 * n_rows))
        axes = np.atleast_1d(axes)
        ax_idx = 0
        if phases:
            ax = axes[ax_idx]
            ax_idx += 1
            labels = [p[0].replace("_", " ") for p in phases]
            vals = [p[1] for p in phases]
            x = np.arange(len(labels))
            ax.bar(x, vals, color="steelblue", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylabel("Time (s)")
            ax.set_title("Computation time by phase")
            ax.grid(True, alpha=0.3, axis="y")
        if indicator_items:
            ax = axes[ax_idx]
            labels = [p[0].replace("_", " ") for p in indicator_items]
            vals = [p[1] for p in indicator_items]
            x = np.arange(len(labels))
            ax.bar(x, vals, color="seagreen", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylabel("Time (s)")
            ax.set_title("Computation time per indicator (cumulative)")
            ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle(f"{result.config.name} – performance", fontsize=11, fontweight="bold")
        plt.tight_layout()
        if filename is None:
            filename = f"performance_timings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
    
    def generate_multi_strategy_equity_curve(
        self,
        results: List[WalkForwardResult],
        filename_prefix: Optional[str] = None,
        top_n: int = 10,
    ) -> str:
        """
        Generate multi-strategy equity curve comparing strategies over time vs 2%pa benchmark.
        
        Shows cumulative returns for top strategies, 2%pa benchmark, and buy-and-hold.
        
        Args:
            results: List of walk-forward results
            filename_prefix: Prefix for output filename
            top_n: Maximum number of strategies to show (default: 10)
            
        Returns:
            Path to the generated chart
        """
        if not results:
            return ""
        
        # Sort by alpha and take top N
        sorted_results = sorted(results, key=lambda r: getattr(r, 'active_alpha', r.outperformance), reverse=True)
        display_results = sorted_results[:top_n] if len(sorted_results) > top_n else sorted_results
        
        # Get date range from first result
        first_result = results[0]
        start_date = first_result.evaluation_start_date
        end_date = first_result.evaluation_end_date
        
        # Create date index
        date_index = pd.date_range(start=start_date, end=end_date, freq='D')
        
        interest_rate_pa = getattr(first_result.config, 'interest_rate_pa', 0.02)
        daily_rate = _daily_rate_from_pa(interest_rate_pa)
        # Cash benchmark: interest calculated daily, payout (compound) at end of each month
        initial_capital = first_result.simulation.initial_capital
        cash_balance = initial_capital
        accrued = 0.0
        prev_date = None
        cash_returns = []
        for date in date_index:
            if _is_new_month(date, prev_date):
                cash_balance += accrued
                accrued = 0.0
            days = 1 if prev_date is None else (date - prev_date).days
            if days > 0:
                accrued += cash_balance * (daily_rate * days)
            # Display balance only (accrued not in account until month-end); monthly steps
            cash_returns.append(((cash_balance - initial_capital) / initial_capital) * 100)
            prev_date = date
        
        # Calculate buy-and-hold returns from actual price data over time
        # We need to load price data to calculate the actual buy-and-hold curve
        # Use the first result's config to get instrument and load data
        try:
            from ..data.loader import DataLoader
            config = first_result.config
            instrument = config.instruments[0] if config.instruments else "djia"
            
            loader = DataLoader.from_scraper(instrument)
            df = loader.load(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
            )
            price_data = df[config.column] if config.column in df.columns else df.iloc[:, 0]
            
            # Filter to evaluation period
            price_data = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
            
            # Calculate buy-and-hold returns day by day
            initial_price = price_data.iloc[0]
            initial_capital = first_result.simulation.initial_capital
            bh_shares = initial_capital / initial_price
            
            # Create buy-and-hold returns aligned with date_index
            bh_returns = []
            for date in date_index:
                # Find closest price data point
                if date in price_data.index:
                    current_price = price_data.loc[date]
                else:
                    # Use forward fill to get latest available price
                    available_prices = price_data[price_data.index <= date]
                    if len(available_prices) > 0:
                        current_price = available_prices.iloc[-1]
                    else:
                        current_price = initial_price
                
                current_value = bh_shares * current_price
                return_pct = ((current_value - initial_capital) / initial_capital) * 100
                bh_returns.append(return_pct)
            
            bh_gain = first_result.buy_and_hold_gain  # Use stored value for label
        except Exception as e:
            # Fallback to linear interpolation if data loading fails
            bh_gain = first_result.buy_and_hold_gain
            bh_returns = [bh_gain * (i / len(date_index)) for i in range(len(date_index))]
        
        # Build equity curves for each strategy
        fig, ax = plt.subplots(figsize=(16, 10))
        
        cash_bench_label = f'{interest_rate_pa * 100:.1f}% p.a. Benchmark'
        ax.plot(date_index, cash_returns, label=cash_bench_label, color='gray', linewidth=2, linestyle='--', alpha=0.7)
        
        # Plot buy-and-hold
        ax.plot(date_index, bh_returns, label=f'Buy-and-Hold ({bh_gain:.1f}%)', color='blue', linewidth=2, alpha=0.7)
        
        # Plot each strategy (cash earns interest; display value = total_value + interest_account only, monthly steps)
        colors = plt.cm.tab10(range(len(display_results)))
        for i, result in enumerate(display_results):
            if not result.simulation.wallet_history:
                continue
            interest_rate_pa_r = getattr(result.config, 'interest_rate_pa', 0.02)
            daily_rate_strat = _daily_rate_from_pa(interest_rate_pa_r)
            initial_cap = result.simulation.initial_capital
            wallet_history = result.simulation.wallet_history
            strategy_dates = [w.timestamp for w in wallet_history]
            interest_account = 0.0
            accrued = 0.0
            prev_date = None
            strategy_values = []
            for date, wallet_state in zip(strategy_dates, wallet_history):
                if _is_new_month(date, prev_date):
                    interest_account += accrued
                    accrued = 0.0
                days = 1 if prev_date is None else (date - prev_date).days
                if days > 0 and (wallet_state.cash + interest_account) > 0:
                    accrued += (wallet_state.cash + interest_account) * (daily_rate_strat * days)
                # Display total_value + interest_account only (accrued not in account until month-end); monthly steps
                strategy_values.append(wallet_state.total_value + interest_account)
                prev_date = date
            strategy_returns_pct = [((v - initial_cap) / initial_cap) * 100 for v in strategy_values]
            strategy_series = pd.Series(strategy_returns_pct, index=pd.DatetimeIndex(strategy_dates))
            strategy_aligned = strategy_series.reindex(date_index, method='ffill').fillna(0.0)
            alpha = getattr(result, 'active_alpha', result.outperformance)
            label = f"{result.config.name[:25]} (α={alpha:+.1f}%)"
            ax.plot(date_index, strategy_aligned.values, label=label, color=colors[i], linewidth=1.5, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title(f'Multi-Strategy Equity Curve vs {interest_rate_pa * 100:.1f}% p.a. Benchmark (Top {len(display_results)})', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        if filename_prefix is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"equity_curve_vs_2pa_{timestamp}.png"
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename_prefix}_{timestamp}.png"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_performance_by_instrument(
        self,
        results: List[WalkForwardResult],
        filename_prefix: Optional[str] = None,
    ) -> str:
        """
        Generate chart showing performance breakdown by instrument.
        
        Only generated if results contain multiple instruments.
        
        Args:
            results: List of walk-forward results
            filename_prefix: Prefix for output filename
            
        Returns:
            Path to the generated chart, or empty string if not applicable
        """
        if not results:
            return ""
        
        # Collect instruments from all results
        instruments = set()
        for result in results:
            if hasattr(result.config, 'instruments') and result.config.instruments:
                instruments.update(result.config.instruments)
        
        # Only generate if we have multiple instruments
        if len(instruments) <= 1:
            return ""
        
        # Group results by instrument
        by_instrument = {}
        for result in results:
            if hasattr(result.config, 'instruments') and result.config.instruments:
                for instrument in result.config.instruments:
                    if instrument not in by_instrument:
                        by_instrument[instrument] = []
                    by_instrument[instrument].append(result)
        
        # Calculate average alpha per instrument
        instrument_alphas = {}
        instrument_counts = {}
        for instrument, inst_results in by_instrument.items():
            alphas = [getattr(r, 'active_alpha', r.outperformance) for r in inst_results]
            instrument_alphas[instrument] = np.mean(alphas) if alphas else 0.0
            instrument_counts[instrument] = len(inst_results)
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        instruments_list = sorted(instrument_alphas.keys())
        alphas_list = [instrument_alphas[inst] for inst in instruments_list]
        counts_list = [instrument_counts[inst] for inst in instruments_list]
        
        # Chart 1: Average alpha by instrument
        colors = ['green' if a > 0 else 'red' for a in alphas_list]
        bars1 = ax1.bar(instruments_list, alphas_list, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_ylabel('Average Alpha (%)', fontsize=11)
        ax1.set_title('Average Performance by Instrument', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, alpha in zip(bars1, alphas_list):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{alpha:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # Chart 2: Number of configs tested per instrument
        bars2 = ax2.bar(instruments_list, counts_list, color='steelblue', alpha=0.7)
        ax2.set_ylabel('Number of Configs Tested', fontsize=11)
        ax2.set_title('Test Coverage by Instrument', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars2, counts_list):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save chart
        if filename_prefix is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_by_instrument_{timestamp}.png"
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename_prefix}_{timestamp}.png"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
