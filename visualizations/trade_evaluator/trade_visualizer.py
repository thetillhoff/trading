"""
Visualizes trade evaluation results on price charts.

Shows entry points, exit points, and trade outcomes with color coding.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import List, Optional
import sys

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from trade_evaluator import TradeEvaluation, TradeOutcome, EvaluationSummary


class TradeVisualizer:
    """Creates visualizations of trade evaluation results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the trade visualizer.
        
        Args:
            output_dir: Directory to save charts (default: current directory)
        """
        if output_dir is None:
            output_dir = Path.cwd()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_evaluation(
        self,
        data: pd.Series,
        evaluations: List[TradeEvaluation],
        summary: EvaluationSummary,
        title: str = "Trade Evaluation Results",
        xlabel: str = "Date",
        ylabel: Optional[str] = None,
        output_filename: Optional[str] = None,
        buy_and_hold_gain: Optional[float] = None
    ) -> Path:
        """
        Create a visualization of trade evaluation results.
        
        Args:
            data: Historical price data
            evaluations: List of trade evaluations
            summary: Evaluation summary statistics
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label (auto-generated if None)
            output_filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot price line
        ax.plot(data.index, data.values, linewidth=1.5, color='gray', alpha=0.7, label='Price')
        
        # Plot trades
        buy_wins = []
        buy_losses = []
        sell_wins = []
        sell_losses = []
        no_outcome = []
        has_connection_lines = False
        
        for eval_result in evaluations:
            signal = eval_result.signal
            entry_date = signal.timestamp
            entry_price = signal.price
            
            # Determine color and marker based on outcome
            if eval_result.outcome == TradeOutcome.TARGET_HIT:
                color = 'green'
                marker = '^' if signal.signal_type.value == 'buy' else 'v'
                size = 200
                edge_color = 'darkgreen'
                if signal.signal_type.value == 'buy':
                    buy_wins.append((entry_date, entry_price, eval_result))
                else:
                    sell_wins.append((entry_date, entry_price, eval_result))
            elif eval_result.outcome == TradeOutcome.STOP_LOSS_HIT:
                color = 'red'
                marker = '^' if signal.signal_type.value == 'buy' else 'v'
                size = 200
                edge_color = 'darkred'
                if signal.signal_type.value == 'buy':
                    buy_losses.append((entry_date, entry_price, eval_result))
                else:
                    sell_losses.append((entry_date, entry_price, eval_result))
            else:
                color = 'orange'
                marker = '^' if signal.signal_type.value == 'buy' else 'v'
                size = 150
                edge_color = 'darkorange'
                no_outcome.append((entry_date, entry_price, eval_result))
            
            # Plot entry point
            ax.scatter(entry_date, entry_price, color=color, marker=marker, s=size,
                      zorder=5, edgecolors=edge_color, linewidths=2, alpha=0.8)
            
            # Draw line to exit point if available
            if eval_result.exit_price and eval_result.exit_timestamp:
                exit_color = 'green' if eval_result.outcome == TradeOutcome.TARGET_HIT else 'red'
                # Only add label for the first connection line
                line_label = 'Entry to Exit Connection' if not has_connection_lines else ''
                ax.plot([entry_date, eval_result.exit_timestamp],
                       [entry_price, eval_result.exit_price],
                       color=exit_color, linestyle='--', alpha=0.6, linewidth=1.5, label=line_label)
                has_connection_lines = True
                
                # Mark exit point
                exit_marker = 'o' if eval_result.outcome == TradeOutcome.TARGET_HIT else 'x'
                exit_size = 100 if eval_result.outcome == TradeOutcome.TARGET_HIT else 150
                ax.scatter(eval_result.exit_timestamp, eval_result.exit_price,
                          color=exit_color, marker=exit_marker, s=exit_size,
                          zorder=4, edgecolors=edge_color, linewidths=1.5, alpha=0.8)
                
                # Add annotation with gain percentage
                if abs(eval_result.gain_percentage) > 0.1:  # Only annotate significant moves
                    ax.annotate(
                        f"{eval_result.gain_percentage:+.1f}%",
                        xy=(eval_result.exit_timestamp, eval_result.exit_price),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=exit_color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=exit_color)
                    )
        
        # Add legend entries
        if buy_wins or sell_wins:
            ax.scatter([], [], color='green', marker='^', s=200,
                      edgecolors='darkgreen', linewidths=2, label='Buy Win (Target Hit)')
            ax.scatter([], [], color='green', marker='v', s=200,
                      edgecolors='darkgreen', linewidths=2, label='Sell Win (Target Hit)')
        if buy_losses or sell_losses:
            ax.scatter([], [], color='red', marker='^', s=200,
                      edgecolors='darkred', linewidths=2, label='Buy Loss (Stop-Loss Hit)')
            ax.scatter([], [], color='red', marker='v', s=200,
                      edgecolors='darkred', linewidths=2, label='Sell Loss (Stop-Loss Hit)')
        if no_outcome:
            ax.scatter([], [], color='orange', marker='^', s=150,
                      edgecolors='darkorange', linewidths=2, label='No Outcome (Buy)')
            ax.scatter([], [], color='orange', marker='v', s=150,
                      edgecolors='darkorange', linewidths=2, label='No Outcome (Sell)')
        
        # Add exit point legend
        if any(e.exit_price for e in evaluations):
            ax.scatter([], [], color='green', marker='o', s=100,
                      edgecolors='darkgreen', linewidths=1.5, label='Target Hit')
            ax.scatter([], [], color='red', marker='x', s=150,
                      linewidths=2, label='Stop-Loss Hit')
        
        # Connection line legend is added in the loop above (first connection line gets the label)
        
        # Set title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        if ylabel is None:
            ylabel = "Price"
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Add summary statistics as text box
        stats_text = (
            f"Total Trades: {summary.total_trades} | "
            f"Win Rate: {summary.win_rate:.1f}% | "
            f"Avg Gain: {summary.average_gain:+.2f}% | "
            f"Avg Loss: {summary.average_loss:+.2f}% | "
            f"Total: {summary.total_gain:+.2f}%"
        )
        if buy_and_hold_gain is not None:
            outperformance = summary.total_gain - buy_and_hold_gain
            stats_text += f" | Buy-and-Hold: {buy_and_hold_gain:+.2f}% | Outperformance: {outperformance:+.2f}%"
        if summary.average_days_held:
            stats_text += f" | Avg Days: {summary.average_days_held:.1f}"
        
        ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
               ha='center', va='bottom', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add description text
        description = (
            "Trade Evaluation: Triangles mark entry points (^=buy, v=sell). "
            "Green = target hit (win), Red = stop-loss hit (loss).\n"
            "Dashed lines connect entry to exit points. Circles = target hit, X = stop-loss hit. "
            "Percentages show gain/loss per trade."
        )
        
        # Adjust layout first to make room for description
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave bottom 5% for description
        fig.text(0.5, 0.02, description, ha='center', va='bottom',
                fontsize=8, style='italic', color='gray')
        
        # Save figure
        if output_filename is None:
            output_filename = f"{title.lower().replace(' ', '_')}.png"
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
