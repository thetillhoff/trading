"""
Visualizes trading signals on price charts with buy/sell markers and targets.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import sys

# Use absolute import for direct script execution
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
from signal_detector import TradingSignal, SignalType


class SignalVisualizer:
    """Creates visualizations with trading signals marked."""
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save charts (None for current directory)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
                plt.style.use('default')
    
    def plot_with_signals(
        self,
        data: pd.Series,
        signals: List[TradingSignal],
        title: str = "Price Chart with Trading Signals",
        xlabel: str = "Date",
        ylabel: str = "Price",
        output_filename: Optional[str] = None,
        figsize: tuple = (14, 8)
    ) -> Path:
        """
        Create a chart with trading signals marked.
        
        Args:
            data: Time series data
            signals: List of trading signals to mark
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            output_filename: Output filename (auto-generated if None)
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price line
        ax.plot(data.index, data.values, linewidth=1.5, color='gray', alpha=0.7, label='Price')
        
        # Plot signals
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        # Track if we have targets/stop-losses for legend
        has_targets = any(s.target_price for s in signals)
        has_stop_loss = any(s.stop_loss for s in signals)
        
        # Plot buy signals
        if buy_signals:
            buy_dates = [s.timestamp for s in buy_signals]
            buy_prices = [s.price for s in buy_signals]
            ax.scatter(
                buy_dates, buy_prices,
                color='green', marker='^', s=200, zorder=5,
                label='Buy Signal', edgecolors='darkgreen', linewidths=2
            )
            
            # Plot buy targets
            for signal in buy_signals:
                if signal.target_price:
                    ax.plot(
                        [signal.timestamp, signal.timestamp],
                        [signal.price, signal.target_price],
                        color='green', linestyle='--', alpha=0.5, linewidth=1.5
                    )
                    ax.scatter(
                        signal.timestamp, signal.target_price,
                        color='lightgreen', marker='o', s=100, zorder=4,
                        edgecolors='green', linewidths=1
                    )
                
                # Plot stop loss
                if signal.stop_loss:
                    ax.scatter(
                        signal.timestamp, signal.stop_loss,
                        color='red', marker='x', s=150, zorder=4,
                        linewidths=2
                    )
        
        # Plot sell signals
        if sell_signals:
            sell_dates = [s.timestamp for s in sell_signals]
            sell_prices = [s.price for s in sell_signals]
            ax.scatter(
                sell_dates, sell_prices,
                color='red', marker='v', s=200, zorder=5,
                label='Sell Signal', edgecolors='darkred', linewidths=2
            )
            
            # Plot sell targets
            for signal in sell_signals:
                if signal.target_price:
                    ax.plot(
                        [signal.timestamp, signal.timestamp],
                        [signal.price, signal.target_price],
                        color='red', linestyle='--', alpha=0.5, linewidth=1.5
                    )
                    ax.scatter(
                        signal.timestamp, signal.target_price,
                        color='lightcoral', marker='o', s=100, zorder=4,
                        edgecolors='red', linewidths=1
                    )
                
                # Plot stop loss
                if signal.stop_loss:
                    ax.scatter(
                        signal.timestamp, signal.stop_loss,
                        color='green', marker='x', s=150, zorder=4,
                        linewidths=2
                    )
        
        # Add target and stop-loss to legend if they exist
        if has_targets:
            # Add dummy scatter for target legend entry
            ax.scatter(
                [], [], color='gray', marker='o', s=100,
                edgecolors='black', linewidths=1, label='Target Price'
            )
        
        if has_stop_loss:
            # Add dummy scatter for stop-loss legend entry
            ax.scatter(
                [], [], color='gray', marker='x', s=150,
                linewidths=2, label='Stop Loss'
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Add description text below the plot
        description = (
            "Trading Signals Reading: Triangles mark entry points (^=buy, v=sell). "
            "Circles show target prices (connected by dashed lines).\n"
            "X markers indicate stop-loss levels. "
            "Targets use Fibonacci projections; stop-loss based on risk/reward ratio."
        )
        # Adjust layout first to make room for description
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave bottom 5% for description
        fig.text(0.5, 0.02, description, ha='center', va='bottom', fontsize=9, 
                style='italic', color='gray')
        
        # Save figure
        if output_filename is None:
            output_filename = f"{title.lower().replace(' ', '_')}.png"
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
