"""
Visualization module for creating charts from processed data.

Designed to be extensible for different chart types and styles.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional, Union
import pandas as pd


class Visualizer:
    """Creates visualizations from processed data."""
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save charts (None for current directory)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style (fallback to default if seaborn not available)
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
                plt.style.use('default')
    
    def plot_line(
        self,
        data: pd.Series,
        title: str = "Time Series Chart",
        xlabel: str = "Date",
        ylabel: str = "Value",
        output_filename: Optional[str] = None,
        figsize: tuple = (12, 6)
    ) -> Path:
        """
        Create a line chart from time series data.
        
        Args:
            data: Series with datetime index
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            output_filename: Output filename (auto-generated if None)
            figsize: Figure size (width, height)
        
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(data.index, data.values, linewidth=2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        if output_filename is None:
            output_filename = f"{title.lower().replace(' ', '_')}.png"
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_multiple_lines(
        self,
        data: pd.DataFrame,
        title: str = "Multi-Series Chart",
        xlabel: str = "Date",
        ylabel: str = "Value",
        output_filename: Optional[str] = None,
        figsize: tuple = (12, 6)
    ) -> Path:
        """
        Create a multi-line chart from DataFrame.
        
        Args:
            data: DataFrame with datetime index and multiple columns
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            output_filename: Output filename (auto-generated if None)
            figsize: Figure size (width, height)
        
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for column in data.columns:
            ax.plot(data.index, data[column], label=column, linewidth=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        if output_filename is None:
            output_filename = f"{title.lower().replace(' ', '_')}.png"
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
