"""
Visualization module for creating charts from processed data.

Designed to be extensible for different chart types and styles.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
# Use absolute import to match visualize_djia.py pattern
from elliott_wave_detector import ElliottWaveDetector, Wave, WaveType, WaveLabel


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
    
    def plot_line_with_elliott_waves(
        self,
        data: pd.Series,
        title: str = "Time Series Chart with Elliott Waves",
        xlabel: str = "Date",
        ylabel: str = "Value",
        output_filename: Optional[str] = None,
        figsize: tuple = (12, 6),
        show_waves: bool = True,
        min_confidence: float = 0.6,
        min_wave_size_ratio: float = 0.05,
        only_complete_patterns: bool = False
    ) -> Path:
        """
        Create a line chart with Elliott Wave color coding.
        
        Args:
            data: Series with datetime index
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            output_filename: Output filename (auto-generated if None)
            figsize: Figure size (width, height)
            show_waves: Whether to detect and display Elliott Waves
            min_confidence: Minimum confidence (0.0-1.0) to display a wave (default: 0.6)
            min_wave_size_ratio: Minimum wave size as ratio of price range (default: 0.05 = 5%)
            only_complete_patterns: Only show complete 5-wave or 3-wave patterns (default: False)
            
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot base line
        ax.plot(data.index, data.values, linewidth=1.5, color='gray', alpha=0.5, label='Price')
        
        waves = []
        if show_waves:
            # Detect Elliott Waves with filtering
            detector = ElliottWaveDetector()
            waves = detector.detect_waves(
                data,
                min_confidence=min_confidence,
                min_wave_size_ratio=min_wave_size_ratio,
                only_complete_patterns=only_complete_patterns
            )
            
            if waves:
                # Define distinct colors for each wave type
                # All 8 waves (1-5, a-c) have unique, clearly distinguishable colors
                # Color scheme designed for maximum visual distinction
                impulse_colors = {
                    WaveLabel.WAVE_1.value: '#2E7D32',  # Green
                    WaveLabel.WAVE_2.value: '#D32F2F',  # Red
                    WaveLabel.WAVE_3.value: '#1976D2',  # Blue
                    WaveLabel.WAVE_4.value: '#FF6F00',  # Deep Orange
                    WaveLabel.WAVE_5.value: '#6A1B9A',  # Purple
                }
                
                # Corrective waves - distinct from impulse waves
                corrective_colors = {
                    WaveLabel.WAVE_A.value: '#E91E63',  # Pink/Magenta
                    WaveLabel.WAVE_B.value: '#00838F',  # Teal/Cyan (distinct from blue and green)
                    WaveLabel.WAVE_C.value: '#F9A825',  # Golden Yellow (distinct from orange and amber)
                }
                
                # Plot each wave segment with appropriate color
                plotted_labels = set()
                for wave in waves:
                    if wave.start_idx < len(data) and wave.end_idx < len(data):
                        wave_data = data.iloc[wave.start_idx:wave.end_idx+1]
                        
                        if wave.wave_type == WaveType.IMPULSE:
                            color = impulse_colors.get(wave.label.value, '#757575')
                        else:
                            color = corrective_colors.get(wave.label.value, '#757575')
                        
                        label = f"Wave {wave.label.value}" if wave.label.value not in plotted_labels else ""
                        if label:
                            plotted_labels.add(wave.label.value)
                        
                        ax.plot(
                            wave_data.index,
                            wave_data.values,
                            linewidth=2.5,
                            color=color,
                            label=label
                        )
                
                # Add legend
                ax.legend(loc='best', fontsize=9, ncol=2)
            else:
                # No waves detected, just show the base line
                ax.plot(data.index, data.values, linewidth=2, color='black', label='Price')
        else:
            # No wave detection requested
            ax.plot(data.index, data.values, linewidth=2, color='black', label='Price')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Add description text below the plot if waves are shown
        if show_waves and waves:
            description = (
                "Elliott Wave Reading: Impulse waves (1-5) show trend direction; "
                "corrective waves (a-c) show corrections.\n"
                "Wave 2/4 ends = potential buy; Wave 5 end = potential sell. "
                "Each wave color is distinct (see legend)."
            )
            # Adjust layout first to make room for description
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave bottom 5% for description
            fig.text(0.5, 0.02, description, ha='center', va='bottom', fontsize=9, 
                    style='italic', color='gray')
        else:
            plt.tight_layout()
        
        # Save figure
        if output_filename is None:
            output_filename = f"{title.lower().replace(' ', '_')}.png"
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
