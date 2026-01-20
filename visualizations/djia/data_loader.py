"""
Data loader module for reading scraper data files.

This module provides a flexible interface for loading data from various scraper outputs.
Designed to be extensible for different data sources and formats.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Union


class DataLoader:
    """Loads data from scraper CSV files."""
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file containing the data
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def load(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame with datetime index and OHLCV columns
        """
        # Read the first row to get column names
        with open(self.data_path, 'r') as f:
            first_line = f.readline().strip()
            column_names = first_line.split(',')
        
        # Read CSV, skipping the first three header rows (Price, Ticker, Date)
        # The actual data starts at row 4 (index 3), use column names from first row
        df = pd.read_csv(
            self.data_path, 
            skiprows=3,  # Skip Price, Ticker, and Date header rows
            index_col=0, 
            parse_dates=True,
            names=column_names[1:]  # Skip the first column (Date) as it's the index
        )
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    @classmethod
    def from_scraper(cls, scraper_name: str, data_filename: str = None) -> 'DataLoader':
        """
        Create a DataLoader from a scraper name.
        
        Args:
            scraper_name: Name of the scraper (e.g., 'djia')
            data_filename: Optional custom data filename (defaults to '{scraper_name}_data.csv')
        
        Returns:
            DataLoader instance
        """
        if data_filename is None:
            data_filename = f"{scraper_name}_data.csv"
        
        # Try multiple paths to find the scraper data
        # First, try relative to current script (for local development)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        
        # Try project_root/scrapers path first
        data_path = project_root / "scrapers" / scraper_name / data_filename
        
        # If not found, try /app/scrapers (Docker container path)
        if not data_path.exists():
            docker_path = Path("/app/scrapers") / scraper_name / data_filename
            if docker_path.exists():
                data_path = docker_path
            else:
                # Try /app/project_root/scrapers (alternative Docker mount)
                alt_path = Path("/app/project_root") / "scrapers" / scraper_name / data_filename
                if alt_path.exists():
                    data_path = alt_path
        
        return cls(data_path)
