"""
Unified data loader for any instrument and time range.

Loads data from scraper CSV files with support for:
- Any instrument (djia, sp500, dax, gold, eurusd, msci_world)
- Date range filtering
- Flexible path resolution (local development and Docker)
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime
from .download import TICKERS_DIR
from .scraper import list_instruments


def list_available_tickers() -> List[str]:
    """Return sorted list of ticker symbols from data/tickers/ (CSV stems). Empty if dir missing."""
    if not TICKERS_DIR.exists():
        return []
    return sorted(p.stem for p in TICKERS_DIR.glob("*.csv"))


class DataLoader:
    """
    Loads data from scraper CSV files.
    
    Supports any instrument and date range filtering.
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file containing the data
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def load(
        self,
        start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        column: Optional[str] = None
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Load data from CSV file with optional filtering.
        
        Args:
            start_date: Start date for filtering (inclusive). If None, no start filter.
            end_date: End date for filtering (inclusive). If None, no end filter.
            column: If specified, return Series for this column instead of DataFrame.
                   If None, return full DataFrame.
        
        Returns:
            DataFrame or Series with datetime index and OHLCV columns (or specified column)
        """
        # Read CSV with Date as index
        df = pd.read_csv(
            self.data_path, 
            index_col=0, 
            parse_dates=True,
        )
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Apply date range filtering
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]
        
        # Return specific column if requested
        if column is not None:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
            return df[column]
        
        return df
    
    @classmethod
    def from_instrument(
        cls,
        instrument_name: str,
        start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        column: Optional[str] = None
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Create a DataLoader and load data for an instrument.
        
        Convenience method that combines from_scraper() and load().
        
        Args:
            instrument_name: Name of the instrument (e.g., 'djia', 'sp500', 'gold')
            start_date: Start date for filtering (inclusive)
            end_date: End date for filtering (inclusive)
            column: If specified, return Series for this column instead of DataFrame
        
        Returns:
            DataFrame or Series with filtered data
        """
        loader = cls.from_scraper(instrument_name)
        return loader.load(start_date=start_date, end_date=end_date, column=column)

    @classmethod
    def from_ticker(
        cls,
        ticker: str,
        start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        column: Optional[str] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Load OHLCV for a discovered ticker from data/tickers/{ticker}.csv.

        Args:
            ticker: Yahoo Finance ticker symbol (e.g. AAPL, BRK-B).
            start_date: Start date for filtering (inclusive).
            end_date: End date for filtering (inclusive).
            column: If specified, return Series for this column.

        Returns:
            DataFrame or Series with filtered data.
        """
        path = TICKERS_DIR / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found for ticker '{ticker}': {path}. "
                "Run asset-analysis with --all-assets --analyze (or download tickers first)."
            )
        return cls(path).load(start_date=start_date, end_date=end_date, column=column)

    @classmethod
    def from_scraper(cls, instrument_name: str) -> 'DataLoader':
        """
        Create a DataLoader from an instrument name.
        
        Args:
            instrument_name: Name of the instrument (e.g., 'djia', 'sp500', 'gold')
                           Available: djia, sp500, dax, gold, eurusd, msci_world
        
        Returns:
            DataLoader instance
        """
        data_filename = f"{instrument_name}.csv"
        
        # Try multiple paths to find the instrument data
        # First, try relative to current script (for local development)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent.parent  # core/data -> core -> trading

        # Try multiple paths in order of preference
        paths_to_try = [
            project_root / "data" / data_filename,  # New unified location
            project_root / "scrapers" / "instruments" / "data" / data_filename,  # Old location
            Path("/app/data") / data_filename,  # Docker new location
            Path("/app/scrapers") / "instruments" / "data" / data_filename,  # Docker old location
        ]

        data_path = None
        for path in paths_to_try:
            if path.exists():
                data_path = path
                break
        if data_path is None and (TICKERS_DIR / data_filename).exists():
            data_path = TICKERS_DIR / data_filename
        if data_path is None:
            available = list_instruments()
            raise FileNotFoundError(
                f"Data file not found for instrument '{instrument_name}'. "
                f"Named instruments: {available}. "
                f"For tickers (e.g. NVDA), ensure data/tickers/{instrument_name}.csv exists (run --download-candidates)."
            )
        return cls(data_path)
