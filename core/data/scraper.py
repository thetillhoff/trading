"""
Data download interface for instruments.

Provides access to scraper functionality for downloading
historical data from Yahoo Finance.
"""
import sys
from pathlib import Path
from typing import Dict, Optional, List

# Import from core data download module
try:
    from .download import (
        download_instrument as _download_instrument,
        download_all as _download_all,
        list_instruments as _list_instruments,
        INSTRUMENTS
    )
except ImportError as e:
    print(f"Warning: Could not import from core.data.download: {e}")
    # Fallback - define minimal interface
    INSTRUMENTS = {}
    def _download_instrument(*args, **kwargs):
        raise ImportError("Download module not found.")
    def _download_all(*args, **kwargs):
        raise ImportError("Download module not found.")
    def _list_instruments():
        print("Download module not found.")


def download_instrument(
    name: str,
    force_refresh: bool = False,
    start_date: str = "1990-01-01",
) -> Optional:
    """
    Download data for a single instrument.
    
    Args:
        name: Instrument name (key in INSTRUMENTS dict)
        force_refresh: If True, download even if cached data exists
        start_date: Start date for historical data
    
    Returns:
        DataFrame with OHLCV data, or None if download failed
    """
    return _download_instrument(name, force_refresh=force_refresh, start_date=start_date)


def download_all(
    force_refresh: bool = False,
    start_date: str = "1990-01-01"
) -> Dict:
    """
    Download data for all instruments.
    
    Args:
        force_refresh: If True, re-download all data
        start_date: Start date for historical data
    
    Returns:
        Dictionary mapping instrument names to DataFrames
    """
    return _download_all(force_refresh=force_refresh, start_date=start_date)


def list_instruments() -> None:
    """Print available instruments."""
    _list_instruments()


def get_available_instruments() -> List[str]:
    """
    Get list of available instrument names.
    
    Returns:
        List of instrument names
    """
    return list(INSTRUMENTS.keys())
