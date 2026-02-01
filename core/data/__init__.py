"""
Data loading and management module.

Provides unified interface for loading data from any instrument
and filtering by date ranges.
"""
from .loader import DataLoader, list_available_tickers
from .scraper import download_instrument, list_instruments, get_available_instruments

__all__ = [
    'DataLoader',
    'list_available_tickers',
    'download_instrument',
    'list_instruments',
    'get_available_instruments',
]
