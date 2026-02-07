"""
Data loading and management module.

Provides unified interface for loading data from any instrument
and filtering by date ranges.
"""
from .loader import DataLoader, list_available_tickers
from .scraper import download_instrument, list_instruments, get_available_instruments
from .preparation import prepare_and_validate, VerifiedDataPrepResult, DataPreparationError

__all__ = [
    'DataLoader',
    'list_available_tickers',
    'download_instrument',
    'list_instruments',
    'get_available_instruments',
    'prepare_and_validate',
    'VerifiedDataPrepResult',
    'DataPreparationError',
]
