"""
Data loading and management module.

Provides unified interface for loading data from any instrument
and filtering by date ranges.
"""
from .loader import DataLoader
from .scraper import download_instrument, list_instruments, get_available_instruments

__all__ = ['DataLoader', 'download_instrument', 'list_instruments', 'get_available_instruments']
