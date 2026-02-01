"""
Asset analysis: instrument metadata, returns/volatility/correlation, candidate scoring.

Reuses core.data download and loader; all data is cache-first (no re-download on rerun).
"""
from .metadata import (
    fetch_metadata,
    load_metadata,
    save_metadata,
    available_assets_metadata_path,
)
from .analytics import compute_returns, compute_volatility_summary, compute_correlation_matrix
from .candidates import score_candidates
from .discovery import get_available_assets, get_available_assets_all_sources, get_all_sources

__all__ = [
    "fetch_metadata",
    "load_metadata",
    "save_metadata",
    "available_assets_metadata_path",
    "compute_returns",
    "compute_volatility_summary",
    "compute_correlation_matrix",
    "score_candidates",
    "get_available_assets",
    "get_available_assets_all_sources",
    "get_all_sources",
]
