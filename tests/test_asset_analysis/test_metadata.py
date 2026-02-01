"""Tests for asset_analysis.metadata (fetch, load, save; cache-first)."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.asset_analysis.metadata import (
    _normalize_info,
    load_metadata,
    save_metadata,
    fetch_metadata,
)


class TestNormalizeInfo:
    """Test _normalize_info extracts expected fields."""

    def test_extracts_expected_keys(self):
        info = {
            "sector": "Technology",
            "industry": "Software",
            "marketCap": 1_000_000,
            "averageVolume": 5_000_000,
            "shortName": "FOO",
            "longName": "Foo Inc.",
        }
        out = _normalize_info(info)
        assert out["sector"] == "Technology"
        assert out["industry"] == "Software"
        assert out["market_cap"] == 1_000_000
        assert out["average_volume"] == 5_000_000
        assert out["short_name"] == "FOO"
        assert out["long_name"] == "Foo Inc."

    def test_missing_keys_become_none(self):
        out = _normalize_info({})
        assert out["sector"] is None
        assert out["market_cap"] is None


class TestLoadSaveMetadata:
    """Test load_metadata and save_metadata round-trip."""

    def test_save_and_load_roundtrip(self):
        meta = {"a": {"sector": "Tech", "market_cap": 100}, "b": {"sector": "Health"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.json"
            save_metadata(meta, path=path)
            assert path.exists()
            loaded = load_metadata(path=path)
            assert loaded == meta

    def test_load_missing_file_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonexistent.json"
            assert load_metadata(path=path) == {}


class TestFetchMetadata:
    """Test fetch_metadata (cache-first; mock yfinance)."""

    def test_fetch_uses_cache_when_file_exists_and_not_refresh(self):
        meta = {"djia": {"sector": "Index", "market_cap": None}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instrument_metadata.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            with patch("core.asset_analysis.metadata._metadata_path", return_value=path):
                with patch("core.asset_analysis.metadata.DATA_DIR", path.parent):
                    result = fetch_metadata(instruments=["djia"], refresh=False)
            assert result == meta

    def test_fetch_calls_yfinance_when_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instrument_metadata.json"
            with patch("core.asset_analysis.metadata._metadata_path", return_value=path):
                with patch("core.asset_analysis.metadata.DATA_DIR", path.parent):
                    mock_ticker = MagicMock()
                    mock_ticker.info = {"sector": "Tech", "marketCap": 100}
                    with patch("yfinance.Ticker", return_value=mock_ticker):
                        result = fetch_metadata(instruments=["djia"], refresh=True)
            assert "djia" in result
            assert result["djia"]["sector"] == "Tech"
            assert result["djia"]["market_cap"] == 100

    def test_fetch_by_tickers_uses_available_assets_path_and_yfinance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "available_assets_metadata.json"
            with patch("core.asset_analysis.metadata.available_assets_metadata_path", return_value=path):
                with patch("core.asset_analysis.metadata.DATA_DIR", path.parent):
                    mock_ticker = MagicMock()
                    mock_ticker.info = {"sector": "Tech", "marketCap": 200}
                    with patch("yfinance.Ticker", return_value=mock_ticker):
                        result = fetch_metadata(tickers=["AAPL", "MSFT"], refresh=True)
            assert "AAPL" in result
            assert "MSFT" in result
            assert result["AAPL"]["sector"] == "Tech"
            assert result["AAPL"]["market_cap"] == 200
            assert path.exists()
            loaded = load_metadata(path=path)
            assert loaded == result
