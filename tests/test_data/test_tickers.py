"""Tests for download_ticker and DataLoader.from_ticker (discovered tickers OHLCV)."""
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.data.download import download_ticker, TICKERS_DIR
from core.data.loader import DataLoader, list_available_tickers


class TestDownloadTicker:
    def test_downloads_and_caches_to_tickers_dir(self, tmp_path):
        df = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Close": [100.5], "Volume": [1e6]},
            index=pd.DatetimeIndex(["2020-01-02"]),
        )
        with patch("core.data.download.TICKERS_DIR", tmp_path):
            with patch("core.data.download.yf.download", return_value=df):
                result, used_cache = download_ticker("AAPL", start_date="2020-01-01", quiet=True)
        assert result is not None
        assert len(result) == 1
        assert used_cache is False
        assert (tmp_path / "AAPL.csv").exists()
        loaded = pd.read_csv(tmp_path / "AAPL.csv", index_col=0, parse_dates=True)
        assert loaded.index[0].year == 2020

    def test_returns_none_when_no_data(self, tmp_path):
        with patch("core.data.download.TICKERS_DIR", tmp_path):
            with patch("core.data.download.yf.download", return_value=pd.DataFrame()):
                result, _ = download_ticker("FAKE", start_date="2020-01-01", quiet=True)
        assert result is None

    def test_uses_cache_when_exists_and_update_stale_false(self, tmp_path):
        # Cache must cover requested start: first row on 2020-01-01
        (tmp_path / "AAPL.csv").write_text(
            "Date,Open,High,Low,Close,Volume\n"
            "2020-01-01,100,101,99,100.5,1000000\n"
        )
        with patch("core.data.download.TICKERS_DIR", tmp_path):
            result, used_cache = download_ticker(
                "AAPL", start_date="2020-01-01", quiet=True, update_stale=False
            )
        assert result is not None
        assert len(result) == 1
        assert used_cache is True


class TestDataLoaderFromTicker:
    def test_loads_from_tickers_dir(self, tmp_path):
        (tmp_path / "AAPL.csv").write_text(
            "Date,Open,High,Low,Close,Volume\n"
            "2020-01-02,100,101,99,100.5,1000000\n"
        )
        with patch("core.data.loader.TICKERS_DIR", tmp_path):
            df = DataLoader.from_ticker(
                "AAPL", start_date="2020-01-01", end_date="2020-01-31"
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "Close" in df.columns

    def test_raises_when_file_missing(self, tmp_path):
        with patch("core.data.loader.TICKERS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError) as exc_info:
                DataLoader.from_ticker("MISSING", start_date="2020-01-01")
        assert "MISSING" in str(exc_info.value)
        assert "data/tickers" in str(exc_info.value) or "tickers" in str(exc_info.value)


class TestListAvailableTickers:
    """list_available_tickers() returns CSV stems from data/tickers/, sorted."""

    def test_empty_dir_returns_empty_list(self, tmp_path):
        with patch("core.data.loader.TICKERS_DIR", tmp_path):
            assert list_available_tickers() == []

    def test_missing_dir_returns_empty_list(self, tmp_path):
        missing = tmp_path / "nonexistent"
        assert not missing.exists()
        with patch("core.data.loader.TICKERS_DIR", missing):
            assert list_available_tickers() == []

    def test_returns_sorted_stems(self, tmp_path):
        (tmp_path / "NVDA.csv").write_text("Date,Close\n2020-01-01,100\n")
        (tmp_path / "A.csv").write_text("Date,Close\n2020-01-01,50\n")
        (tmp_path / "AAPL.csv").write_text("Date,Close\n2020-01-01,150\n")
        with patch("core.data.loader.TICKERS_DIR", tmp_path):
            assert list_available_tickers() == ["A", "AAPL", "NVDA"]
