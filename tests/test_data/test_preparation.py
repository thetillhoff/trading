"""
Tests for data preparation module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from core.data.preparation import prepare_and_validate, DataPreparationError, _generate_eval_dates


def _synthetic_series(days: int = 400, start: str = "2010-01-01", base: float = 100.0):
    """Synthetic price series for testing."""
    dates = pd.date_range(start, periods=days, freq="B")
    rng = np.random.RandomState(42)
    prices = base + np.cumsum(rng.randn(days) * 0.5) + np.linspace(0, 10, days)
    return pd.Series(np.maximum(prices, 1.0), index=dates, name="Close")


def test_generate_eval_dates():
    """Test eval_dates generation from a series."""
    series = _synthetic_series(100, "2010-01-01")
    start = pd.Timestamp("2010-02-01")
    end = pd.Timestamp("2010-06-01")
    step_days = 30
    
    eval_dates = _generate_eval_dates(series, start, end, step_days)
    
    assert len(eval_dates) > 0
    assert all(isinstance(d, pd.Timestamp) for d in eval_dates)
    assert eval_dates[0] >= start
    assert eval_dates[-1] <= end
    assert eval_dates == sorted(eval_dates)


def test_prepare_and_validate_success():
    """Test successful data preparation."""
    series = _synthetic_series(400, "2010-01-01")
    
    def fake_from_instrument(instrument_name, start_date=None, end_date=None, column="Close"):
        return series
    
    with patch("core.data.preparation.DataLoader.from_instrument", fake_from_instrument):
        result = prepare_and_validate(
            instruments=["test_inst"],
            start_date="2010-06-01",
            end_date="2011-06-01",
            lookback_days=180,
            step_days=30,
            min_history_days=100,
            column="Close",
        )
    
    assert result is not None
    assert len(result.eval_dates) > 0
    assert result.start_date is not None
    assert result.end_date is not None
    assert result.load_start is not None
    assert result.instruments == ["test_inst"]


def test_prepare_and_validate_multi_instrument():
    """Test data prep with multiple instruments."""
    series1 = _synthetic_series(400, "2010-01-01")
    series2 = _synthetic_series(400, "2010-01-01", base=200.0)
    
    data_by_inst = {"inst1": series1, "inst2": series2}
    
    def fake_from_instrument(instrument_name, start_date=None, end_date=None, column="Close"):
        return data_by_inst[instrument_name]
    
    with patch("core.data.preparation.DataLoader.from_instrument", fake_from_instrument):
        result = prepare_and_validate(
            instruments=["inst1", "inst2"],
            start_date="2010-06-01",
            end_date="2011-06-01",
            lookback_days=180,
            step_days=30,
            min_history_days=100,
            column="Close",
        )
    
    assert len(result.eval_dates) > 0
    assert result.instruments == ["inst1", "inst2"]


def test_prepare_and_validate_skips_missing_instrument():
    """When one instrument has data and one doesn't, result contains only the one with data."""
    series = _synthetic_series(400, "2010-01-01")
    call_count = [0]

    def fake_from_instrument(instrument_name, start_date=None, end_date=None, column="Close"):
        call_count[0] += 1
        if instrument_name == "missing_inst":
            raise FileNotFoundError("No data")
        return series

    with patch("core.data.preparation.DataLoader.from_instrument", fake_from_instrument):
        result = prepare_and_validate(
            instruments=["inst_ok", "missing_inst", "inst_ok2"],
            start_date="2010-06-01",
            end_date="2011-06-01",
            lookback_days=180,
            step_days=30,
            min_history_days=100,
            column="Close",
        )
    assert result.instruments == ["inst_ok", "inst_ok2"]
    assert len(result.eval_dates) > 0


def test_prepare_and_validate_missing_data():
    """Test that data prep raises when all instruments have no data (skip leaves none)."""
    def fake_from_instrument(instrument_name, start_date=None, end_date=None, column="Close"):
        raise FileNotFoundError(f"No data for {instrument_name}")
    
    with patch("core.data.preparation.DataLoader.from_instrument", fake_from_instrument):
        with pytest.raises(DataPreparationError, match="No instruments have data"):
            prepare_and_validate(
                instruments=["missing_inst"],
                start_date="2010-06-01",
                end_date="2011-06-01",
                lookback_days=180,
                step_days=30,
                min_history_days=100,
                column="Close",
            )


def test_prepare_and_validate_insufficient_history():
    """Test that data prep raises when all instruments have insufficient history (skip leaves none)."""
    # Very short series - not enough history
    short_series = _synthetic_series(50, "2010-05-01")
    
    def fake_from_instrument(instrument_name, start_date=None, end_date=None, column="Close"):
        return short_series
    
    with patch("core.data.preparation.DataLoader.from_instrument", fake_from_instrument):
        with pytest.raises(DataPreparationError, match="No instruments have data"):
            prepare_and_validate(
                instruments=["short_inst"],
                start_date="2010-06-01",
                end_date="2011-06-01",
                lookback_days=180,
                step_days=30,
                min_history_days=100,
                column="Close",
            )
