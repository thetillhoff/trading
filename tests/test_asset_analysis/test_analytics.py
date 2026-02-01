"""Tests for asset_analysis.analytics (returns, volatility, correlation)."""
import pandas as pd
import pytest

from core.asset_analysis.analytics import (
    compute_returns,
    compute_volatility_summary,
    compute_correlation_matrix,
)


def _sample_df(days=10, start="2020-01-01"):
    idx = pd.date_range(start=start, periods=days, freq="D")
    return pd.DataFrame(
        {"Close": 100 + pd.Series(range(days), index=idx).cumsum(), "Volume": 1_000_000},
        index=idx,
    )


class TestComputeReturns:
    """Test compute_returns."""

    def test_empty_input_returns_empty_dataframe(self):
        assert compute_returns({}).empty

    def test_single_instrument_returns_series(self):
        df = _sample_df(5)
        ret = compute_returns({"a": df})
        assert ret.shape[1] == 1
        assert ret.columns[0] == "a"
        assert len(ret) == 4  # first row is NaN for pct_change

    def test_returns_bounds(self):
        df = _sample_df(20)
        ret = compute_returns({"a": df})
        assert ret["a"].min() >= -1.0 or ret["a"].max() <= 10.0  # sanity


class TestComputeVolatilitySummary:
    """Test compute_volatility_summary."""

    def test_empty_input_returns_empty_dataframe(self):
        out = compute_volatility_summary({})
        assert out.empty
        assert "annualized_vol" in out.columns or out.columns.empty

    def test_single_instrument_has_row(self):
        df = _sample_df(30)
        out = compute_volatility_summary({"a": df}, window_days=5)
        assert len(out) == 1
        assert out["instrument"].iloc[0] == "a"
        assert "annualized_vol" in out.columns
        assert "avg_daily_return" in out.columns
        assert "rolling_vol" in out.columns


class TestComputeCorrelationMatrix:
    """Test compute_correlation_matrix."""

    def test_empty_input_returns_empty_dataframe(self):
        assert compute_correlation_matrix({}).empty

    def test_single_instrument_returns_empty(self):
        df = _sample_df(10)
        out = compute_correlation_matrix({"a": df})
        assert out.empty  # need at least 2 for correlation

    def test_two_instruments_returns_2x2(self):
        df1 = _sample_df(20, start="2020-01-01")
        df2 = _sample_df(20, start="2020-01-01")
        df2["Close"] = 200 - df2["Close"]  # negative correlation
        out = compute_correlation_matrix({"a": df1, "b": df2})
        assert out.shape == (2, 2)
        assert out.loc["a", "a"] == pytest.approx(1.0)
        assert out.loc["b", "b"] == pytest.approx(1.0)
        assert -1 <= out.loc["a", "b"] <= 1
