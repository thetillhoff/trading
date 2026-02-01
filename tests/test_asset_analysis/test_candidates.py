"""Tests for asset_analysis.candidates (liquidity, volatility, correlation diversity)."""
import pandas as pd
import pytest

from core.asset_analysis.candidates import score_candidates


def _sample_df(days=30, start="2020-01-01", volume=1_000_000):
    idx = pd.date_range(start=start, periods=days, freq="D")
    return pd.DataFrame(
        {"Close": 100 + pd.Series(range(days), index=idx).cumsum(), "Volume": volume},
        index=idx,
    )


class TestScoreCandidates:
    """Test score_candidates output shape and columns."""

    def test_empty_input_returns_empty_dataframe(self):
        out = score_candidates({})
        assert out.empty
        assert list(out.columns) == [
            "instrument",
            "name",
            "sector",
            "industry",
            "liquidity_score",
            "volatility_score",
            "correlation_diversity_score",
            "composite_score",
        ]

    def test_output_columns_present(self):
        df = _sample_df(25)
        out = score_candidates({"a": df})
        assert "instrument" in out.columns
        assert "name" in out.columns
        assert "sector" in out.columns
        assert "industry" in out.columns
        assert "liquidity_score" in out.columns
        assert "volatility_score" in out.columns
        assert "correlation_diversity_score" in out.columns
        assert "composite_score" in out.columns
        assert len(out) == 1
        assert out["instrument"].iloc[0] == "a"

    def test_sector_industry_from_metadata(self):
        df = _sample_df(25)
        metadata = {"a": {"sector": "Technology", "industry": "Software"}}
        out = score_candidates({"a": df}, metadata=metadata)
        assert out["sector"].iloc[0] == "Technology"
        assert out["industry"].iloc[0] == "Software"

    def test_name_from_metadata(self):
        df = _sample_df(25)
        metadata = {"a": {"long_name": "Apple Inc.", "short_name": "Apple"}}
        out = score_candidates({"a": df}, metadata=metadata)
        assert out["name"].iloc[0] == "Apple Inc."
        metadata2 = {"b": {"short_name": "Microsoft"}}  # no long_name
        out2 = score_candidates({"b": df}, metadata=metadata2)
        assert out2["name"].iloc[0] == "Microsoft"

    def test_higher_volume_gives_higher_liquidity_score(self):
        df_low = _sample_df(25, volume=100)
        df_high = _sample_df(25, volume=10_000_000)
        out_low = score_candidates({"a": df_low})
        out_high = score_candidates({"b": df_high})
        # When only one instrument, normalized liquidity is 0.5 (constant). So we need two.
        out_both = score_candidates({"low": df_low, "high": df_high})
        low_liq = out_both[out_both["instrument"] == "low"]["liquidity_score"].iloc[0]
        high_liq = out_both[out_both["instrument"] == "high"]["liquidity_score"].iloc[0]
        assert high_liq >= low_liq

    def test_composite_score_bounded(self):
        df = _sample_df(25)
        out = score_candidates({"a": df})
        assert 0 <= out["composite_score"].iloc[0] <= 1.0
