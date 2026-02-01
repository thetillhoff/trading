"""Tests for asset_analysis.filters (ASSET_CATEGORIES excluded list)."""
import pandas as pd
import pytest

from core.asset_analysis.filters import (
    EXCLUDED_KEYWORDS,
    filter_candidates_by_asset_categories,
)


class TestFilterCandidatesByAssetCategories:
    def test_empty_passthrough(self):
        df = pd.DataFrame(columns=["instrument", "sector", "industry"])
        out = filter_candidates_by_asset_categories(df)
        assert out.empty

    def test_excludes_oil_gas(self):
        df = pd.DataFrame([
            {"instrument": "XOM", "sector": "Energy", "industry": "Oil & Gas E&P"},
            {"instrument": "AAPL", "sector": "Technology", "industry": "Consumer Electronics"},
        ])
        out = filter_candidates_by_asset_categories(df)
        assert len(out) == 1
        assert out["instrument"].iloc[0] == "AAPL"

    def test_excludes_real_estate_reit(self):
        df = pd.DataFrame([
            {"instrument": "O", "sector": "Real Estate", "industry": "REIT—Diversified"},
            {"instrument": "MSFT", "sector": "Technology", "industry": "Software"},
        ])
        out = filter_candidates_by_asset_categories(df)
        assert len(out) == 1
        assert out["instrument"].iloc[0] == "MSFT"

    def test_keeps_nan_sector_industry(self):
        df = pd.DataFrame([
            {"instrument": "A", "sector": None, "industry": None},
        ])
        out = filter_candidates_by_asset_categories(df)
        assert len(out) == 1

    def test_excluded_keywords_non_empty(self):
        assert len(EXCLUDED_KEYWORDS) > 0
        assert "oil & gas" in EXCLUDED_KEYWORDS
        assert "real estate" in EXCLUDED_KEYWORDS
