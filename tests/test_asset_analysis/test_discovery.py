"""Tests for asset_analysis.discovery (sources, deduping by exact match)."""
from unittest.mock import patch

import pytest

from core.asset_analysis.discovery import (
    get_available_assets_all_sources,
    get_all_sources,
)


class TestGetAvailableAssetsAllSources:
    def test_dedupes_exact_match_only(self):
        # One list per source (sp500, nasdaq100, dax, djia); BRK.B and BRK-B both kept
        with patch(
            "core.asset_analysis.discovery.get_available_assets",
            side_effect=[
                ["BRK.B", "AAPL"],
                ["BRK-B", "MSFT"],
                [],
                [],
            ],
        ):
            result = get_available_assets_all_sources(refresh=False)
        assert set(result) == {"BRK.B", "AAPL", "BRK-B", "MSFT"}
        assert len(result) == 4

    def test_returns_all_sources_union(self):
        with patch(
            "core.asset_analysis.discovery.get_available_assets",
            side_effect=[["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]],
        ):
            result = get_available_assets_all_sources(refresh=False)
        assert set(result) == {"A", "B", "C", "D"}
        assert len(result) == 4

    def test_all_sources_called(self):
        with patch(
            "core.asset_analysis.discovery.get_available_assets",
            return_value=[],
        ) as m:
            get_available_assets_all_sources(refresh=False)
        assert m.call_count == len(get_all_sources())
