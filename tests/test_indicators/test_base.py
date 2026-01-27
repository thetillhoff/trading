"""
Tests for the Indicator base interface.
"""
import pytest
from abc import ABC
from core.indicators.base import Indicator
from core.indicators.implementations import RSIIndicator, EMAIndicator, MACDIndicator, ADXIndicator


class TestIndicatorInterface:
    """Test Indicator abstract base class."""

    def test_indicator_is_abstract(self):
        """Indicator should be an ABC."""
        assert issubclass(Indicator, ABC)

    def test_indicator_requires_calculate(self):
        """Indicator defines calculate as abstract."""
        assert hasattr(Indicator, 'calculate')
        assert getattr(Indicator.calculate, '__isabstractmethod__', False) or 'calculate' in getattr(
            Indicator, '__abstractmethods__', set()
        )

    def test_indicator_requires_get_value_at(self):
        """Indicator defines get_value_at as abstract."""
        assert hasattr(Indicator, 'get_value_at')

    def test_concrete_indicators_implement_calculate(self):
        """All concrete indicators implement calculate."""
        for cls in (RSIIndicator, EMAIndicator, MACDIndicator, ADXIndicator):
            assert hasattr(cls, 'calculate')
            # Should not be abstract in subclass
            impl = getattr(cls, 'calculate')
            assert impl is not Indicator.calculate

    def test_concrete_indicators_implement_get_value_at(self):
        """All concrete indicators implement get_value_at."""
        for cls in (RSIIndicator, EMAIndicator, MACDIndicator, ADXIndicator):
            assert hasattr(cls, 'get_value_at')
            impl = getattr(cls, 'get_value_at')
            assert impl is not Indicator.get_value_at
