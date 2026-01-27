"""
Tests for individual indicator implementations.
"""
import pytest
import pandas as pd
import numpy as np
from core.indicators.implementations import RSIIndicator, EMAIndicator, MACDIndicator, ADXIndicator
from core.indicators.base import Indicator


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(
        100 + np.arange(100) * 0.5 + np.random.randn(100) * 2,
        index=dates
    )
    return prices


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    base = 100 + np.arange(100) * 0.5
    noise = np.random.randn(100) * 2
    df = pd.DataFrame({
        'Open': base + noise,
        'High': base + abs(noise) + 1,
        'Low': base - abs(noise) - 1,
        'Close': base + noise * 0.5,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    return df


class TestIndicatorInterface:
    """Test that all indicators implement the Indicator interface."""
    
    def test_rsi_implements_interface(self):
        """RSIIndicator should implement Indicator interface."""
        assert issubclass(RSIIndicator, Indicator)
    
    def test_ema_implements_interface(self):
        """EMAIndicator should implement Indicator interface."""
        assert issubclass(EMAIndicator, Indicator)
    
    def test_macd_implements_interface(self):
        """MACDIndicator should implement Indicator interface."""
        assert issubclass(MACDIndicator, Indicator)
    
    def test_adx_implements_interface(self):
        """ADXIndicator should implement Indicator interface."""
        assert issubclass(ADXIndicator, Indicator)


class TestRSIIndicator:
    """Test RSI indicator implementation."""
    
    def test_calculate(self, sample_prices):
        """RSI should calculate values."""
        rsi = RSIIndicator(period=14)
        values = rsi.calculate(sample_prices)
        
        assert len(values) == len(sample_prices)
        valid = values.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100
    
    def test_get_value_at(self, sample_prices):
        """Should get value at specific timestamp."""
        rsi = RSIIndicator(period=14)
        timestamp = sample_prices.index[50]
        value = rsi.get_value_at(sample_prices, timestamp)
        
        assert value is not None
        assert 0 <= value <= 100


class TestEMAIndicator:
    """Test EMA indicator implementation."""
    
    def test_calculate(self, sample_prices):
        """EMA should calculate values."""
        ema = EMAIndicator(period=20)
        values = ema.calculate(sample_prices)
        
        assert len(values) == len(sample_prices)
    
    def test_different_periods(self, sample_prices):
        """Different periods should give different values."""
        ema1 = EMAIndicator(period=10)
        ema2 = EMAIndicator(period=20)
        
        values1 = ema1.calculate(sample_prices)
        values2 = ema2.calculate(sample_prices)
        
        assert not values1.dropna().equals(values2.dropna())


class TestMACDIndicator:
    """Test MACD indicator implementation."""

    def test_calculate(self, sample_prices):
        """MACD should calculate histogram."""
        macd = MACDIndicator()
        values = macd.calculate(sample_prices)

        assert len(values) == len(sample_prices)

    def test_calculate_components(self, sample_prices):
        """Should return all three components."""
        macd = MACDIndicator()
        line, signal, histogram = macd.calculate_components(sample_prices)

        assert len(line) == len(sample_prices)
        assert len(signal) == len(sample_prices)
        assert len(histogram) == len(sample_prices)

    def test_get_value_at(self, sample_prices):
        """Should get histogram value at specific timestamp."""
        macd = MACDIndicator()
        timestamp = sample_prices.index[50]
        value = macd.get_value_at(sample_prices, timestamp)
        assert value is not None or pd.isna(macd.calculate(sample_prices).loc[timestamp])


class TestADXIndicator:
    """Test ADX indicator implementation."""

    def test_calculate(self, sample_prices):
        """ADX should calculate values from price series."""
        adx = ADXIndicator(period=14)
        values = adx.calculate(sample_prices)
        assert len(values) == len(sample_prices)
        valid = values.dropna()
        if len(valid) > 0:
            assert valid.min() >= 0
            assert valid.max() <= 100

    def test_get_value_at(self, sample_prices):
        """Should get value at specific timestamp."""
        adx = ADXIndicator(period=14)
        timestamp = sample_prices.index[50]
        value = adx.get_value_at(sample_prices, timestamp)
        assert value is not None or pd.isna(adx.calculate(sample_prices).loc[timestamp])

    def test_calculate_from_ohlc(self, sample_ohlcv):
        """ADX from OHLC should return series matching length of input."""
        adx = ADXIndicator(period=14)
        high = sample_ohlcv['High']
        low = sample_ohlcv['Low']
        close = sample_ohlcv['Close']
        values = adx.calculate_from_ohlc(high, low, close)
        assert len(values) == len(sample_ohlcv)
        valid = values.dropna()
        if len(valid) > 0:
            assert valid.min() >= 0
            assert valid.max() <= 100
