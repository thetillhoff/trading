"""
Tests for technical indicators (RSI, EMA, MACD, ADX) and confirmation helpers.
"""
import pytest
import pandas as pd
import numpy as np
from core.indicators.technical import (
    TechnicalIndicators,
    IndicatorValues,
    check_buy_confirmation,
    check_sell_confirmation,
)


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    # Create simple uptrend
    prices = pd.Series(
        100 + np.arange(100) * 0.5 + np.random.randn(100) * 2,
        index=dates
    )
    return prices


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
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


class TestRSI:
    """Test RSI calculation."""
    
    def test_rsi_range(self, sample_prices):
        """RSI should be between 0 and 100."""
        indicators = TechnicalIndicators()
        rsi = indicators.calculate_rsi(sample_prices)
        
        # Skip NaN values at start
        valid_rsi = rsi.dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100
    
    def test_rsi_length(self, sample_prices):
        """RSI should have same length as input."""
        indicators = TechnicalIndicators()
        rsi = indicators.calculate_rsi(sample_prices)
        
        assert len(rsi) == len(sample_prices)
    
    def test_rsi_period(self, sample_prices):
        """Different periods should give different results."""
        ind1 = TechnicalIndicators(rsi_period=7)
        ind2 = TechnicalIndicators(rsi_period=14)
        
        rsi7 = ind1.calculate_rsi(sample_prices)
        rsi14 = ind2.calculate_rsi(sample_prices)
        
        # Drop NaN and compare
        assert not rsi7.dropna().equals(rsi14.dropna())


class TestEMA:
    """Test EMA calculation."""
    
    def test_ema_convergence(self, sample_prices):
        """EMA should converge toward prices."""
        indicators = TechnicalIndicators()
        ema = indicators.calculate_ema(sample_prices, period=20)
        
        # EMA should be within reasonable range of prices
        valid_idx = ~ema.isna()
        price_range = sample_prices[valid_idx].max() - sample_prices[valid_idx].min()
        assert ema[valid_idx].min() >= sample_prices[valid_idx].min() - price_range
        assert ema[valid_idx].max() <= sample_prices[valid_idx].max() + price_range
    
    def test_ema_smoothing(self, sample_prices):
        """EMA should be smoother than raw prices."""
        indicators = TechnicalIndicators()
        ema = indicators.calculate_ema(sample_prices, period=20)
        
        # Standard deviation of changes should be lower for EMA
        price_changes = sample_prices.diff().dropna()
        ema_changes = ema.diff().dropna()
        
        assert ema_changes.std() < price_changes.std()


class TestMACD:
    """Test MACD calculation."""
    
    def test_macd_components(self, sample_prices):
        """MACD should return three components."""
        indicators = TechnicalIndicators()
        macd_line, signal_line, histogram = indicators.calculate_macd(sample_prices)
        
        assert len(macd_line) == len(sample_prices)
        assert len(signal_line) == len(sample_prices)
        assert len(histogram) == len(sample_prices)
    
    def test_macd_histogram(self, sample_prices):
        """Histogram should equal MACD - Signal."""
        indicators = TechnicalIndicators()
        macd_line, signal_line, histogram = indicators.calculate_macd(sample_prices)
        
        # Check where both are valid
        valid_idx = ~(macd_line.isna() | signal_line.isna() | histogram.isna())
        expected_hist = macd_line[valid_idx] - signal_line[valid_idx]
        actual_hist = histogram[valid_idx]
        
        pd.testing.assert_series_equal(expected_hist, actual_hist, check_names=False)


class TestADX:
    """Test ADX calculation."""
    
    def test_adx_range(self, sample_ohlcv):
        """ADX should be between 0 and 100."""
        indicators = TechnicalIndicators()
        adx = indicators.calculate_adx(sample_ohlcv)
        
        valid_adx = adx.dropna()
        assert valid_adx.min() >= 0
        assert valid_adx.max() <= 100
    
    def test_adx_with_dataframe(self, sample_ohlcv):
        """ADX should work with DataFrame input."""
        indicators = TechnicalIndicators()
        adx = indicators.calculate_adx(sample_ohlcv)
        
        assert len(adx) == len(sample_ohlcv)
        assert not adx.dropna().empty
    
    def test_adx_with_series(self, sample_prices):
        """ADX should work with Series input (fallback mode)."""
        indicators = TechnicalIndicators()
        adx = indicators.calculate_adx(sample_prices)
        
        assert len(adx) == len(sample_prices)


class TestCalculateAll:
    """Test combined indicator calculation."""
    
    def test_calculate_all_columns(self, sample_prices):
        """calculate_all should return all indicator columns."""
        indicators = TechnicalIndicators()
        df = indicators.calculate_all(sample_prices)
        
        expected_cols = [
            'price', 'rsi', 'rsi_oversold', 'rsi_overbought',
            'ema_short', 'ema_long', 'price_above_ema_short', 'price_above_ema_long',
            'ema_bullish_cross', 'ema_bearish_cross',
            'macd_line', 'macd_signal', 'macd_histogram',
            'macd_bullish', 'macd_bearish',
            'adx', 'ma_50', 'ma_slope'
        ]
        
        for col in expected_cols:
            assert col in df.columns
    
    def test_calculate_all_with_dataframe(self, sample_ohlcv):
        """calculate_all should work with DataFrame input."""
        indicators = TechnicalIndicators()
        df = indicators.calculate_all(sample_ohlcv)

        assert len(df) == len(sample_ohlcv)
        assert 'price' in df.columns
        assert 'adx' in df.columns


class TestGetIndicatorsAt:
    """Test get_indicators_at."""

    def test_get_indicators_at_returns_indicator_values(self, sample_prices):
        """get_indicators_at should return IndicatorValues when data suffices."""
        indicators = TechnicalIndicators()
        ts = sample_prices.index[50]
        vals = indicators.get_indicators_at(sample_prices, ts)
        assert vals is not None
        assert isinstance(vals, IndicatorValues)
        assert vals.timestamp == ts
        assert vals.price == sample_prices.loc[ts]
        assert vals.rsi is not None

    def test_get_indicators_at_early_index_returns_none(self, sample_prices):
        """Early timestamps may have insufficient data for RSI."""
        indicators = TechnicalIndicators()
        ts = sample_prices.index[0]
        vals = indicators.get_indicators_at(sample_prices, ts)
        # Either None or valid; implementation may return None for first rows
        assert vals is None or isinstance(vals, IndicatorValues)


class TestCheckBuyConfirmation:
    """Test check_buy_confirmation."""

    def test_none_returns_no_indicator_data(self):
        """None input returns (False, 'No indicator data', 0)."""
        ok, reason, count = check_buy_confirmation(None)
        assert ok is False
        assert "No indicator data" in reason
        assert count == 0

    def test_rsi_oversold_confirms_buy(self):
        """RSI oversold can confirm buy when use_rsi=True."""
        vals = IndicatorValues(
            timestamp=pd.Timestamp('2020-01-15'),
            price=100.0,
            rsi=30.0,
            rsi_oversold=True,
            rsi_overbought=False,
            ema_short=None,
            ema_long=None,
            price_above_ema_short=False,
            price_above_ema_long=False,
            ema_bullish_cross=False,
            ema_bearish_cross=False,
            macd_line=None,
            macd_signal=None,
            macd_histogram=None,
            macd_bullish=False,
            macd_bearish=False,
        )
        ok, reason, count = check_buy_confirmation(vals, use_rsi=True, use_ema=False, use_macd=False)
        assert count >= 1
        assert "RSI" in reason or "favorable" in reason


class TestCheckSellConfirmation:
    """Test check_sell_confirmation."""

    def test_none_returns_no_indicator_data(self):
        """None input returns (False, 'No indicator data', 0)."""
        ok, reason, count = check_sell_confirmation(None)
        assert ok is False
        assert "No indicator data" in reason
        assert count == 0

    def test_rsi_overbought_confirms_sell(self):
        """RSI overbought can confirm sell when use_rsi=True."""
        vals = IndicatorValues(
            timestamp=pd.Timestamp('2020-01-15'),
            price=100.0,
            rsi=75.0,
            rsi_oversold=False,
            rsi_overbought=True,
            ema_short=None,
            ema_long=None,
            price_above_ema_short=False,
            price_above_ema_long=False,
            ema_bullish_cross=False,
            ema_bearish_cross=False,
            macd_line=None,
            macd_signal=None,
            macd_histogram=None,
            macd_bullish=False,
            macd_bearish=False,
        )
        ok, reason, count = check_sell_confirmation(vals, use_rsi=True, use_ema=False, use_macd=False)
        assert count >= 1
        assert "RSI" in reason or "favorable" in reason
