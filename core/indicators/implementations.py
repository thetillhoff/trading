"""
Individual indicator implementations following the Indicator interface.

These classes provide a uniform interface for all technical indicators,
making it easier to add new indicators without modifying existing code.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from .base import Indicator
from ..shared.defaults import (
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    EMA_SHORT_PERIOD, EMA_LONG_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
)


class RSIIndicator(Indicator):
    """Relative Strength Index indicator."""
    
    def __init__(self, period: int = RSI_PERIOD):
        self.period = period
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI values."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(span=self.period, min_periods=self.period).mean()
        avg_loss = loss.ewm(span=self.period, min_periods=self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_value_at(self, prices: pd.Series, timestamp: pd.Timestamp) -> Optional[float]:
        """Get RSI value at specific timestamp."""
        rsi = self.calculate(prices)
        if timestamp in rsi.index:
            val = rsi[timestamp]
            return None if pd.isna(val) else val
        return None


class EMAIndicator(Indicator):
    """Exponential Moving Average indicator."""
    
    def __init__(self, period: int = EMA_SHORT_PERIOD):
        self.period = period
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """Calculate EMA values."""
        return prices.ewm(span=self.period, min_periods=self.period).mean()
    
    def get_value_at(self, prices: pd.Series, timestamp: pd.Timestamp) -> Optional[float]:
        """Get EMA value at specific timestamp."""
        ema = self.calculate(prices)
        if timestamp in ema.index:
            val = ema[timestamp]
            return None if pd.isna(val) else val
        return None


class MACDIndicator(Indicator):
    """MACD (Moving Average Convergence Divergence) indicator."""
    
    def __init__(
        self,
        fast: int = MACD_FAST,
        slow: int = MACD_SLOW,
        signal: int = MACD_SIGNAL
    ):
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """
        Calculate MACD histogram (MACD line - Signal line).
        
        Returns histogram as it's the most commonly used MACD value.
        """
        macd_line, signal_line, histogram = self.calculate_components(prices)
        return histogram
    
    def calculate_components(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate all MACD components: line, signal, histogram."""
        ema_fast = prices.ewm(span=self.fast, min_periods=self.fast).mean()
        ema_slow = prices.ewm(span=self.slow, min_periods=self.slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, min_periods=self.signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def get_value_at(self, prices: pd.Series, timestamp: pd.Timestamp) -> Optional[float]:
        """Get MACD histogram value at specific timestamp."""
        histogram = self.calculate(prices)
        if timestamp in histogram.index:
            val = histogram[timestamp]
            return None if pd.isna(val) else val
        return None


class ADXIndicator(Indicator):
    """ADX (Average Directional Index) for trend strength."""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.Series) -> pd.Series:
        """
        Calculate ADX from Close prices (simplified).
        
        Note: This is a fallback for when only Close is available.
        For accurate ADX, use calculate_from_ohlc() with High/Low/Close.
        """
        high = data
        low = data
        
        tr = high.rolling(2).max() - low.rolling(2).min()
        tr = tr.fillna(0)
        
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)
        
        plus_dm[up_move > down_move] = up_move[up_move > down_move].clip(lower=0)
        minus_dm[down_move > up_move] = down_move[down_move > up_move].clip(lower=0)
        
        atr = tr.rolling(self.period).mean()
        plus_di = 100 * (plus_dm.rolling(self.period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(self.period).mean() / atr.replace(0, np.nan))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(self.period).mean()
        
        return adx.fillna(0)
    
    def calculate_from_ohlc(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ADX from High/Low/Close (accurate method)."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0.0, index=close.index)
        minus_dm = pd.Series(0.0, index=close.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
        
        atr = tr.rolling(self.period).mean()
        plus_di = 100 * (plus_dm.rolling(self.period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(self.period).mean() / atr.replace(0, np.nan))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(self.period).mean()
        
        return adx.fillna(0)
    
    def get_value_at(self, prices: pd.Series, timestamp: pd.Timestamp) -> Optional[float]:
        """Get ADX value at specific timestamp."""
        adx = self.calculate(prices)
        if timestamp in adx.index:
            val = adx[timestamp]
            return None if pd.isna(val) else val
        return None


# Export all indicator classes
__all__ = ['RSIIndicator', 'EMAIndicator', 'MACDIndicator', 'ADXIndicator']
