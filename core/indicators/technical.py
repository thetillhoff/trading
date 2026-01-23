"""
Technical indicators for trading signal confirmation.

Provides RSI, EMA, MACD, and other indicators that can be used
to confirm or filter trading signals.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add paths for imports
core_dir = Path(__file__).parent.parent.parent
project_root = core_dir.parent
sys.path.insert(0, str(project_root))

# Import centralized defaults (single source of truth)
from core.shared.defaults import (
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    EMA_SHORT_PERIOD, EMA_LONG_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
)


@dataclass
class IndicatorValues:
    """Container for indicator values at a specific point in time."""
    timestamp: pd.Timestamp
    price: float
    
    # RSI
    rsi: Optional[float] = None
    rsi_oversold: bool = False  # RSI < oversold threshold
    rsi_overbought: bool = False  # RSI > overbought threshold
    
    # Moving Averages
    ema_short: Optional[float] = None  # Short-period EMA
    ema_long: Optional[float] = None  # Long-period EMA
    price_above_ema_short: bool = False
    price_above_ema_long: bool = False
    ema_bullish_cross: bool = False  # Short EMA crossed above long EMA
    ema_bearish_cross: bool = False  # Short EMA crossed below long EMA
    
    # MACD
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_bullish: bool = False  # Histogram > 0 or turning positive
    macd_bearish: bool = False  # Histogram < 0 or turning negative


class TechnicalIndicators:
    """Calculates technical indicators from price data."""
    
    def __init__(
        self,
        rsi_period: int = RSI_PERIOD,  # From shared.defaults
        rsi_oversold: int = RSI_OVERSOLD,  # From shared.defaults
        rsi_overbought: int = RSI_OVERBOUGHT,  # From shared.defaults
        ema_short_period: int = EMA_SHORT_PERIOD,  # From shared.defaults
        ema_long_period: int = EMA_LONG_PERIOD,  # From shared.defaults
        macd_fast: int = MACD_FAST,  # From shared.defaults
        macd_slow: int = MACD_SLOW,  # From shared.defaults
        macd_signal: int = MACD_SIGNAL,  # From shared.defaults
    ):
        """
        Initialize indicator calculator.
        
        Args:
            rsi_period: Period for RSI calculation (default: from shared.defaults.RSI_PERIOD)
            rsi_oversold: RSI level below which is oversold (default: from shared.defaults.RSI_OVERSOLD)
            rsi_overbought: RSI level above which is overbought (default: from shared.defaults.RSI_OVERBOUGHT)
            ema_short_period: Short EMA period (default: from shared.defaults.EMA_SHORT_PERIOD)
            ema_long_period: Long EMA period (default: from shared.defaults.EMA_LONG_PERIOD)
            macd_fast: MACD fast period (default: from shared.defaults.MACD_FAST)
            macd_slow: MACD slow period (default: from shared.defaults.MACD_SLOW)
            macd_signal: MACD signal period (default: from shared.defaults.MACD_SIGNAL)
        """
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        # Use exponential moving average for smoothing
        avg_gain = gain.ewm(span=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(span=self.rsi_period, min_periods=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, min_periods=period).mean()
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = prices.ewm(span=self.macd_fast, min_periods=self.macd_fast).mean()
        ema_slow = prices.ewm(span=self.macd_slow, min_periods=self.macd_slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, min_periods=self.macd_signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_adx(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate ADX (Average Directional Index) for trend strength detection.
        
        ADX measures trend strength (0-100):
        - ADX > 30: Strong trend
        - ADX 20-30: Moderate trend
        - ADX < 20: Weak/no trend
        
        Args:
            prices: Price series
            period: Period for ADX calculation (default: 14)
        
        Returns:
            Series with ADX values
        """
        # Simplified ADX using price-based approximation
        # True ADX requires High/Low/Close, but we only have Close prices
        high = prices
        low = prices
        
        # True Range (simplified)
        tr = high.rolling(2).max() - low.rolling(2).min()
        tr = tr.fillna(0)
        
        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()
        
        # Initialize directional movement series
        plus_dm = pd.Series(0.0, index=prices.index)
        minus_dm = pd.Series(0.0, index=prices.index)
        
        # Positive directional movement
        plus_dm[up_move > down_move] = up_move[up_move > down_move].clip(lower=0)
        # Negative directional movement
        minus_dm[down_move > up_move] = down_move[down_move > up_move].clip(lower=0)
        
        # Smoothed averages
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        
        # Directional Index (DX)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        
        # ADX (smoothed DX)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(0)
    
    def calculate_all(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate all indicators and return as DataFrame.
        
        Args:
            prices: Price series with datetime index
            
        Returns:
            DataFrame with all indicator values
        """
        df = pd.DataFrame(index=prices.index)
        df['price'] = prices
        
        # RSI
        df['rsi'] = self.calculate_rsi(prices)
        df['rsi_oversold'] = df['rsi'] < self.rsi_oversold
        df['rsi_overbought'] = df['rsi'] > self.rsi_overbought
        
        # EMAs
        df['ema_short'] = self.calculate_ema(prices, self.ema_short_period)
        df['ema_long'] = self.calculate_ema(prices, self.ema_long_period)
        df['price_above_ema_short'] = prices > df['ema_short']
        df['price_above_ema_long'] = prices > df['ema_long']
        
        # EMA crossovers
        ema_diff = df['ema_short'] - df['ema_long']
        ema_diff_prev = ema_diff.shift(1)
        df['ema_bullish_cross'] = (ema_diff > 0) & (ema_diff_prev <= 0)
        df['ema_bearish_cross'] = (ema_diff < 0) & (ema_diff_prev >= 0)
        
        # MACD
        macd_line, signal_line, histogram = self.calculate_macd(prices)
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # MACD conditions
        histogram_prev = histogram.shift(1)
        df['macd_bullish'] = (histogram > 0) | ((histogram > histogram_prev) & (histogram_prev < 0))
        df['macd_bearish'] = (histogram < 0) | ((histogram < histogram_prev) & (histogram_prev > 0))
        
        # ADX (for regime detection - trend strength)
        df['adx'] = self.calculate_adx(prices)
        
        # 50-day MA and slope (for regime detection - trend direction)
        df['ma_50'] = prices.rolling(50).mean()
        df['ma_slope'] = df['ma_50'].diff(5)  # 5-day slope for smoothness
        
        return df
    
    def get_indicators_at(
        self,
        prices: pd.Series,
        timestamp: pd.Timestamp
    ) -> Optional[IndicatorValues]:
        """
        Get indicator values at a specific timestamp.
        
        Args:
            prices: Price series (must include data before timestamp for calculation)
            timestamp: Timestamp to get indicators for
            
        Returns:
            IndicatorValues at the timestamp, or None if insufficient data
        """
        if timestamp not in prices.index:
            # Find nearest timestamp
            idx = prices.index.get_indexer([timestamp], method='ffill')[0]
            if idx < 0:
                return None
            timestamp = prices.index[idx]
        
        # Calculate all indicators
        df = self.calculate_all(prices)
        
        if timestamp not in df.index:
            return None
        
        row = df.loc[timestamp]
        
        # Check for NaN values (insufficient data for calculation)
        if pd.isna(row['rsi']):
            return None
        
        return IndicatorValues(
            timestamp=timestamp,
            price=row['price'],
            rsi=row['rsi'],
            rsi_oversold=row['rsi_oversold'],
            rsi_overbought=row['rsi_overbought'],
            ema_short=row['ema_short'],
            ema_long=row['ema_long'],
            price_above_ema_short=row['price_above_ema_short'],
            price_above_ema_long=row['price_above_ema_long'],
            ema_bullish_cross=row['ema_bullish_cross'],
            ema_bearish_cross=row['ema_bearish_cross'],
            macd_line=row['macd_line'],
            macd_signal=row['macd_signal'],
            macd_histogram=row['macd_histogram'],
            macd_bullish=row['macd_bullish'],
            macd_bearish=row['macd_bearish'],
        )


def check_buy_confirmation(
    indicators: IndicatorValues,
    use_rsi: bool = True,
    use_ema: bool = True,
    use_macd: bool = True,
    require_all: bool = False,
) -> Tuple[bool, str, int]:
    """
    Check if indicators confirm a buy signal.
    
    Args:
        indicators: Current indicator values
        use_rsi: Check RSI confirmation
        use_ema: Check EMA confirmation
        use_macd: Check MACD confirmation
        require_all: If True, all enabled indicators must confirm.
                    If False, at least one must confirm.
    
    Returns:
        Tuple of (confirmed: bool, reason: str, confirmation_count: int)
        confirmation_count is the number of indicators that confirmed (0-3)
    """
    if indicators is None:
        return False, "No indicator data", 0
    
    confirmations = []
    reasons = []
    enabled_count = 0
    
    if use_rsi:
        enabled_count += 1
        if indicators.rsi is None:
            # Skip RSI check if not calculated (insufficient data)
            pass
        elif indicators.rsi_oversold or indicators.rsi < 40:
            confirmations.append(True)
            reasons.append(f"RSI={indicators.rsi:.0f} (favorable)")
        else:
            confirmations.append(False)
            reasons.append(f"RSI={indicators.rsi:.0f} (not oversold)")
    
    if use_ema:
        enabled_count += 1
        if indicators.ema_short is None:
            # Skip EMA check if not calculated
            pass
        elif indicators.price_above_ema_short or indicators.ema_bullish_cross:
            confirmations.append(True)
            reasons.append("EMA bullish")
        else:
            confirmations.append(False)
            reasons.append("EMA bearish")
    
    if use_macd:
        enabled_count += 1
        if indicators.macd_histogram is None:
            # Skip MACD check if not calculated
            pass
        elif indicators.macd_bullish:
            confirmations.append(True)
            reasons.append("MACD bullish")
        else:
            confirmations.append(False)
            reasons.append("MACD bearish")
    
    if not confirmations:
        # No indicators had enough data - skip confirmation
        return True, "Insufficient indicator data", 0
    
    if require_all:
        confirmed = all(confirmations)
    else:
        confirmed = any(confirmations)
    
    # Count how many indicators confirmed
    confirmation_count = sum(1 for c in confirmations if c)
    
    reason = " | ".join(reasons)
    return confirmed, reason, confirmation_count


def check_sell_confirmation(
    indicators: IndicatorValues,
    use_rsi: bool = True,
    use_ema: bool = True,
    use_macd: bool = True,
    require_all: bool = False,
) -> Tuple[bool, str, int]:
    """
    Check if indicators confirm a sell signal.
    
    Args:
        indicators: Current indicator values
        use_rsi: Check RSI confirmation
        use_ema: Check EMA confirmation
        use_macd: Check MACD confirmation
        require_all: If True, all enabled indicators must confirm.
                    If False, at least one must confirm.
    
    Returns:
        Tuple of (confirmed: bool, reason: str, confirmation_count: int)
        confirmation_count is the number of indicators that confirmed (0-3)
    """
    if indicators is None:
        return False, "No indicator data", 0
    
    confirmations = []
    reasons = []
    
    if use_rsi:
        if indicators.rsi is None:
            # Skip RSI check if not calculated
            pass
        elif indicators.rsi_overbought or indicators.rsi > 60:
            confirmations.append(True)
            reasons.append(f"RSI={indicators.rsi:.0f} (favorable)")
        else:
            confirmations.append(False)
            reasons.append(f"RSI={indicators.rsi:.0f} (not overbought)")
    
    if use_ema:
        if indicators.ema_short is None:
            # Skip EMA check if not calculated
            pass
        elif not indicators.price_above_ema_short or indicators.ema_bearish_cross:
            confirmations.append(True)
            reasons.append("EMA bearish")
        else:
            confirmations.append(False)
            reasons.append("EMA bullish")
    
    if use_macd:
        if indicators.macd_histogram is None:
            # Skip MACD check if not calculated
            pass
        elif indicators.macd_bearish:
            confirmations.append(True)
            reasons.append("MACD bearish")
        else:
            confirmations.append(False)
            reasons.append("MACD bullish")
    
    if not confirmations:
        # No indicators had enough data - skip confirmation
        return True, "Insufficient indicator data", 0
    
    if require_all:
        confirmed = all(confirmations)
    else:
        confirmed = any(confirmations)
    
    # Count how many indicators confirmed
    confirmation_count = sum(1 for c in confirmations if c)
    
    reason = " | ".join(reasons)
    return confirmed, reason, confirmation_count
