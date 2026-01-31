"""
Technical indicators for trading signal confirmation.

Provides RSI, EMA, MACD, and other indicators that can be used
to confirm or filter trading signals.
"""
import time
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

from ..shared.defaults import (
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    EMA_SHORT_PERIOD, EMA_LONG_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ATR_PERIOD, VOLATILITY_WINDOW,
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

    # Volatility (risk assessment, position sizing, confirmation filter)
    atr: Optional[float] = None  # Average True Range (absolute)
    atr_pct: Optional[float] = None  # ATR / price (e.g. for sizing)
    volatility_20: Optional[float] = None  # 20-day rolling std of returns


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
        atr_period: int = ATR_PERIOD,  # From shared.defaults
        volatility_window: int = VOLATILITY_WINDOW,
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
            atr_period: ATR period for volatility (default: from shared.defaults.ATR_PERIOD)
            volatility_window: Rolling window for return-volatility (default: VOLATILITY_WINDOW)
        """
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.volatility_window = volatility_window
    
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
    
    def calculate_atr(self, data: Union[pd.Series, pd.DataFrame], period: Optional[int] = None) -> pd.Series:
        """
        Calculate ATR (Average True Range). Uses High/Low/Close if DataFrame; else Close-only range.

        Args:
            data: DataFrame with High/Low/Close or Series (Close)
            period: ATR period (default: self.atr_period)

        Returns:
            Series of ATR values
        """
        period = period or self.atr_period
        if isinstance(data, pd.DataFrame) and all(c in data.columns for c in ('High', 'Low', 'Close')):
            high, low, close = data['High'], data['Low'], data['Close']
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        else:
            prices = data['Close'] if isinstance(data, pd.DataFrame) else data
            # Close-only approximation: range over period
            tr = prices.rolling(period, min_periods=period).apply(lambda x: x.max() - x.min(), raw=True)
            tr = tr.fillna(0)
        atr = tr.rolling(period, min_periods=period).mean()
        return atr

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate ADX (Average Directional Index) for trend strength detection.
        
        ADX measures trend strength (0-100):
        - ADX > 30: Strong trend
        - ADX 20-30: Moderate trend
        - ADX < 20: Weak/no trend
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close' columns OR Series with Close prices
            period: Period for ADX calculation (default: 14)
        
        Returns:
            Series with ADX values
        """
        # Handle both DataFrame (preferred) and Series (fallback) input
        if isinstance(data, pd.DataFrame) and 'High' in data.columns and 'Low' in data.columns and 'Close' in data.columns:
            # Use proper High/Low/Close for accurate ADX
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # True Range
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = pd.Series(0.0, index=data.index)
            minus_dm = pd.Series(0.0, index=data.index)
            
            plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
            minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
        else:
            # Fallback: use Close prices only (less accurate)
            if isinstance(data, pd.Series):
                prices = data
            else:
                prices = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
            
            high = prices
            low = prices
            
            tr = high.rolling(2).max() - low.rolling(2).min()
            tr = tr.fillna(0)
            
            up_move = high.diff()
            down_move = -low.diff()
            
            plus_dm = pd.Series(0.0, index=prices.index)
            minus_dm = pd.Series(0.0, index=prices.index)
            
            plus_dm[up_move > down_move] = up_move[up_move > down_move].clip(lower=0)
            minus_dm[down_move > up_move] = down_move[down_move > up_move].clip(lower=0)
        
        # Smoothed averages using Wilder's smoothing
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        
        # Directional Index (DX)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        
        # ADX (smoothed DX)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(0)
    
    def calculate_all(
        self,
        data: Union[pd.Series, pd.DataFrame],
        timings: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Calculate all indicators and return as DataFrame.

        Args:
            data: Price series or DataFrame with OHLCV columns
            timings: If provided, accumulate per-indicator elapsed seconds (keys: indicator_rsi, etc.)

        Returns:
            DataFrame with all indicator values
        """
        def _acc(key: str, elapsed: float) -> None:
            if timings is not None:
                timings[key] = timings.get(key, 0.0) + elapsed

        # Handle both Series and DataFrame input
        if isinstance(data, pd.Series):
            prices = data
            df = pd.DataFrame(index=prices.index)
            df['price'] = prices
        else:
            # DataFrame with OHLCV columns
            prices = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
            df = pd.DataFrame(index=data.index)
            df['price'] = prices

        # RSI
        t0 = time.perf_counter()
        df['rsi'] = self.calculate_rsi(prices)
        df['rsi_oversold'] = df['rsi'] < self.rsi_oversold
        df['rsi_overbought'] = df['rsi'] > self.rsi_overbought
        _acc('indicator_rsi', time.perf_counter() - t0)

        # EMAs
        t0 = time.perf_counter()
        df['ema_short'] = self.calculate_ema(prices, self.ema_short_period)
        df['ema_long'] = self.calculate_ema(prices, self.ema_long_period)
        df['price_above_ema_short'] = prices > df['ema_short']
        df['price_above_ema_long'] = prices > df['ema_long']
        ema_diff = df['ema_short'] - df['ema_long']
        ema_diff_prev = ema_diff.shift(1)
        df['ema_bullish_cross'] = (ema_diff > 0) & (ema_diff_prev <= 0)
        df['ema_bearish_cross'] = (ema_diff < 0) & (ema_diff_prev >= 0)
        _acc('indicator_ema', time.perf_counter() - t0)

        # MACD
        t0 = time.perf_counter()
        macd_line, signal_line, histogram = self.calculate_macd(prices)
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        histogram_prev = histogram.shift(1)
        df['macd_bullish'] = (histogram > 0) | ((histogram > histogram_prev) & (histogram_prev < 0))
        df['macd_bearish'] = (histogram < 0) | ((histogram < histogram_prev) & (histogram_prev > 0))
        _acc('indicator_macd', time.perf_counter() - t0)

        # ADX
        t0 = time.perf_counter()
        df['adx'] = self.calculate_adx(data)
        df['ma_50'] = prices.rolling(50).mean()
        df['ma_slope'] = df['ma_50'].diff(5)
        _acc('indicator_adx_ma', time.perf_counter() - t0)

        # Returns + volatility
        t0 = time.perf_counter()
        df['return_pct'] = prices.pct_change()
        df['volatility_20'] = df['return_pct'].rolling(self.volatility_window).std()
        _acc('indicator_volatility', time.perf_counter() - t0)

        # ATR
        t0 = time.perf_counter()
        df['atr'] = self.calculate_atr(data)
        df['atr_pct'] = (df['atr'] / prices).replace(0, np.nan)
        _acc('indicator_atr', time.perf_counter() - t0)

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
            atr=row['atr'] if not pd.isna(row.get('atr', np.nan)) else None,
            atr_pct=row['atr_pct'] if not pd.isna(row.get('atr_pct', np.nan)) else None,
            volatility_20=row['volatility_20'] if not pd.isna(row.get('volatility_20', np.nan)) else None,
        )


def _confirmation_flags_buy(indicators: IndicatorValues) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    """Per-indicator confirmation for buy: (rsi_ok, ema_ok, macd_ok). None if indicator not calculated."""
    rsi_ok = None
    if indicators.rsi is not None:
        rsi_ok = indicators.rsi_oversold or indicators.rsi < 40

    ema_ok = None
    if indicators.ema_short is not None:
        ema_ok = indicators.price_above_ema_short or indicators.ema_bullish_cross

    macd_ok = None
    if indicators.macd_histogram is not None:
        macd_ok = indicators.macd_bullish

    return rsi_ok, ema_ok, macd_ok


def _confirmation_flags_sell(indicators: IndicatorValues) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    """Per-indicator confirmation for sell: (rsi_ok, ema_ok, macd_ok). None if indicator not calculated."""
    rsi_ok = None
    if indicators.rsi is not None:
        rsi_ok = indicators.rsi_overbought or indicators.rsi > 60

    ema_ok = None
    if indicators.ema_short is not None:
        ema_ok = not indicators.price_above_ema_short or indicators.ema_bearish_cross

    macd_ok = None
    if indicators.macd_histogram is not None:
        macd_ok = indicators.macd_bearish

    return rsi_ok, ema_ok, macd_ok


def confirmation_weighted_score(
    indicators: Optional[IndicatorValues],
    use_rsi: bool,
    use_ema: bool,
    use_macd: bool,
    weights: Optional[Dict[str, float]] = None,
    for_buy: bool = True,
) -> Optional[float]:
    """
    Weighted confirmation score in [0, 1]. Returns None if weights not provided or no data.
    """
    if indicators is None or not weights:
        return None
    flags = _confirmation_flags_buy(indicators) if for_buy else _confirmation_flags_sell(indicators)
    w_rsi = weights.get("rsi", 1.0)
    w_ema = weights.get("ema", 1.0)
    w_macd = weights.get("macd", 1.0)
    total_weight = 0.0
    score = 0.0
    if use_rsi and flags[0] is not None:
        total_weight += w_rsi
        score += w_rsi * (1.0 if flags[0] else 0.0)
    if use_ema and flags[1] is not None:
        total_weight += w_ema
        score += w_ema * (1.0 if flags[1] else 0.0)
    if use_macd and flags[2] is not None:
        total_weight += w_macd
        score += w_macd * (1.0 if flags[2] else 0.0)
    if total_weight <= 0:
        return None
    return score / total_weight


def check_buy_confirmation(
    indicators: IndicatorValues,
    use_rsi: bool = True,
    use_ema: bool = True,
    use_macd: bool = True,
) -> Tuple[bool, str, int]:
    """
    Check if indicators confirm a buy signal (at least one enabled indicator must confirm).
    
    Args:
        indicators: Current indicator values
        use_rsi: Check RSI confirmation
        use_ema: Check EMA confirmation
        use_macd: Check MACD confirmation
    
    Returns:
        Tuple of (confirmed: bool, reason: str, confirmation_count: int)
        confirmation_count is the number of indicators that confirmed (0-3)
    """
    if indicators is None:
        return False, "No indicator data", 0

    rsi_ok, ema_ok, macd_ok = _confirmation_flags_buy(indicators)
    confirmations = []
    reasons = []

    if use_rsi:
        if rsi_ok is None:
            pass
        elif rsi_ok:
            confirmations.append(True)
            reasons.append(f"RSI={indicators.rsi:.0f} (favorable)")
        else:
            confirmations.append(False)
            reasons.append(f"RSI={indicators.rsi:.0f} (not oversold)")

    if use_ema:
        if ema_ok is None:
            pass
        elif ema_ok:
            confirmations.append(True)
            reasons.append("EMA bullish")
        else:
            confirmations.append(False)
            reasons.append("EMA bearish")

    if use_macd:
        if macd_ok is None:
            pass
        elif macd_ok:
            confirmations.append(True)
            reasons.append("MACD bullish")
        else:
            confirmations.append(False)
            reasons.append("MACD bearish")

    if not confirmations:
        return True, "Insufficient indicator data", 0

    confirmed = any(confirmations)
    confirmation_count = sum(1 for c in confirmations if c)
    reason = " | ".join(reasons)
    return confirmed, reason, confirmation_count


def check_sell_confirmation(
    indicators: IndicatorValues,
    use_rsi: bool = True,
    use_ema: bool = True,
    use_macd: bool = True,
) -> Tuple[bool, str, int]:
    """
    Check if indicators confirm a sell signal (at least one enabled indicator must confirm).
    
    Args:
        indicators: Current indicator values
        use_rsi: Check RSI confirmation
        use_ema: Check EMA confirmation
        use_macd: Check MACD confirmation
    
    Returns:
        Tuple of (confirmed: bool, reason: str, confirmation_count: int)
        confirmation_count is the number of indicators that confirmed (0-3)
    """
    if indicators is None:
        return False, "No indicator data", 0

    rsi_ok, ema_ok, macd_ok = _confirmation_flags_sell(indicators)
    confirmations = []
    reasons = []

    if use_rsi:
        if rsi_ok is None:
            pass
        elif rsi_ok:
            confirmations.append(True)
            reasons.append(f"RSI={indicators.rsi:.0f} (favorable)")
        else:
            confirmations.append(False)
            reasons.append(f"RSI={indicators.rsi:.0f} (not overbought)")

    if use_ema:
        if ema_ok is None:
            pass
        elif ema_ok:
            confirmations.append(True)
            reasons.append("EMA bearish")
        else:
            confirmations.append(False)
            reasons.append("EMA bullish")

    if use_macd:
        if macd_ok is None:
            pass
        elif macd_ok:
            confirmations.append(True)
            reasons.append("MACD bearish")
        else:
            confirmations.append(False)
            reasons.append("MACD bullish")

    if not confirmations:
        return True, "Insufficient indicator data", 0

    confirmed = any(confirmations)
    confirmation_count = sum(1 for c in confirmations if c)
    reason = " | ".join(reasons)
    return confirmed, reason, confirmation_count
