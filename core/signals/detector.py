"""
Unified signal detector using all indicators.

Treats all indicators (RSI, EMA, MACD, Elliott Wave) equally:
- Indicators calculate values from price data
- Signal generation interprets those values to create trading signals
"""
import pandas as pd
from typing import List, Optional, Tuple
import sys
from pathlib import Path

# Add paths for imports
core_dir = Path(__file__).parent.parent.parent
project_root = core_dir.parent
sys.path.insert(0, str(project_root))

from core.shared.types import SignalType, TradingSignal
from core.shared.defaults import INDICATOR_WARMUP_PERIOD, ADX_REGIME_THRESHOLD
from core.indicators.technical import TechnicalIndicators, IndicatorValues, check_buy_confirmation, check_sell_confirmation
from core.indicators.elliott_wave import ElliottWaveDetector, Wave, WaveType, WaveLabel

# Alias for backward compatibility
Signal = TradingSignal


class SignalDetector:
    """
    Unified signal detector using all indicators.
    
    All indicators (RSI, EMA, MACD, Elliott Wave) are treated equally:
    - They calculate values from price data
    - Signal generation interprets those values
    """
    
    def __init__(self, config):
        """
        Initialize the signal detector.
        
        Args:
            config: SignalConfig or StrategyConfig with indicator settings
        """
        self.config = config
        
        # Create technical indicators calculator
        self.technical_indicators = TechnicalIndicators(
            rsi_period=getattr(config, 'rsi_period', 7),
            rsi_oversold=getattr(config, 'rsi_oversold', 25),
            rsi_overbought=getattr(config, 'rsi_overbought', 75),
            ema_short_period=getattr(config, 'ema_short_period', 20),
            ema_long_period=getattr(config, 'ema_long_period', 50),
            macd_fast=getattr(config, 'macd_fast', 12),
            macd_slow=getattr(config, 'macd_slow', 26),
            macd_signal=getattr(config, 'macd_signal', 12),
        )
        
        # Create Elliott Wave detector if enabled (shared for both regular and inverted)
        self.elliott_detector = None
        if getattr(config, 'use_elliott_wave', False) or getattr(config, 'use_elliott_wave_inverted', False):
            self.elliott_detector = ElliottWaveDetector()
    
    def detect_signals(self, data: pd.Series) -> List[Signal]:
        """
        Detect trading signals from all enabled indicators.
        
        Args:
            data: Price time series with datetime index
        
        Returns:
            List of trading signals
        """
        signals = []
        
        # Calculate technical indicators
        indicator_df = None
        if self._uses_technical_indicators():
            indicator_df = self.technical_indicators.calculate_all(data)
        
        # Get signals from technical indicators
        if self._uses_technical_indicators():
            tech_signals = self._get_technical_indicator_signals(data, indicator_df)
            signals.extend(tech_signals)
        
        # Get signals from Elliott Wave (treated as indicator)
        if getattr(self.config, 'use_elliott_wave', False) and self.elliott_detector:
            ew_signals = self._get_elliott_wave_signals(data, indicator_df)
            signals.extend(ew_signals)
        
        # Get signals from inverted Elliott Wave (for sell signal generation)
        if getattr(self.config, 'use_elliott_wave_inverted', False) and self.elliott_detector:
            inverted_ew_signals = self._get_inverted_elliott_wave_signals(data, indicator_df)
            signals.extend(inverted_ew_signals)
        
        # Filter by signal type
        signal_types = getattr(self.config, 'signal_types', 'all')
        if signal_types == "buy":
            signals = [s for s in signals if s.signal_type == SignalType.BUY]
        elif signal_types == "sell":
            signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        # Sort by timestamp and remove duplicates
        signals = sorted(signals, key=lambda s: s.timestamp)
        signals = self._deduplicate_signals(signals)
        
        return signals

    def detect_signals_with_indicators(self, data: pd.Series) -> Tuple[List[Signal], pd.DataFrame]:
        """
        Detect trading signals and return both signals and indicator dataframe.

        Args:
            data: Price time series with datetime index

        Returns:
            Tuple of (signals list, indicator dataframe)
        """
        signals = []

        # Calculate technical indicators
        indicator_df = None
        if self._uses_technical_indicators():
            indicator_df = self.technical_indicators.calculate_all(data)

        # Get signals from technical indicators
        if self._uses_technical_indicators():
            tech_signals = self._get_technical_indicator_signals(data, indicator_df)
            signals.extend(tech_signals)

        # Get signals from Elliott Wave (treated as indicator)
        if getattr(self.config, 'use_elliott_wave', False) and self.elliott_detector:
            ew_signals = self._get_elliott_wave_signals(data, indicator_df)
            signals.extend(ew_signals)
        
        # Get signals from inverted Elliott Wave (for sell signal generation)
        if getattr(self.config, 'use_elliott_wave_inverted', False) and self.elliott_detector:
            inverted_ew_signals = self._get_inverted_elliott_wave_signals(data, indicator_df)
            signals.extend(inverted_ew_signals)

        # Filter by signal type
        signal_types = getattr(self.config, 'signal_types', 'all')
        if signal_types == "buy":
            signals = [s for s in signals if s.signal_type == SignalType.BUY]
        elif signal_types == "sell":
            signals = [s for s in signals if s.signal_type == SignalType.SELL]

        # Sort by timestamp and remove duplicates
        signals = sorted(signals, key=lambda s: s.timestamp)
        signals = self._deduplicate_signals(signals)

        return signals, indicator_df

    def _invert_price_data(self, data: pd.Series) -> pd.Series:
        """
        Invert price data to detect bearish patterns as bullish patterns.
        
        Uses: inverted_price = max_price + min_price - price
        This preserves the relative structure while flipping the direction.
        
        Args:
            data: Original price time series
            
        Returns:
            Inverted price series with same index
        """
        max_price = data.max()
        min_price = data.min()
        inverted_data = max_price + min_price - data
        return inverted_data

    def _uses_technical_indicators(self) -> bool:
        """Check if any technical indicators are enabled."""
        return (getattr(self.config, 'use_rsi', False) or
                getattr(self.config, 'use_ema', False) or
                getattr(self.config, 'use_macd', False))
    
    def _get_technical_indicator_signals(
        self,
        data: pd.Series,
        indicator_df: pd.DataFrame,
    ) -> List[Signal]:
        """Generate signals from technical indicators (RSI, EMA, MACD)."""
        signals = []
        
        if indicator_df is None or len(indicator_df) < INDICATOR_WARMUP_PERIOD:
            return signals
        
        start_idx = INDICATOR_WARMUP_PERIOD
        
        for i in range(start_idx, len(indicator_df)):
            row = indicator_df.iloc[i]
            prev_row = indicator_df.iloc[i - 1]
            timestamp = indicator_df.index[i]
            price = row['price']
            
            # Skip if indicators not calculated
            if pd.isna(row.get('rsi')) or pd.isna(row.get('macd_histogram')):
                continue
            
            buy_reasons = []
            sell_reasons = []
            
            # RSI signals
            if getattr(self.config, 'use_rsi', False):
                if prev_row['rsi_oversold'] and not row['rsi_oversold']:
                    buy_reasons.append(f"RSI exit oversold ({row['rsi']:.0f})")
                if not prev_row['rsi_overbought'] and row['rsi_overbought']:
                    sell_reasons.append(f"RSI enter overbought ({row['rsi']:.0f})")
            
            # EMA signals
            if getattr(self.config, 'use_ema', False):
                if row['ema_bullish_cross']:
                    buy_reasons.append("EMA bullish cross")
                if row['ema_bearish_cross']:
                    sell_reasons.append("EMA bearish cross")
            
            # MACD signals
            if getattr(self.config, 'use_macd', False):
                if prev_row['macd_histogram'] < 0 and row['macd_histogram'] >= 0:
                    buy_reasons.append("MACD cross above zero")
                if prev_row['macd_histogram'] > 0 and row['macd_histogram'] <= 0:
                    sell_reasons.append("MACD cross below zero")
            
            # Apply trend filter if enabled
            use_trend_filter = getattr(self.config, 'use_trend_filter', False)
            ema_short = row.get('ema_short')
            ema_long = row.get('ema_long')
            
            # Determine trend direction
            is_bullish_trend = ema_short > ema_long if (ema_short is not None and ema_long is not None) else None
            
            # Apply require-all-indicators filter if enabled
            require_all = getattr(self.config, 'require_all_indicators', False)
            if require_all:
                # Count how many indicators are enabled
                enabled_count = sum([
                    getattr(self.config, 'use_rsi', False),
                    getattr(self.config, 'use_ema', False),
                    getattr(self.config, 'use_macd', False),
                ])
                # Filter signals that don't have all indicators confirming
                if len(buy_reasons) < enabled_count:
                    buy_reasons = []
                if len(sell_reasons) < enabled_count:
                    sell_reasons = []
            
            # Generate signals
            if buy_reasons:
                # Apply trend filter: only generate BUY signals in bullish trend
                if use_trend_filter and is_bullish_trend is not None:
                    if not is_bullish_trend:
                        # Skip this BUY signal - trend is bearish
                        buy_reasons = []
                
                if buy_reasons:  # Re-check after filtering
                    confidence = 0.5 + (len(buy_reasons) * 0.15)
                    reasoning = " | ".join(buy_reasons)
                    if use_trend_filter and is_bullish_trend:
                        reasoning += " | Trend: Bullish"
                    
                    signals.append(Signal(
                        signal_type=SignalType.BUY,
                        timestamp=timestamp,
                        price=price,
                        confidence=min(confidence, 0.9),
                        source="indicator",
                        reasoning=reasoning,
                        indicator_confirmations=len(buy_reasons),
                        rsi_value=row.get('rsi'),
                        ema_short=ema_short,
                        ema_long=ema_long,
                        macd_value=row.get('macd'),
                        macd_signal=row.get('macd_signal'),
                        macd_histogram=row.get('macd_histogram'),
                        trend_filter_active=use_trend_filter,
                    ))

            if sell_reasons:
                # Apply trend filter: only generate SELL signals in bearish trend
                if use_trend_filter and is_bullish_trend is not None:
                    if is_bullish_trend:
                        # Skip this SELL signal - trend is bullish
                        sell_reasons = []
                
                if sell_reasons:  # Re-check after filtering
                    confidence = 0.5 + (len(sell_reasons) * 0.15)
                    reasoning = " | ".join(sell_reasons)
                    if use_trend_filter and not is_bullish_trend:
                        reasoning += " | Trend: Bearish"
                    
                    signals.append(Signal(
                        signal_type=SignalType.SELL,
                        timestamp=timestamp,
                        price=price,
                        confidence=min(confidence, 0.9),
                        source="indicator",
                        reasoning=reasoning,
                        indicator_confirmations=len(sell_reasons),
                        rsi_value=row.get('rsi'),
                        ema_short=ema_short,
                        ema_long=ema_long,
                        macd_value=row.get('macd'),
                        macd_signal=row.get('macd_signal'),
                        macd_histogram=row.get('macd_histogram'),
                        trend_filter_active=use_trend_filter,
                    ))
        
        return signals
    
    def _get_elliott_wave_signals(
        self,
        data: pd.Series,
        indicator_df: Optional[pd.DataFrame] = None
    ) -> List[Signal]:
        """Generate signals from Elliott Wave patterns (treated as indicator)."""
        signals = []
        
        if not self.elliott_detector:
            return signals
        
        # Detect waves
        min_confidence = getattr(self.config, 'min_confidence', 0.65)
        min_wave_size = getattr(self.config, 'min_wave_size', 0.03)
        
        try:
            waves = self.elliott_detector.detect_waves(
                data,
                min_confidence=min_confidence,
                min_wave_size_ratio=min_wave_size,
                only_complete_patterns=False
            )
        except Exception as e:
            return signals  # Return empty on error
        
        # Generate signals from wave completions with regime detection
        for wave in waves:
            # Detect market regime at wave completion
            regime = "BEAR"  # Default to BEAR (conservative, uses original EW signals)
            
            if getattr(self.config, 'use_regime_detection', False) and indicator_df is not None:
                # Get timestamp for wave completion
                if wave.end_idx < len(data):
                    wave_timestamp = data.index[wave.end_idx]
                    regime = self._detect_market_regime(data, indicator_df, wave_timestamp)
            
            # Generate signal with regime-adapted logic
            signal = self._wave_to_signal(wave, data, indicator_df, market_regime=regime)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _get_inverted_elliott_wave_signals(
        self,
        data: pd.Series,
        indicator_df: Optional[pd.DataFrame] = None
    ) -> List[Signal]:
        """
        Generate signals from inverted Elliott Wave patterns (for sell signal generation).
        
        Inverts price data, runs Elliott Wave detection on inverted data, then inverts
        the signal types to generate sell signals from bearish patterns.
        
        Args:
            data: Original price time series
            indicator_df: Technical indicators dataframe (optional)
            
        Returns:
            List of trading signals (inverted from detected waves)
        """
        signals = []
        
        if not getattr(self.config, 'use_elliott_wave_inverted', False):
            return signals
        
        if not self.elliott_detector:
            return signals
        
        # Invert price data
        inverted_data = self._invert_price_data(data)
        
        # Detect waves on inverted data
        min_confidence = getattr(self.config, 'min_confidence_inverted', 0.65)
        min_wave_size = getattr(self.config, 'min_wave_size_inverted', 0.02)
        
        try:
            waves = self.elliott_detector.detect_waves(
                inverted_data,
                min_confidence=min_confidence,
                min_wave_size_ratio=min_wave_size,
                only_complete_patterns=False
            )
        except Exception as e:
            return signals  # Return empty on error
        
        # Generate signals from inverted waves
        # When a "buy" signal is detected on inverted data (bearish pattern),
        # it becomes a "sell" signal in the original market
        for wave in waves:
            # Map inverted wave to original data indices
            # The wave indices are relative to inverted_data, but we need to map
            # them back to the original data (indices should be the same)
            if wave.end_idx >= len(data):
                continue
            
            # Convert wave to signal with inverted signal types
            signal = self._inverted_wave_to_signal(wave, data, indicator_df)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _inverted_wave_to_signal(
        self,
        wave: Wave,
        data: pd.Series,
        indicator_df: Optional[pd.DataFrame] = None
    ) -> Optional[Signal]:
        """
        Convert an inverted Elliott Wave to a trading signal with inverted signal types.
        
        When waves are detected on inverted data:
        - Inverted Wave 2/4 (end of correction in inverted = bearish pattern) → SELL signal
        - Inverted Wave 5 (completion in inverted = bearish completion) → BUY signal
        
        Args:
            wave: Elliott Wave pattern detected on inverted data
            data: Original price series (not inverted)
            indicator_df: Technical indicators dataframe (optional)
        
        Returns:
            Trading signal with inverted signal type
        """
        if wave.end_idx >= len(data):
            return None
        
        price = data.iloc[wave.end_idx]
        timestamp = data.index[wave.end_idx]
        
        # Determine signal type from inverted wave
        # On inverted data, what looks like a buy (Wave 2/4) is actually a sell
        # and what looks like a sell (Wave 5) is actually a buy
        if wave.wave_type == WaveType.IMPULSE:
            if wave.label == WaveLabel.WAVE_2:
                # Inverted Wave 2 (correction in inverted = bearish pattern) → SELL
                signal_type = SignalType.SELL
                base_confidence = wave.confidence * 0.8
                base_reasoning = "Inverted Elliott Wave: End of Wave 2 correction (bearish pattern)"
            
            elif wave.label == WaveLabel.WAVE_4:
                # Inverted Wave 4 (correction in inverted = bearish pattern) → SELL
                signal_type = SignalType.SELL
                base_confidence = wave.confidence * 0.7
                base_reasoning = "Inverted Elliott Wave: End of Wave 4 correction (bearish pattern)"
            
            elif wave.label == WaveLabel.WAVE_5:
                # Inverted Wave 5 (completion in inverted = bearish completion) → BUY
                signal_type = SignalType.BUY
                base_confidence = wave.confidence * 0.9
                base_reasoning = "Inverted Elliott Wave: End of Wave 5 (bearish completion)"
            
            else:
                # Other wave labels not used for signals
                return None
        else:
            # Non-impulse waves not used for signals
            return None
        
        # Create signal with inverted type
        signal = Signal(
            signal_type=signal_type,
            timestamp=timestamp,
            price=price,
            confidence=base_confidence,
            source="elliott_inverted",
            reasoning=base_reasoning,
            wave=wave,
            indicator_confirmations=0,
        )
        
        # Check technical indicator confirmation if available
        if indicator_df is not None and self._uses_technical_indicators():
            confirmed, reason, count = self._check_indicator_confirmation(
                signal, indicator_df
            )
            if confirmed:
                signal.reasoning += f" | {reason}"
                signal.indicator_confirmations = count
                signal.source = "combined_inverted"
        
        return signal
    
    def _wave_to_signal(
        self,
        wave: Wave,
        data: pd.Series,
        indicator_df: Optional[pd.DataFrame] = None,
        market_regime: str = "BEAR"
    ) -> Optional[Signal]:
        """
        Convert an Elliott Wave to a trading signal with regime-based adaptation.
        
        Args:
            wave: Elliott Wave pattern
            data: Price series
            indicator_df: Technical indicators dataframe
            market_regime: Market regime ("BULL", "BEAR", or "SIDEWAYS")
        
        Returns:
            Trading signal with regime-adapted signal type
        """
        if wave.end_idx >= len(data):
            return None
        
        price = data.iloc[wave.end_idx]
        timestamp = data.index[wave.end_idx]
        
        # Generate signals based on wave type and label
        # Original logic:
        # - Buy signals: End of corrective waves (Wave 2, Wave 4)
        # - Sell signals: End of impulse completion (Wave 5)
        
        # Determine original signal type and confidence
        if wave.wave_type == WaveType.IMPULSE:
            if wave.label == WaveLabel.WAVE_2:
                original_signal_type = SignalType.BUY
                base_confidence = wave.confidence * 0.8
                base_reasoning = "End of Wave 2 correction"
            
            elif wave.label == WaveLabel.WAVE_4:
                original_signal_type = SignalType.BUY
                base_confidence = wave.confidence * 0.7
                base_reasoning = "End of Wave 4 correction"
            
            elif wave.label == WaveLabel.WAVE_5:
                original_signal_type = SignalType.SELL
                base_confidence = wave.confidence * 0.9
                base_reasoning = "End of Wave 5 (impulse complete)"
            
            else:
                # Other wave labels not used for signals
                return None
        else:
            # Non-impulse waves not used for signals
            return None
        
        # Apply regime-based signal inversion
        invert_signals = getattr(self.config, 'use_regime_detection', False)
        
        if invert_signals and market_regime == "BULL":
            # In BULL markets, invert signals:
            # - Wave 2/4 corrections → SELL (sell the rally, counter-trend)
            # - Wave 5 completions → BUY (re-enter on trend continuation)
            if original_signal_type == SignalType.BUY:
                final_signal_type = SignalType.SELL
                reasoning = f"{base_reasoning} (BULL regime - inverted)"
            else:
                final_signal_type = SignalType.BUY
                reasoning = f"{base_reasoning} (BULL regime - re-entry)"
        else:
            # BEAR or SIDEWAYS markets: use original signals
            final_signal_type = original_signal_type
            if market_regime != "BEAR":
                reasoning = f"{base_reasoning} ({market_regime} regime)"
            else:
                reasoning = base_reasoning
        
        # Create signal with final type
        signal = Signal(
            signal_type=final_signal_type,
            timestamp=timestamp,
            price=price,
            confidence=base_confidence,
            source="elliott",
            reasoning=reasoning,
            wave=wave,
            indicator_confirmations=0,
        )
        
        # Check technical indicator confirmation if available
        if indicator_df is not None and self._uses_technical_indicators():
            confirmed, reason, count = self._check_indicator_confirmation(
                signal, indicator_df
            )
            if confirmed:
                signal.reasoning += f" | {reason}"
                signal.indicator_confirmations = count
                signal.source = "combined"
        
        return signal
    
    def _check_indicator_confirmation(
        self,
        signal: Signal,
        indicator_df: pd.DataFrame,
    ) -> Tuple[bool, str, int]:
        """Check if technical indicators confirm a signal."""
        if signal.timestamp not in indicator_df.index:
            idx = indicator_df.index.get_indexer([signal.timestamp], method='ffill')[0]
            if idx < 0:
                return False, "No indicator data", 0
            timestamp = indicator_df.index[idx]
        else:
            timestamp = signal.timestamp
        
        row = indicator_df.loc[timestamp]
        
        # Build indicator values
        indicators = IndicatorValues(
            timestamp=timestamp,
            price=row['price'],
            rsi=row['rsi'] if not pd.isna(row['rsi']) else None,
            rsi_oversold=row['rsi_oversold'] if not pd.isna(row['rsi_oversold']) else False,
            rsi_overbought=row['rsi_overbought'] if not pd.isna(row['rsi_overbought']) else False,
            ema_short=row['ema_short'] if not pd.isna(row['ema_short']) else None,
            ema_long=row['ema_long'] if not pd.isna(row['ema_long']) else None,
            price_above_ema_short=row['price_above_ema_short'] if not pd.isna(row['price_above_ema_short']) else False,
            price_above_ema_long=row['price_above_ema_long'] if not pd.isna(row['price_above_ema_long']) else False,
            ema_bullish_cross=row['ema_bullish_cross'] if not pd.isna(row['ema_bullish_cross']) else False,
            ema_bearish_cross=row['ema_bearish_cross'] if not pd.isna(row['ema_bearish_cross']) else False,
            macd_line=row['macd_line'] if not pd.isna(row['macd_line']) else None,
            macd_signal=row['macd_signal'] if not pd.isna(row['macd_signal']) else None,
            macd_histogram=row['macd_histogram'] if not pd.isna(row['macd_histogram']) else None,
            macd_bullish=row['macd_bullish'] if not pd.isna(row['macd_bullish']) else False,
            macd_bearish=row['macd_bearish'] if not pd.isna(row['macd_bearish']) else False,
        )
        
        if signal.signal_type == SignalType.BUY:
            return check_buy_confirmation(
                indicators,
                use_rsi=getattr(self.config, 'use_rsi', False),
                use_ema=getattr(self.config, 'use_ema', False),
                use_macd=getattr(self.config, 'use_macd', False),
                require_all=getattr(self.config, 'require_all_indicators', False),
            )
        else:
            return check_sell_confirmation(
                indicators,
                use_rsi=getattr(self.config, 'use_rsi', False),
                use_ema=getattr(self.config, 'use_ema', False),
                use_macd=getattr(self.config, 'use_macd', False),
                require_all=getattr(self.config, 'require_all_indicators', False),
            )
    
    def _detect_market_regime(
        self,
        data: pd.Series,
        indicator_df: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> str:
        """
        Detect market regime at given timestamp using ADX and MA slope.
        
        Args:
            data: Price series
            indicator_df: DataFrame with calculated indicators (must include 'adx' and 'ma_slope')
            timestamp: Timestamp to detect regime for
        
        Returns:
            "BULL" - Strong uptrend (ADX > 30, MA slope positive)
            "BEAR" - Strong downtrend (ADX > 30, MA slope negative) or weak trend
            "SIDEWAYS" - Choppy/ranging market (ADX < 30)
        """
        # Default to BEAR (conservative - use original EW signals)
        if timestamp not in indicator_df.index:
            return "BEAR"
        
        row = indicator_df.loc[timestamp]
        adx = row.get('adx', 0)
        ma_slope = row.get('ma_slope', 0)
        
        # Handle missing/NaN values
        if pd.isna(adx) or pd.isna(ma_slope):
            return "BEAR"
        
        # Regime detection logic:
        # Strong trend (ADX > threshold) + direction from MA slope
        adx_threshold = getattr(self.config, 'adx_threshold', ADX_REGIME_THRESHOLD)
        if adx > adx_threshold:
            if ma_slope > 0:
                return "BULL"  # Strong bull trend
            else:
                return "BEAR"  # Strong bear trend
        else:
            # Weak trend - treat as SIDEWAYS (choppy, use original EW signals)
            return "SIDEWAYS"
    
    def _deduplicate_signals(self, signals: List[Signal]) -> List[Signal]:
        """Remove duplicate signals on the same day."""
        seen = set()
        unique = []
        
        for sig in signals:
            key = (sig.timestamp.date(), sig.signal_type)
            if key not in seen:
                seen.add(key)
                unique.append(sig)
        
        return unique
