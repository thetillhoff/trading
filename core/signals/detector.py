"""
Unified signal detector using all indicators.

Treats all indicators (RSI, EMA, MACD, Elliott Wave) equally:
- Indicators calculate values from price data
- Signal generation interprets those values to create trading signals
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ..shared.types import SignalType, TradingSignal
from ..shared.defaults import INDICATOR_WARMUP_PERIOD, ADX_REGIME_THRESHOLD
from ..indicators.technical import (
    TechnicalIndicators,
    IndicatorValues,
    check_buy_confirmation,
    check_sell_confirmation,
    confirmation_weighted_score,
)
from ..indicators.elliott_wave import ElliottWaveDetector, Wave, WaveType, WaveLabel

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
            atr_period=getattr(config, 'atr_period', 14),
            volatility_window=getattr(config, 'volatility_window', 20),
        )
        
        # Create Elliott Wave detector if enabled (shared for regular, inverted, and inverted-exit)
        self.elliott_detector = None
        if (getattr(config, 'use_elliott_wave', False) or getattr(config, 'use_elliott_wave_inverted', False)
                or getattr(config, 'use_elliott_wave_inverted_exit', False)):
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
            ew_signals, _ = self._get_elliott_wave_signals(data, indicator_df)
            signals.extend(ew_signals)

        # Get signals from inverted Elliott Wave: exit-only (sell-to-close) or open-short
        if getattr(self.config, 'use_elliott_wave_inverted_exit', False) and self.elliott_detector:
            inverted_ew_signals, _ = self._get_inverted_elliott_wave_signals(data, indicator_df)
            for s in inverted_ew_signals:
                if s.signal_type == SignalType.SELL:
                    s.close_long_only = True
            signals.extend(inverted_ew_signals)
        elif getattr(self.config, 'use_elliott_wave_inverted', False) and self.elliott_detector:
            inverted_ew_signals, _ = self._get_inverted_elliott_wave_signals(data, indicator_df)
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

    def detect_signals_with_indicators(
        self,
        data: pd.Series,
        timings: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Signal], pd.DataFrame, List[Wave]]:
        """
        Detect trading signals and return signals, indicator dataframe, and waves for target calculation.

        Args:
            data: Price time series with datetime index
            timings: If provided, per-indicator elapsed seconds are accumulated here

        Returns:
            Tuple of (signals list, indicator dataframe, list of Elliott waves from this run)
        """
        signals: List[Signal] = []
        all_waves: List[Wave] = []

        # Calculate technical indicators
        indicator_df = None
        if self._uses_technical_indicators():
            indicator_df = self.technical_indicators.calculate_all(data, timings=timings)

        # Get signals from technical indicators
        if self._uses_technical_indicators():
            tech_signals = self._get_technical_indicator_signals(data, indicator_df)
            signals.extend(tech_signals)

        # Get signals from Elliott Wave (treated as indicator)
        if getattr(self.config, 'use_elliott_wave', False) and self.elliott_detector:
            ew_signals, ew_waves = self._get_elliott_wave_signals(data, indicator_df)
            signals.extend(ew_signals)
            all_waves.extend(ew_waves)

        # Get signals from inverted Elliott Wave: exit-only (sell-to-close) or open-short
        if getattr(self.config, 'use_elliott_wave_inverted_exit', False) and self.elliott_detector:
            inverted_ew_signals, inv_waves = self._get_inverted_elliott_wave_signals(data, indicator_df)
            for s in inverted_ew_signals:
                if s.signal_type == SignalType.SELL:
                    s.close_long_only = True
            signals.extend(inverted_ew_signals)
            all_waves.extend(inv_waves)
        elif getattr(self.config, 'use_elliott_wave_inverted', False) and self.elliott_detector:
            inverted_ew_signals, inv_waves = self._get_inverted_elliott_wave_signals(data, indicator_df)
            signals.extend(inverted_ew_signals)
            all_waves.extend(inv_waves)

        # Filter by signal type
        signal_types = getattr(self.config, 'signal_types', 'all')
        if signal_types == "buy":
            signals = [s for s in signals if s.signal_type == SignalType.BUY]
        elif signal_types == "sell":
            signals = [s for s in signals if s.signal_type == SignalType.SELL]

        # Filter by min confirmations and min certainty (signal quality)
        signals = self._filter_signals_by_quality(signals)

        # Sort by timestamp and remove duplicates
        signals = sorted(signals, key=lambda s: s.timestamp)
        signals = self._deduplicate_signals(signals)

        return signals, indicator_df, all_waves

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
    ) -> Tuple[List[Signal], List[Wave]]:
        """Generate signals from Elliott Wave patterns (treated as indicator). Returns (signals, waves)."""
        signals: List[Signal] = []
        waves: List[Wave] = []

        if not self.elliott_detector:
            return signals, waves

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
        except Exception:
            return signals, waves

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

        return signals, waves
    
    def _get_inverted_elliott_wave_signals(
        self,
        data: pd.Series,
        indicator_df: Optional[pd.DataFrame] = None
    ) -> Tuple[List[Signal], List[Wave]]:
        """
        Generate signals from inverted Elliott Wave patterns (for sell signal generation).
        Returns (signals, waves) with waves on inverted data for target calculation.

        Inverts price data, runs Elliott Wave detection on inverted data, then inverts
        the signal types to generate sell signals from bearish patterns.
        """
        signals: List[Signal] = []
        waves: List[Wave] = []

        # Run when either open-short or exit-only inverted EW is enabled
        if not (getattr(self.config, 'use_elliott_wave_inverted', False)
                or getattr(self.config, 'use_elliott_wave_inverted_exit', False)):
            return signals, waves

        if not self.elliott_detector:
            return signals, waves

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
        except Exception:
            return signals, waves

        # Generate signals from inverted waves
        for wave in waves:
            if wave.end_idx >= len(data):
                continue
            signal = self._inverted_wave_to_signal(wave, data, indicator_df)
            if signal:
                signals.append(signal)

        return signals, waves
    
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
            confirmed, reason, count, indicators = self._check_indicator_confirmation(
                signal, indicator_df
            )
            if confirmed:
                signal.reasoning += f" | {reason}"
                signal.indicator_confirmations = count
                signal.source = "combined_inverted"
            weights = getattr(self.config, 'indicator_weights', None)
            if weights and indicators is not None:
                score = confirmation_weighted_score(
                    indicators,
                    use_rsi=getattr(self.config, 'use_rsi', False),
                    use_ema=getattr(self.config, 'use_ema', False),
                    use_macd=getattr(self.config, 'use_macd', False),
                    weights=weights,
                    for_buy=(signal.signal_type == SignalType.BUY),
                )
                if score is not None:
                    signal.confirmation_score = score

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
        
        # Apply regime-based signal inversion (configurable; avoid silent no-ops)
        invert_signals = (
            getattr(self.config, 'use_regime_detection', False)
            and getattr(self.config, 'invert_signals_in_bull', True)
        )
        
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
            confirmed, reason, count, indicators = self._check_indicator_confirmation(
                signal, indicator_df
            )
            if confirmed:
                signal.reasoning += f" | {reason}"
                signal.indicator_confirmations = count
                signal.source = "combined"
            weights = getattr(self.config, 'indicator_weights', None)
            if weights and indicators is not None:
                score = confirmation_weighted_score(
                    indicators,
                    use_rsi=getattr(self.config, 'use_rsi', False),
                    use_ema=getattr(self.config, 'use_ema', False),
                    use_macd=getattr(self.config, 'use_macd', False),
                    weights=weights,
                    for_buy=(signal.signal_type == SignalType.BUY),
                )
                if score is not None:
                    signal.confirmation_score = score

        return signal

    def _check_indicator_confirmation(
        self,
        signal: Signal,
        indicator_df: pd.DataFrame,
    ) -> Tuple[bool, str, int, Optional[IndicatorValues]]:
        """Check if technical indicators confirm a signal. Returns (confirmed, reason, count, indicators)."""
        if signal.timestamp not in indicator_df.index:
            idx = indicator_df.index.get_indexer([signal.timestamp], method='ffill')[0]
            if idx < 0:
                return False, "No indicator data", 0, None
            timestamp = indicator_df.index[idx]
        else:
            timestamp = signal.timestamp

        row = indicator_df.loc[timestamp]

        def _nan(x, default=None):
            return default if (x is None or pd.isna(x)) else x
        indicators = IndicatorValues(
            timestamp=timestamp,
            price=row['price'],
            rsi=_nan(row['rsi']),
            rsi_oversold=bool(_nan(row['rsi_oversold'], False)),
            rsi_overbought=bool(_nan(row['rsi_overbought'], False)),
            ema_short=_nan(row['ema_short']),
            ema_long=_nan(row['ema_long']),
            price_above_ema_short=bool(_nan(row['price_above_ema_short'], False)),
            price_above_ema_long=bool(_nan(row['price_above_ema_long'], False)),
            ema_bullish_cross=bool(_nan(row['ema_bullish_cross'], False)),
            ema_bearish_cross=bool(_nan(row['ema_bearish_cross'], False)),
            macd_line=_nan(row['macd_line']),
            macd_signal=_nan(row['macd_signal']),
            macd_histogram=_nan(row['macd_histogram']),
            macd_bullish=bool(_nan(row['macd_bullish'], False)),
            macd_bearish=bool(_nan(row['macd_bearish'], False)),
            atr=_nan(row.get('atr')),
            atr_pct=_nan(row.get('atr_pct')),
            volatility_20=_nan(row.get('volatility_20')),
        )

        # Volatility filter: skip confirmation when 20d vol exceeds threshold
        if getattr(self.config, 'use_volatility_filter', False) and indicators.volatility_20 is not None:
            vol_max = getattr(self.config, 'volatility_max', 0.02)
            if indicators.volatility_20 > vol_max:
                return False, f"Volatility too high ({indicators.volatility_20:.4f} > {vol_max})", 0, indicators

        use_rsi = getattr(self.config, 'use_rsi', False)
        use_ema = getattr(self.config, 'use_ema', False)
        use_macd = getattr(self.config, 'use_macd', False)

        if signal.signal_type == SignalType.BUY:
            confirmed, reason, count = check_buy_confirmation(
                indicators, use_rsi=use_rsi, use_ema=use_ema, use_macd=use_macd
            )
        else:
            confirmed, reason, count = check_sell_confirmation(
                indicators, use_rsi=use_rsi, use_ema=use_ema, use_macd=use_macd
            )

        return confirmed, reason, count, indicators
    
    def _detect_market_regime(
        self,
        data: pd.Series,
        indicator_df: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> str:
        """
        Detect market regime at given timestamp using a selectable regime model.
        
        Args:
            data: Price series
            indicator_df: DataFrame with calculated indicators (must include 'adx' and 'ma_slope')
            timestamp: Timestamp to detect regime for
        
        Returns:
            "BULL" - Strong uptrend (ADX > 30, MA slope positive)
            "BEAR" - Strong downtrend (ADX > 30, MA slope negative) or weak trend
            "SIDEWAYS" - Choppy/ranging market (ADX < 30)
        """
        mode = getattr(self.config, "regime_mode", "adx_ma")
        if mode == "trend_vol":
            return self._detect_market_regime_trend_vol(indicator_df, timestamp)

        # Default / backward-compatible
        return self._detect_market_regime_adx_ma(indicator_df, timestamp)

    def _detect_market_regime_adx_ma(
        self,
        indicator_df: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> str:
        """ADX + MA-slope regime classifier (legacy)."""
        # Default to BEAR (conservative - use original EW signals)
        if timestamp not in indicator_df.index:
            return "BEAR"

        row = indicator_df.loc[timestamp]
        adx = row.get('adx', 0)
        ma_slope = row.get('ma_slope', 0)

        # Handle missing/NaN values
        if pd.isna(adx) or pd.isna(ma_slope):
            return "BEAR"

        # Strong trend (ADX > threshold) + direction from MA slope
        adx_threshold = getattr(self.config, 'adx_threshold', ADX_REGIME_THRESHOLD)
        if adx > adx_threshold:
            return "BULL" if ma_slope > 0 else "BEAR"

        # Weak trend - treat as SIDEWAYS (choppy, use original EW signals)
        return "SIDEWAYS"

    def _detect_market_regime_trend_vol(
        self,
        indicator_df: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> str:
        """
        Close-only regime classifier using MA slope + return volatility.

        Intent:
        - classify obvious trends via (abs(ma_slope)/price) threshold
        - treat low-slope environments as SIDEWAYS
        - conservatively avoid classifying as BULL in high volatility
        """
        if timestamp not in indicator_df.index:
            return "BEAR"

        row = indicator_df.loc[timestamp]
        price = row.get("price", None)
        ma_slope = row.get("ma_slope", 0.0)
        vol = row.get("volatility_20", 0.0)

        if price is None or pd.isna(price) or pd.isna(ma_slope):
            return "BEAR"

        slope_threshold = getattr(self.config, "regime_slope_threshold", 0.0005)
        slope_ratio = abs(ma_slope) / price if price else 0.0
        if slope_ratio < slope_threshold:
            return "SIDEWAYS"

        regime = "BULL" if ma_slope > 0 else "BEAR"

        vol_threshold = getattr(self.config, "regime_vol_threshold", 0.015)
        if regime == "BULL" and vol is not None and not pd.isna(vol) and vol > vol_threshold:
            return "SIDEWAYS"

        return regime

    def _filter_signals_by_quality(self, signals: List[Signal]) -> List[Signal]:
        """Keep only signals meeting min_confirmations and min_certainty (if set)."""
        min_confirmations = getattr(self.config, 'min_confirmations', None)
        min_certainty = getattr(self.config, 'min_certainty', None)
        if min_confirmations is None and min_certainty is None:
            return signals
        filtered = []
        for s in signals:
            ok_conf = min_confirmations is None or getattr(s, 'indicator_confirmations', 0) >= min_confirmations
            effective_certainty = (
                s.confirmation_score
                if getattr(s, 'confirmation_score', None) is not None
                else getattr(s, 'indicator_confirmations', 0) / 3.0
            )
            ok_cert = min_certainty is None or effective_certainty >= min_certainty
            if ok_conf and ok_cert:
                filtered.append(s)
        return filtered
    
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
