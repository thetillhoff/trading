"""
Detects buy and sell signals based on Elliott Wave patterns.

Elliott Wave Theory suggests:
- Buy signals: End of wave 2 or wave 4 (corrections in uptrend)
- Sell signals: End of wave 5 (completion of impulse)
- Also considers corrective wave patterns for counter-trend opportunities
"""
import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add parent directories to path for imports
djia_dir = Path('/app/djia')
if not djia_dir.exists():
    djia_dir = Path(__file__).parent.parent / 'djia'
sys.path.insert(0, str(djia_dir))
from elliott_wave_detector import ElliottWaveDetector, Wave, WaveType, WaveLabel


class SignalType(Enum):
    """Type of trading signal."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """Represents a trading signal with target prices."""
    signal_type: SignalType
    timestamp: pd.Timestamp
    price: float
    wave: Wave
    confidence: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""


class SignalDetector:
    """Detects trading signals from Elliott Wave patterns."""
    
    def __init__(self):
        """Initialize the signal detector."""
        self.detector = ElliottWaveDetector()
    
    def detect_signals(
        self,
        data: pd.Series,
        min_confidence: float = 0.6,
        min_wave_size_ratio: float = 0.05
    ) -> List[TradingSignal]:
        """
        Detect buy and sell signals from Elliott Wave patterns.
        
        Args:
            data: Time series data with datetime index
            min_confidence: Minimum confidence for wave detection
            min_wave_size_ratio: Minimum wave size ratio
            
        Returns:
            List of trading signals
        """
        # Detect waves
        waves = self.detector.detect_waves(
            data,
            min_confidence=min_confidence,
            min_wave_size_ratio=min_wave_size_ratio,
            only_complete_patterns=False
        )
        
        if not waves:
            return []
        
        signals = []
        
        # Analyze waves for trading signals
        for i, wave in enumerate(waves):
            signal = self._analyze_wave_for_signal(data, wave, waves, i)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _analyze_wave_for_signal(
        self,
        data: pd.Series,
        wave: Wave,
        all_waves: List[Wave],
        wave_index: int
    ) -> Optional[TradingSignal]:
        """
        Analyze a single wave to determine if it represents a trading signal.
        
        Args:
            data: Time series data
            wave: Wave to analyze
            all_waves: All detected waves
            wave_index: Index of current wave in all_waves list
            
        Returns:
            TradingSignal if signal detected, None otherwise
        """
        # Get wave end price and timestamp
        if wave.end_idx >= len(data):
            return None
        
        price = data.iloc[wave.end_idx]
        timestamp = data.index[wave.end_idx]
        
        # Buy signals: End of corrective waves in uptrend (wave 2, wave 4, wave b)
        if wave.wave_type == WaveType.IMPULSE:
            if wave.label == WaveLabel.WAVE_2:
                # End of wave 2 - potential buy (correction in uptrend)
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    timestamp=timestamp,
                    price=price,
                    wave=wave,
                    confidence=wave.confidence * 0.8,  # Slightly lower confidence for signals
                    reasoning="End of Wave 2 correction - potential buy entry"
                )
            elif wave.label == WaveLabel.WAVE_4:
                # End of wave 4 - potential buy (correction in uptrend)
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    timestamp=timestamp,
                    price=price,
                    wave=wave,
                    confidence=wave.confidence * 0.8,
                    reasoning="End of Wave 4 correction - potential buy entry"
                )
            elif wave.label == WaveLabel.WAVE_5:
                # End of wave 5 - potential sell (completion of impulse)
                return TradingSignal(
                    signal_type=SignalType.SELL,
                    timestamp=timestamp,
                    price=price,
                    wave=wave,
                    confidence=wave.confidence * 0.9,
                    reasoning="End of Wave 5 - potential sell (impulse complete)"
                )
        
        elif wave.wave_type == WaveType.CORRECTIVE:
            if wave.label == WaveLabel.WAVE_B:
                # End of wave b - potential counter-trend opportunity
                # Check if this is in a downtrend (wave b up in downtrend = sell)
                # or uptrend (wave b down in uptrend = buy)
                prev_wave = self._find_previous_wave(all_waves, wave_index)
                if prev_wave and prev_wave.direction == "down":
                    # Wave b up in downtrend - potential sell
                    return TradingSignal(
                        signal_type=SignalType.SELL,
                        timestamp=timestamp,
                        price=price,
                        wave=wave,
                        confidence=wave.confidence * 0.7,
                        reasoning="End of Wave B (corrective up) - potential sell"
                    )
                elif prev_wave and prev_wave.direction == "up":
                    # Wave b down in uptrend - potential buy
                    return TradingSignal(
                        signal_type=SignalType.BUY,
                        timestamp=timestamp,
                        price=price,
                        wave=wave,
                        confidence=wave.confidence * 0.7,
                        reasoning="End of Wave B (corrective down) - potential buy"
                    )
        
        return None
    
    def _find_previous_wave(
        self,
        waves: List[Wave],
        current_index: int
    ) -> Optional[Wave]:
        """Find the previous wave before the current one."""
        if current_index > 0:
            return waves[current_index - 1]
        return None
