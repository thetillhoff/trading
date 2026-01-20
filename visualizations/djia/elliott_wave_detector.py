"""
Elliott Wave detection module for identifying wave patterns in price data.

Elliott Wave Theory identifies recurring patterns in price movements:
- Impulse waves: 5 waves in the direction of the trend (1, 2, 3, 4, 5)
- Corrective waves: 3 waves against the trend (a, b, c)

Key rules:
- Wave 2 cannot retrace more than 100% of Wave 1
- Wave 3 cannot be the shortest of waves 1, 3, and 5
- Wave 4 cannot overlap with Wave 1 (except in diagonal triangles)
- Wave 3 is often the longest and strongest
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class WaveType(Enum):
    """Type of Elliott Wave."""
    IMPULSE = "impulse"  # Waves 1, 2, 3, 4, 5
    CORRECTIVE = "corrective"  # Waves a, b, c


class WaveLabel(Enum):
    """Labels for Elliott Waves."""
    WAVE_1 = "1"
    WAVE_2 = "2"
    WAVE_3 = "3"
    WAVE_4 = "4"
    WAVE_5 = "5"
    WAVE_A = "a"
    WAVE_B = "b"
    WAVE_C = "c"
    UNKNOWN = "?"


@dataclass
class Wave:
    """Represents a single Elliott Wave."""
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    wave_type: WaveType
    label: WaveLabel
    direction: str  # "up" or "down"
    confidence: float  # 0.0 to 1.0


class ElliottWaveDetector:
    """Detects Elliott Wave patterns in price data."""
    
    def __init__(
        self,
        min_wave_length: Optional[int] = None,
        max_wave_length: Optional[int] = None,
        retracement_threshold: float = 0.236
    ):
        """
        Initialize the Elliott Wave detector.
        
        Args:
            min_wave_length: Minimum number of data points for a wave (None = no restriction).
                            Practical filter, not from Elliott Wave theory. Must be set explicitly.
            max_wave_length: Maximum number of data points for a wave (None = no restriction).
                           Practical limit. Must be set explicitly.
            retracement_threshold: Minimum retracement percentage to consider a wave
                                  (0.236 = 23.6% Fibonacci level, based on Elliott Wave theory)
        
        Note: min_wave_length and max_wave_length are practical implementation filters,
        not requirements from Elliott Wave theory. Elliott Wave theory focuses on wave
        degrees (timeframes), internal structure, and relative proportions between waves,
        not absolute minimum durations. By default, no restrictions are applied - these
        must be set explicitly if filtering is desired.
        """
        self.min_wave_length = min_wave_length
        self.max_wave_length = max_wave_length
        self.retracement_threshold = retracement_threshold
    
    def detect_waves(
        self,
        data: pd.Series,
        min_confidence: float = 0.0,
        min_wave_size_ratio: float = 0.0,
        only_complete_patterns: bool = False
    ) -> List[Wave]:
        """
        Detect Elliott Wave patterns in the price data.
        
        Args:
            data: Time series data with datetime index
            min_confidence: Minimum confidence threshold (0.0 to 1.0) to include a wave
                           (practical filter, not from Elliott Wave theory)
            min_wave_size_ratio: Minimum wave size as ratio of price range (0.0 to 1.0)
                                (practical filter, default 0.05 = 5% to reduce noise,
                                not a requirement from Elliott Wave theory)
            only_complete_patterns: If True, only return complete 5-wave or 3-wave patterns
                                   (based on Elliott Wave theory - complete patterns are more reliable)
            
        Returns:
            List of detected waves
        
        Note: min_confidence and min_wave_size_ratio are practical filters to reduce noise
        and false positives. They are not requirements from Elliott Wave theory, which focuses
        on wave relationships, Fibonacci ratios, and internal structure rather than absolute
        size thresholds. Use the filter optimizer to find appropriate values for your data.
        """
        # Check minimum wave length if specified
        if self.min_wave_length is not None:
            if len(data) < self.min_wave_length * 2:
                return []
            # Scale min_wave_length based on data size to reduce noise in large datasets
            data_length = len(data)
            if data_length > 1000:
                # For large datasets, require longer waves
                adjusted_min_length = max(self.min_wave_length, int(data_length * 0.01))
            else:
                adjusted_min_length = self.min_wave_length
        else:
            adjusted_min_length = None
        
        # Find local extrema (peaks and troughs)
        # Use adjusted_min_length for window size if specified, otherwise use default
        window_size = adjusted_min_length if adjusted_min_length is not None else 3
        extrema = self._find_extrema(data, window=window_size)
        
        if len(extrema) < 5:
            return []
        
        # Identify wave patterns
        waves = self._identify_wave_patterns(data, extrema)
        
        # Filter waves
        if min_confidence > 0.0 or min_wave_size_ratio > 0.0 or only_complete_patterns:
            waves = self._filter_waves(data, waves, min_confidence, min_wave_size_ratio, only_complete_patterns)
        
        return waves
    
    def _filter_waves(
        self,
        data: pd.Series,
        waves: List[Wave],
        min_confidence: float,
        min_wave_size_ratio: float,
        only_complete_patterns: bool
    ) -> List[Wave]:
        """
        Filter waves based on criteria.
        
        Args:
            data: Time series data
            waves: List of detected waves
            min_confidence: Minimum confidence threshold
            min_wave_size_ratio: Minimum wave size as ratio of price range
            only_complete_patterns: Only return complete patterns
            
        Returns:
            Filtered list of waves
        """
        filtered = []
        
        # Calculate price range for size filtering
        price_range = data.max() - data.min()
        min_wave_size = price_range * min_wave_size_ratio if min_wave_size_ratio > 0 else 0
        
        if only_complete_patterns:
            # Group waves by pattern and only keep complete ones
            pattern_groups = self._group_waves_by_pattern(waves)
            for pattern_waves in pattern_groups.values():
                if self._is_complete_pattern(pattern_waves):
                    filtered.extend(pattern_waves)
        else:
            filtered = waves
        
        # Apply confidence, size, and length filters
        result = []
        for wave in filtered:
            # Confidence filter
            if wave.confidence < min_confidence:
                continue
            
            # Size filter
            if min_wave_size > 0:
                wave_size = abs(wave.end_price - wave.start_price)
                if wave_size < min_wave_size:
                    continue
            
            # Length filter (max_wave_length)
            if self.max_wave_length is not None:
                wave_length = wave.end_idx - wave.start_idx
                if wave_length > self.max_wave_length:
                    continue
            
            result.append(wave)
        
        return result
    
    def _group_waves_by_pattern(self, waves: List[Wave]) -> Dict[str, List[Wave]]:
        """Group waves that belong to the same pattern."""
        patterns = {}
        for wave in waves:
            # Create a pattern key based on wave type and nearby waves
            pattern_key = f"{wave.wave_type.value}_{wave.start_idx // 100}"
            if pattern_key not in patterns:
                patterns[pattern_key] = []
            patterns[pattern_key].append(wave)
        return patterns
    
    def _is_complete_pattern(self, waves: List[Wave]) -> bool:
        """Check if waves form a complete pattern (5-wave impulse or 3-wave corrective)."""
        if not waves:
            return False
        
        # Check for complete impulse pattern (1, 2, 3, 4, 5)
        impulse_labels = {WaveLabel.WAVE_1, WaveLabel.WAVE_2, WaveLabel.WAVE_3, WaveLabel.WAVE_4, WaveLabel.WAVE_5}
        wave_labels = {wave.label for wave in waves if wave.wave_type == WaveType.IMPULSE}
        if wave_labels == impulse_labels:
            return True
        
        # Check for complete corrective pattern (a, b, c)
        corrective_labels = {WaveLabel.WAVE_A, WaveLabel.WAVE_B, WaveLabel.WAVE_C}
        wave_labels = {wave.label for wave in waves if wave.wave_type == WaveType.CORRECTIVE}
        if wave_labels == corrective_labels:
            return True
        
        return False
    
    def _find_extrema(self, data: pd.Series, window: int = 3) -> List[Tuple[int, float, str]]:
        """
        Find local peaks and troughs in the data.
        
        Args:
            data: Time series data
            window: Window size for finding extrema
            
        Returns:
            List of (index, value, type) tuples where type is 'peak' or 'trough'
        """
        extrema = []
        values = data.values
        indices = data.index
        
        for i in range(window, len(values) - window):
            # Check for peak
            if all(values[i] >= values[i-j] for j in range(1, window+1)) and \
               all(values[i] >= values[i+j] for j in range(1, window+1)):
                extrema.append((i, values[i], 'peak'))
            
            # Check for trough
            elif all(values[i] <= values[i-j] for j in range(1, window+1)) and \
                 all(values[i] <= values[i+j] for j in range(1, window+1)):
                extrema.append((i, values[i], 'trough'))
        
        return extrema
    
    def _identify_wave_patterns(
        self,
        data: pd.Series,
        extrema: List[Tuple[int, float, str]]
    ) -> List[Wave]:
        """
        Identify Elliott Wave patterns from extrema.
        
        Args:
            data: Time series data
            extrema: List of local extrema
            
        Returns:
            List of identified waves
        """
        waves = []
        
        # Need at least 5 extrema for a 5-wave pattern
        if len(extrema) < 5:
            return waves
        
        # Try to identify impulse waves (1-2-3-4-5)
        impulse_waves = self._identify_impulse_waves(data, extrema)
        waves.extend(impulse_waves)
        
        # Try to identify corrective waves (a-b-c)
        corrective_waves = self._identify_corrective_waves(data, extrema, impulse_waves)
        waves.extend(corrective_waves)
        
        return waves
    
    def _identify_impulse_waves(
        self,
        data: pd.Series,
        extrema: List[Tuple[int, float, str]]
    ) -> List[Wave]:
        """
        Identify impulse waves (1, 2, 3, 4, 5).
        
        Args:
            data: Time series data
            extrema: List of local extrema
            
        Returns:
            List of impulse waves
        """
        waves = []
        
        # Look for 5-wave patterns
        for i in range(len(extrema) - 4):
            # Get 5 consecutive extrema
            e1, e2, e3, e4, e5 = extrema[i:i+5]
            
            # Determine overall trend direction
            trend_up = e5[1] > e1[1]
            
            if trend_up:
                # Upward trend: peaks should be higher, troughs should be higher
                if (e1[2] == 'trough' and e2[2] == 'peak' and e3[2] == 'trough' and
                    e4[2] == 'peak' and e5[2] == 'trough'):
                    # Pattern: trough-peak-trough-peak-trough (waves 1-2-3-4-5)
                    if self._validate_impulse_pattern(data, e1, e2, e3, e4, e5, trend_up):
                        waves.extend(self._create_impulse_waves(data, e1, e2, e3, e4, e5))
            else:
                # Downward trend: peaks should be lower, troughs should be lower
                if (e1[2] == 'peak' and e2[2] == 'trough' and e3[2] == 'peak' and
                    e4[2] == 'trough' and e5[2] == 'peak'):
                    # Pattern: peak-trough-peak-trough-peak (waves 1-2-3-4-5)
                    if self._validate_impulse_pattern(data, e1, e2, e3, e4, e5, trend_up):
                        waves.extend(self._create_impulse_waves(data, e1, e2, e3, e4, e5))
        
        return waves
    
    def _validate_impulse_pattern(
        self,
        data: pd.Series,
        e1: Tuple[int, float, str],
        e2: Tuple[int, float, str],
        e3: Tuple[int, float, str],
        e4: Tuple[int, float, str],
        e5: Tuple[int, float, str],
        trend_up: bool
    ) -> bool:
        """
        Validate that extrema form a valid Elliott Wave impulse pattern.
        
        Args:
            data: Time series data
            e1-e5: Five extrema points
            trend_up: Whether trend is upward
            
        Returns:
            True if pattern is valid
        """
        # Wave 2 cannot retrace more than 100% of Wave 1
        wave1_size = abs(e2[1] - e1[1])
        wave2_size = abs(e3[1] - e2[1])
        
        if wave2_size > wave1_size:
            return False
        
        # Wave 4 cannot overlap with Wave 1 (except in diagonal triangles)
        if trend_up:
            if e4[1] <= e1[1]:  # Wave 4 trough below Wave 1 start
                return False
        else:
            if e4[1] >= e1[1]:  # Wave 4 peak above Wave 1 start
                return False
        
        # Wave 3 cannot be the shortest of waves 1, 3, and 5
        wave3_size = abs(e4[1] - e3[1])
        wave5_size = abs(e5[1] - e4[1])
        
        wave_sizes = [wave1_size, wave3_size, wave5_size]
        if wave3_size == min(wave_sizes):
            return False
        
        return True
    
    def _create_impulse_waves(
        self,
        data: pd.Series,
        e1: Tuple[int, float, str],
        e2: Tuple[int, float, str],
        e3: Tuple[int, float, str],
        e4: Tuple[int, float, str],
        e5: Tuple[int, float, str]
    ) -> List[Wave]:
        """
        Create Wave objects for a 5-wave impulse pattern.
        
        Args:
            data: Time series data
            e1-e5: Five extrema points
            
        Returns:
            List of Wave objects
        """
        waves = []
        indices = data.index
        
        # Wave 1: e1 to e2
        waves.append(Wave(
            start_idx=e1[0],
            end_idx=e2[0],
            start_price=e1[1],
            end_price=e2[1],
            wave_type=WaveType.IMPULSE,
            label=WaveLabel.WAVE_1,
            direction="up" if e2[1] > e1[1] else "down",
            confidence=0.7
        ))
        
        # Wave 2: e2 to e3
        waves.append(Wave(
            start_idx=e2[0],
            end_idx=e3[0],
            start_price=e2[1],
            end_price=e3[1],
            wave_type=WaveType.IMPULSE,
            label=WaveLabel.WAVE_2,
            direction="down" if e2[1] > e1[1] else "up",
            confidence=0.7
        ))
        
        # Wave 3: e3 to e4
        waves.append(Wave(
            start_idx=e3[0],
            end_idx=e4[0],
            start_price=e3[1],
            end_price=e4[1],
            wave_type=WaveType.IMPULSE,
            label=WaveLabel.WAVE_3,
            direction="up" if e2[1] > e1[1] else "down",
            confidence=0.8
        ))
        
        # Wave 4: e4 to e5
        waves.append(Wave(
            start_idx=e4[0],
            end_idx=e5[0],
            start_price=e4[1],
            end_price=e5[1],
            wave_type=WaveType.IMPULSE,
            label=WaveLabel.WAVE_4,
            direction="down" if e2[1] > e1[1] else "up",
            confidence=0.7
        ))
        
        return waves
    
    def _identify_corrective_waves(
        self,
        data: pd.Series,
        extrema: List[Tuple[int, float, str]],
        impulse_waves: List[Wave]
    ) -> List[Wave]:
        """
        Identify corrective waves (a, b, c).
        
        Args:
            data: Time series data
            extrema: List of local extrema
            impulse_waves: Previously identified impulse waves
            
        Returns:
            List of corrective waves
        """
        waves = []
        
        # Look for 3-wave corrective patterns
        for i in range(len(extrema) - 2):
            e1, e2, e3 = extrema[i:i+3]
            
            # Check if this is a corrective pattern (opposite to trend)
            # Simple heuristic: if it's not part of an impulse wave, it might be corrective
            if self._is_corrective_pattern(data, e1, e2, e3):
                waves.extend(self._create_corrective_waves(data, e1, e2, e3))
        
        return waves
    
    def _is_corrective_pattern(
        self,
        data: pd.Series,
        e1: Tuple[int, float, str],
        e2: Tuple[int, float, str],
        e3: Tuple[int, float, str]
    ) -> bool:
        """
        Check if three extrema form a corrective pattern.
        
        Args:
            data: Time series data
            e1, e2, e3: Three extrema points
            
        Returns:
            True if pattern looks corrective
        """
        # Corrective waves typically retrace 38.2%, 50%, or 61.8% of the previous move
        move_size = abs(e3[1] - e1[1])
        retrace_size = abs(e2[1] - e1[1])
        
        if move_size == 0:
            return False
        
        retrace_ratio = retrace_size / move_size
        
        # Typical Fibonacci retracement levels
        return 0.236 <= retrace_ratio <= 0.786
    
    def _create_corrective_waves(
        self,
        data: pd.Series,
        e1: Tuple[int, float, str],
        e2: Tuple[int, float, str],
        e3: Tuple[int, float, str]
    ) -> List[Wave]:
        """
        Create Wave objects for a 3-wave corrective pattern.
        
        Args:
            data: Time series data
            e1, e2, e3: Three extrema points
            
        Returns:
            List of Wave objects
        """
        waves = []
        
        # Wave a: e1 to e2
        waves.append(Wave(
            start_idx=e1[0],
            end_idx=e2[0],
            start_price=e1[1],
            end_price=e2[1],
            wave_type=WaveType.CORRECTIVE,
            label=WaveLabel.WAVE_A,
            direction="up" if e2[1] > e1[1] else "down",
            confidence=0.6
        ))
        
        # Wave b: e2 to e3
        waves.append(Wave(
            start_idx=e2[0],
            end_idx=e3[0],
            start_price=e2[1],
            end_price=e3[1],
            wave_type=WaveType.CORRECTIVE,
            label=WaveLabel.WAVE_B,
            direction="down" if e2[1] > e1[1] else "up",
            confidence=0.6
        ))
        
        # Wave c: e3 continues (we'll mark it as ending at e3 for now)
        waves.append(Wave(
            start_idx=e2[0],
            end_idx=e3[0],
            start_price=e2[1],
            end_price=e3[1],
            wave_type=WaveType.CORRECTIVE,
            label=WaveLabel.WAVE_C,
            direction="up" if e2[1] > e1[1] else "down",
            confidence=0.5
        ))
        
        return waves
    
    def get_wave_segments(
        self,
        data: pd.Series,
        waves: List[Wave]
    ) -> Dict[str, List[Tuple[pd.Timestamp, float]]]:
        """
        Get wave segments for plotting.
        
        Args:
            data: Time series data
            waves: List of detected waves
            
        Returns:
            Dictionary mapping wave labels to lists of (timestamp, price) tuples
        """
        segments = {}
        indices = data.index
        
        for wave in waves:
            label = wave.label.value
            if label not in segments:
                segments[label] = []
            
            # Get data points for this wave
            wave_data = data.iloc[wave.start_idx:wave.end_idx+1]
            for idx, price in wave_data.items():
                segments[label].append((idx, price))
        
        return segments
