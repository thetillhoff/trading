"""
Elliott Wave types: wave type, label, and Wave dataclass.

Extracted for reuse so other detectors or strategies can use Wave/WaveType/WaveLabel
without pulling in ElliottWaveDetector. New wave-based strategies can import these types.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
