"""
Indicator calculation module.

Provides all trading indicators:
- Technical indicators (RSI, EMA, MACD)
- Elliott Wave pattern detection

All indicators follow a unified interface for calculation and signal generation.
"""
from .technical import TechnicalIndicators, IndicatorValues, check_buy_confirmation, check_sell_confirmation
from .elliott_wave import ElliottWaveDetector, Wave, WaveType, WaveLabel

__all__ = [
    'TechnicalIndicators',
    'IndicatorValues',
    'check_buy_confirmation',
    'check_sell_confirmation',
    'ElliottWaveDetector',
    'Wave',
    'WaveType',
    'WaveLabel',
]
