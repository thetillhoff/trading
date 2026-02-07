"""
Indicator calculation module.

Provides all trading indicators:
- Technical indicators (RSI, EMA, MACD)
- Elliott Wave pattern detection
- Disk-based caching for cross-run indicator reuse

All indicators follow a unified interface for calculation and signal generation.
"""
from .technical import (
    TechnicalIndicators,
    IndicatorValues,
    check_buy_confirmation,
    check_sell_confirmation,
    confirmation_weighted_score,
)
from .elliott_wave import ElliottWaveDetector, Wave, WaveType, WaveLabel
from .disk_cache import (
    compute_indicator_cache_key,
    get_cached_indicator,
    save_cached_indicator,
    get_cache_stats,
    clear_cache,
)

__all__ = [
    'TechnicalIndicators',
    'IndicatorValues',
    'check_buy_confirmation',
    'check_sell_confirmation',
    'confirmation_weighted_score',
    'ElliottWaveDetector',
    'Wave',
    'WaveType',
    'WaveLabel',
    'compute_indicator_cache_key',
    'get_cached_indicator',
    'save_cached_indicator',
    'get_cache_stats',
    'clear_cache',
]
