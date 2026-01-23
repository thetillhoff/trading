"""
Shared types and defaults for the trading system.

This module provides:
- SignalType enum and TradingSignal dataclass
- Centralized default values for all indicator parameters
"""
from .types import SignalType, TradingSignal
from .defaults import (
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    EMA_SHORT_PERIOD, EMA_LONG_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ELLIOTT_MIN_CONFIDENCE, ELLIOTT_MIN_WAVE_SIZE,
    RISK_REWARD_RATIO, POSITION_SIZE_PCT, MAX_POSITIONS,
    USE_CONFIDENCE_SIZING, CONFIDENCE_SIZE_MULTIPLIER,
    STEP_DAYS, LOOKBACK_DAYS,
)

__all__ = [
    'SignalType',
    'TradingSignal',
    'RSI_PERIOD', 'RSI_OVERSOLD', 'RSI_OVERBOUGHT',
    'EMA_SHORT_PERIOD', 'EMA_LONG_PERIOD',
    'MACD_FAST', 'MACD_SLOW', 'MACD_SIGNAL',
    'ELLIOTT_MIN_CONFIDENCE', 'ELLIOTT_MIN_WAVE_SIZE',
    'RISK_REWARD_RATIO', 'POSITION_SIZE_PCT', 'MAX_POSITIONS',
    'USE_CONFIDENCE_SIZING', 'CONFIDENCE_SIZE_MULTIPLIER',
    'STEP_DAYS', 'LOOKBACK_DAYS',
]
