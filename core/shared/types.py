"""
Shared types for trading signal modules.

This module consolidates the SignalType enum and TradingSignal dataclass
that are used across multiple modules to avoid code duplication and
inconsistent type checking.
"""
import pandas as pd
from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Type of trading signal."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """
    Represents a trading signal with target prices.

    This is the unified signal class used across all trading modules.
    """
    signal_type: SignalType
    timestamp: pd.Timestamp
    price: float
    confidence: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""
    wave: Optional[Any] = None  # Wave object if from Elliott Wave detection
    source: str = "elliott"  # "elliott", "indicator", or "combined"
    indicator_confirmations: int = 0  # Number of indicators that confirmed this signal (for confidence-based sizing)

    # Indicator values at signal time (for analysis)
    rsi_value: Optional[float] = None
    ema_short: Optional[float] = None
    ema_long: Optional[float] = None
    macd_value: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Trend filter metadata
    trend_filter_active: bool = False  # Whether trend filter was applied to this signal
