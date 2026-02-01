"""
Signal filtering and deduplication for the detector pipeline.

Pure functions: filter by type, filter by quality (min_confirmations / min_certainty),
and deduplicate by (date, signal_type). Used by SignalDetector after signal generation.
"""
from typing import List, Any

from ..shared.types import SignalType, TradingSignal


def filter_signals_by_type(
    signals: List[TradingSignal],
    signal_types: str,
) -> List[TradingSignal]:
    """
    Keep only signals of the requested type.

    Args:
        signals: List of trading signals
        signal_types: "buy", "sell", or "all"

    Returns:
        Filtered list (same list if signal_types == "all")
    """
    if signal_types == "buy":
        return [s for s in signals if s.signal_type == SignalType.BUY]
    if signal_types == "sell":
        return [s for s in signals if s.signal_type == SignalType.SELL]
    return signals


def filter_signals_by_quality(
    signals: List[TradingSignal],
    config: Any,
) -> List[TradingSignal]:
    """
    Keep only signals meeting min_confirmations and min_certainty (if set on config).

    Args:
        signals: List of trading signals
        config: Object with optional min_confirmations, min_certainty attributes

    Returns:
        Filtered list; unchanged if both min_confirmations and min_certainty are None
    """
    min_confirmations = getattr(config, "min_confirmations", None)
    min_certainty = getattr(config, "min_certainty", None)
    if min_confirmations is None and min_certainty is None:
        return signals
    filtered = []
    for s in signals:
        ok_conf = min_confirmations is None or getattr(
            s, "indicator_confirmations", 0
        ) >= min_confirmations
        effective_certainty = (
            s.confirmation_score
            if getattr(s, "confirmation_score", None) is not None
            else getattr(s, "indicator_confirmations", 0) / 3.0
        )
        ok_cert = min_certainty is None or effective_certainty >= min_certainty
        if ok_conf and ok_cert:
            filtered.append(s)
    return filtered


def deduplicate_signals(signals: List[TradingSignal]) -> List[TradingSignal]:
    """
    Remove duplicate signals on the same day (first occurrence kept per (date, signal_type)).

    Args:
        signals: List of trading signals (assumed sorted by timestamp)

    Returns:
        Deduplicated list preserving order
    """
    seen = set()
    unique = []
    for sig in signals:
        key = (sig.timestamp.date(), sig.signal_type)
        if key not in seen:
            seen.add(key)
            unique.append(sig)
    return unique
