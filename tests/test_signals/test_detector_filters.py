"""Tests for detector_filters: filter by type, filter by quality, deduplicate."""
import pytest
import pandas as pd
from types import SimpleNamespace

from core.signals.detector_filters import (
    filter_signals_by_type,
    filter_signals_by_quality,
    deduplicate_signals,
)
from core.shared.types import SignalType, TradingSignal


def _signal(
    ts_date: str,
    signal_type: SignalType,
    confirmations: int = 0,
    confirmation_score=None,
    price: float = 100.0,
):
    return TradingSignal(
        signal_type=signal_type,
        timestamp=pd.Timestamp(ts_date),
        price=price,
        confidence=0.7,
        source="test",
        reasoning="",
        indicator_confirmations=confirmations,
        confirmation_score=confirmation_score,
    )


class TestFilterSignalsByType:
    def test_all_returns_unchanged(self):
        buy = _signal("2020-01-01", SignalType.BUY)
        sell = _signal("2020-01-02", SignalType.SELL)
        signals = [buy, sell]
        assert filter_signals_by_type(signals, "all") == signals

    def test_buy_keeps_only_buy(self):
        buy = _signal("2020-01-01", SignalType.BUY)
        sell = _signal("2020-01-02", SignalType.SELL)
        result = filter_signals_by_type([buy, sell], "buy")
        assert result == [buy]

    def test_sell_keeps_only_sell(self):
        buy = _signal("2020-01-01", SignalType.BUY)
        sell = _signal("2020-01-02", SignalType.SELL)
        result = filter_signals_by_type([buy, sell], "sell")
        assert result == [sell]


class TestFilterSignalsByQuality:
    def test_min_confirmations_filters(self):
        config = SimpleNamespace(min_confirmations=2, min_certainty=None)
        signals = [
            _signal("2020-01-01", SignalType.BUY, 0),
            _signal("2020-01-02", SignalType.BUY, 1),
            _signal("2020-01-03", SignalType.BUY, 2),
            _signal("2020-01-04", SignalType.BUY, 3),
        ]
        result = filter_signals_by_quality(signals, config)
        assert len(result) == 2
        assert all(getattr(s, "indicator_confirmations", 0) >= 2 for s in result)

    def test_min_certainty_filters(self):
        config = SimpleNamespace(min_confirmations=None, min_certainty=0.66)
        low = _signal("2020-01-01", SignalType.BUY, 1, confirmation_score=0.5)
        high = _signal("2020-01-02", SignalType.BUY, 2, confirmation_score=0.7)
        result = filter_signals_by_quality([low, high], config)
        assert len(result) == 1
        assert result[0].confirmation_score == 0.7

    def test_min_certainty_uses_count_when_no_score(self):
        config = SimpleNamespace(min_confirmations=None, min_certainty=0.66)
        one = _signal("2020-01-01", SignalType.BUY, 1, confirmation_score=None)
        two = _signal("2020-01-02", SignalType.BUY, 2, confirmation_score=None)
        result = filter_signals_by_quality([one, two], config)
        assert len(result) == 1
        assert result[0].indicator_confirmations == 2

    def test_no_filter_when_both_none(self):
        config = SimpleNamespace(min_confirmations=None, min_certainty=None)
        signals = [
            _signal("2020-01-01", SignalType.BUY, 0),
            _signal("2020-01-02", SignalType.BUY, 3),
        ]
        result = filter_signals_by_quality(signals, config)
        assert len(result) == 2


class TestDeduplicateSignals:
    def test_same_day_same_type_different_time_both_kept(self):
        # Different timestamps on same day should be kept (multiple opportunities)
        a = _signal("2020-01-01 09:00", SignalType.BUY)
        b = _signal("2020-01-01 15:00", SignalType.BUY)
        result = deduplicate_signals([a, b])
        assert len(result) == 2
        
    def test_exact_duplicate_removed(self):
        # Exact same timestamp, type, and price should deduplicate
        a = _signal("2020-01-01 09:00", SignalType.BUY, price=100.0)
        b = _signal("2020-01-01 09:00", SignalType.BUY, price=100.0)
        result = deduplicate_signals([a, b])
        assert len(result) == 1
        assert result[0].timestamp == a.timestamp

    def test_same_day_different_type_both_kept(self):
        buy = _signal("2020-01-01 09:00", SignalType.BUY)
        sell = _signal("2020-01-01 15:00", SignalType.SELL)
        result = deduplicate_signals([buy, sell])
        assert len(result) == 2

    def test_different_days_both_kept(self):
        a = _signal("2020-01-01", SignalType.BUY)
        b = _signal("2020-01-02", SignalType.BUY)
        result = deduplicate_signals([a, b])
        assert len(result) == 2
