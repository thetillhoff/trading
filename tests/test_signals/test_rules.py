"""Tests for pluggable signal rules (RSI, EMA, MACD) and rule engine."""
import pandas as pd
import numpy as np
from types import SimpleNamespace

from core.signals.rules import (
    RsiRule,
    EmaRule,
    MacdRule,
    get_technical_rules,
    apply_trend_filter,
)


def _row(overrides=None):
    """Default indicator row; overrides merged in."""
    d = {
        "price": 100.0,
        "rsi": 50.0,
        "rsi_oversold": False,
        "rsi_overbought": False,
        "ema_short": 99.0,
        "ema_long": 101.0,
        "ema_bullish_cross": False,
        "ema_bearish_cross": False,
        "macd_histogram": 0.0,
    }
    if overrides:
        d.update(overrides)
    return pd.Series(d)


class TestRsiRule:
    def test_exit_oversold_adds_buy_reason(self):
        rule = RsiRule()
        prev = _row({"rsi_oversold": True, "rsi": 20})
        row = _row({"rsi_oversold": False, "rsi": 30})
        buy, sell = rule.evaluate(row, prev, SimpleNamespace())
        assert len(buy) == 1
        assert "RSI exit oversold" in buy[0]
        assert len(sell) == 0

    def test_enter_overbought_adds_sell_reason(self):
        rule = RsiRule()
        prev = _row({"rsi_overbought": False, "rsi": 70})
        row = _row({"rsi_overbought": True, "rsi": 80})
        buy, sell = rule.evaluate(row, prev, SimpleNamespace())
        assert len(sell) == 1
        assert "overbought" in sell[0]
        assert len(buy) == 0

    def test_nan_rsi_returns_empty(self):
        rule = RsiRule()
        row = _row({"rsi": np.nan})
        prev = _row()
        buy, sell = rule.evaluate(row, prev, SimpleNamespace())
        assert buy == [] and sell == []


class TestEmaRule:
    def test_bullish_cross_adds_buy_reason(self):
        rule = EmaRule()
        row = _row({"ema_bullish_cross": True})
        prev = _row({"ema_bullish_cross": False})
        buy, sell = rule.evaluate(row, prev, SimpleNamespace())
        assert buy == ["EMA bullish cross"]
        assert sell == []

    def test_bearish_cross_adds_sell_reason(self):
        rule = EmaRule()
        row = _row({"ema_bearish_cross": True})
        buy, sell = rule.evaluate(row, _row(), SimpleNamespace())
        assert sell == ["EMA bearish cross"]
        assert buy == []


class TestMacdRule:
    def test_cross_above_zero_adds_buy_reason(self):
        rule = MacdRule()
        prev = _row({"macd_histogram": -0.5})
        row = _row({"macd_histogram": 0.1})
        buy, sell = rule.evaluate(row, prev, SimpleNamespace())
        assert buy == ["MACD cross above zero"]
        assert sell == []

    def test_cross_below_zero_adds_sell_reason(self):
        rule = MacdRule()
        prev = _row({"macd_histogram": 0.5})
        row = _row({"macd_histogram": -0.1})
        buy, sell = rule.evaluate(row, prev, SimpleNamespace())
        assert sell == ["MACD cross below zero"]
        assert buy == []


class TestGetTechnicalRules:
    def test_returns_only_enabled_rules(self):
        cfg = SimpleNamespace(use_rsi=True, use_ema=False, use_macd=True)
        rules = get_technical_rules(cfg)
        assert len(rules) == 2
        assert isinstance(rules[0], RsiRule)
        assert isinstance(rules[1], MacdRule)

    def test_empty_when_none_enabled(self):
        cfg = SimpleNamespace(use_rsi=False, use_ema=False, use_macd=False)
        rules = get_technical_rules(cfg)
        assert rules == []


class TestApplyTrendFilter:
    def test_no_filter_returns_unchanged(self):
        cfg = SimpleNamespace(use_trend_filter=False)
        row = _row()
        buy, sell = apply_trend_filter(["a"], ["b"], row, cfg)
        assert buy == ["a"]
        assert sell == ["b"]

    def test_bullish_trend_drops_sell_reasons(self):
        cfg = SimpleNamespace(use_trend_filter=True)
        row = _row({"ema_short": 102.0, "ema_long": 100.0})
        buy, sell = apply_trend_filter(["buy"], ["sell"], row, cfg)
        assert buy == ["buy"]
        assert sell == []

    def test_bearish_trend_drops_buy_reasons(self):
        cfg = SimpleNamespace(use_trend_filter=True)
        row = _row({"ema_short": 98.0, "ema_long": 100.0})
        buy, sell = apply_trend_filter(["buy"], ["sell"], row, cfg)
        assert buy == []
        assert sell == ["sell"]
