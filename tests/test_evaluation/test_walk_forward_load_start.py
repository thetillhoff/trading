"""
Tests for "feed history before timeframe": when start_date is set, loader is called with
load_start = start_date - lookback_days so day-1 evaluation has full indicator history.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from core.evaluation.walk_forward import WalkForwardEvaluator
from core.signals.config import StrategyConfig


def _synthetic_series(start: str, end: str, base: float = 100.0) -> pd.Series:
    """Synthetic price series with datetime index."""
    dates = pd.date_range(start=start, end=end, freq="B")
    n = len(dates)
    rng = np.random.RandomState(42)
    prices = base + np.cumsum(rng.randn(n) * 0.5) + np.linspace(0, 10, n)
    return pd.Series(np.maximum(prices, 1.0), index=dates)


def test_single_instrument_load_start_requested_when_start_date_set():
    """When config.start_date is set, DataLoader.from_instrument is called with start_date = config.start_date - lookback_days."""
    config = StrategyConfig(
        name="test",
        instruments=["djia"],
        start_date="2008-01-01",
        end_date="2008-06-01",
        lookback_days=365,
        step_days=90,
        use_elliott_wave=True,
        min_confidence=0.3,
        min_wave_size=0.01,
        use_rsi=True,
        use_ema=True,
        use_macd=True,
        risk_reward=2.0,
        max_days=60,
        position_size_pct=0.2,
        max_positions=5,
    )
    # Data from 2000 so we have full lookback before 2008-01-01
    full_series = _synthetic_series("2000-01-01", "2012-06-01")
    recorded_calls = []

    def fake_from_instrument(instrument_name, start_date=None, end_date=None, column="Close"):
        recorded_calls.append({"start_date": start_date, "end_date": end_date})
        # Return data from requested start (or full series); filter in-memory for test
        if start_date is not None and end_date is not None:
            mask = (full_series.index >= pd.Timestamp(start_date)) & (full_series.index <= pd.Timestamp(end_date))
            return full_series.loc[mask].copy()
        return full_series.copy()

    with patch("core.data.loader.DataLoader.from_instrument", fake_from_instrument):
        evaluator = WalkForwardEvaluator(lookback_days=365, step_days=1, min_history_days=100)
        result = evaluator.evaluate_multi_instrument(config, verbose=False)

    assert len(recorded_calls) == 1
    load_start = recorded_calls[0]["start_date"]
    expected_load_start = (pd.Timestamp("2008-01-01") - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    assert load_start == expected_load_start
    assert result.evaluation_start_date == pd.Timestamp("2008-01-01")
    assert result.evaluation_end_date == pd.Timestamp("2008-06-01")


def test_multi_instrument_load_start_requested_when_start_date_set():
    """When config.start_date is set, each instrument load uses start_date = config.start_date - lookback_days."""
    config = StrategyConfig(
        name="test",
        instruments=["inst1", "inst2"],
        start_date="2010-06-01",
        end_date="2011-06-01",
        lookback_days=180,
        step_days=30,
        use_elliott_wave=True,
        min_confidence=0.3,
        min_wave_size=0.01,
        use_rsi=True,
        use_ema=True,
        use_macd=True,
        risk_reward=2.0,
        max_days=60,
        position_size_pct=0.2,
        max_positions=5,
    )
    series = _synthetic_series("2000-01-01", "2012-06-01")
    data_by_inst = {"inst1": series, "inst2": series.copy()}
    recorded_calls = []

    def fake_from_instrument(instrument_name, start_date=None, end_date=None, column="Close"):
        recorded_calls.append({"instrument": instrument_name, "start_date": start_date})
        return data_by_inst[instrument_name]

    with patch("core.data.loader.DataLoader.from_instrument", fake_from_instrument):
        evaluator = WalkForwardEvaluator(lookback_days=180, step_days=30, min_history_days=100)
        result = evaluator.evaluate_multi_instrument(config, verbose=False)

    expected_load_start = (pd.Timestamp("2010-06-01") - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    assert len(recorded_calls) == 2
    for call in recorded_calls:
        assert call["start_date"] == expected_load_start
    assert result.evaluation_start_date == pd.Timestamp("2010-06-01")
    assert result.evaluation_end_date == pd.Timestamp("2011-06-01")


def test_warn_when_full_lookback_not_available(capsys):
    """When loader returns data that starts after load_start, evaluation still runs and a warning is printed."""
    config = StrategyConfig(
        name="test",
        instruments=["djia"],
        start_date="2008-01-01",
        end_date="2008-06-01",
        lookback_days=365,
        step_days=90,
        use_elliott_wave=True,
        min_confidence=0.3,
        min_wave_size=0.01,
        use_rsi=True,
        use_ema=True,
        use_macd=True,
        risk_reward=2.0,
        max_days=60,
        position_size_pct=0.2,
        max_positions=5,
    )
    # Data only from 2008-01-01 onward (no extra history)
    short_series = _synthetic_series("2008-01-01", "2008-06-01")

    def fake_from_instrument(instrument_name, start_date=None, end_date=None, column="Close"):
        return short_series.copy()

    with patch("core.data.loader.DataLoader.from_instrument", fake_from_instrument):
        evaluator = WalkForwardEvaluator(lookback_days=365, step_days=1, min_history_days=100)
        result = evaluator.evaluate_multi_instrument(config, verbose=True)

    out = capsys.readouterr()
    assert "Requested 365 days of history" in out.out or "Requested 365 days of history" in out.err
    assert "only" in out.out or "only" in out.err
    assert "reduced lookback" in out.out or "reduced lookback" in out.err
    assert result is not None
    # When full lookback is not available, evaluate() may push start forward (min_history_days)
    assert result.evaluation_start_date >= pd.Timestamp("2008-01-01")
    assert result.evaluation_end_date == pd.Timestamp("2008-06-01")
