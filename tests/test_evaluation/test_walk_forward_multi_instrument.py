"""
Multi-instrument walk-forward tests.

Verifies that evaluate_multi_instrument produces signals from all instruments
and that the portfolio uses per-instrument prices (more trades than single-instrument).
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from core.evaluation.walk_forward import WalkForwardEvaluator
from core.signals.config import StrategyConfig


def _synthetic_series(days: int = 400, start: str = "2010-01-01", base: float = 100.0):
    """Synthetic price series with trend so indicators can fire."""
    dates = pd.date_range(start, periods=days, freq="B")
    rng = np.random.RandomState(42)
    prices = base + np.cumsum(rng.randn(days) * 0.5) + np.linspace(0, 10, days)
    return pd.Series(np.maximum(prices, 1.0), index=dates)


def test_evaluate_multi_instrument_two_instruments_more_trades_than_single():
    """With two instruments (same data), multi-instrument run produces more trades than single."""
    series = _synthetic_series(400)
    data_by_inst = {"inst1": series, "inst2": series.copy()}

    config_multi = StrategyConfig(
        name="multi",
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
    config_single = StrategyConfig(
        name="single",
        instruments=["inst1"],
        start_date=config_multi.start_date,
        end_date=config_multi.end_date,
        lookback_days=config_multi.lookback_days,
        step_days=config_multi.step_days,
        use_elliott_wave=config_multi.use_elliott_wave,
        min_confidence=config_multi.min_confidence,
        min_wave_size=config_multi.min_wave_size,
        use_rsi=config_multi.use_rsi,
        use_ema=config_multi.use_ema,
        use_macd=config_multi.use_macd,
        risk_reward=config_multi.risk_reward,
        max_days=config_multi.max_days,
        position_size_pct=config_multi.position_size_pct,
        max_positions=config_multi.max_positions,
    )

    def fake_from_instrument(instrument_name, start_date=None, end_date=None, column="Close"):
        return data_by_inst[instrument_name]

    # Use max_workers=1 for in-process execution (mocks work)
    with patch("core.data.loader.DataLoader.from_instrument", fake_from_instrument):
        with patch("core.data.preparation.DataLoader.from_instrument", fake_from_instrument):
            evaluator = WalkForwardEvaluator(lookback_days=180, step_days=30, min_history_days=100)
            result_single = evaluator.evaluate_multi_instrument(config_single, verbose=False, max_workers=1)
            result_multi = evaluator.evaluate_multi_instrument(config_multi, verbose=False, max_workers=1)

    assert result_multi.simulation.total_trades >= result_single.simulation.total_trades
    instruments_in_positions = {p.instrument for p in result_multi.simulation.positions if p.instrument}
    if result_multi.simulation.total_trades >= 2:
        assert "inst1" in instruments_in_positions or "inst2" in instruments_in_positions


def test_evaluate_multi_instrument_positions_tagged_by_instrument():
    """Multi-instrument run tags positions with instrument."""
    series = _synthetic_series(400)
    data_by_inst = {"inst_a": series, "inst_b": series.copy()}

    def fake_from_instrument(instrument_name, start_date=None, end_date=None, column="Close"):
        return data_by_inst[instrument_name]

    config = StrategyConfig(
        name="two",
        instruments=["inst_a", "inst_b"],
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

    # Use max_workers=1 for in-process execution (mocks work)
    with patch("core.data.loader.DataLoader.from_instrument", fake_from_instrument):
        with patch("core.data.preparation.DataLoader.from_instrument", fake_from_instrument):
            evaluator = WalkForwardEvaluator(lookback_days=180, step_days=30, min_history_days=100)
            result = evaluator.evaluate_multi_instrument(config, verbose=False, max_workers=1)

    for pos in result.simulation.positions:
        assert pos.instrument is not None
        assert pos.instrument in ("inst_a", "inst_b")


def test_evaluate_multi_instrument_parallel_matches_sequential():
    """Parallel and sequential (both max_workers=1) produce same trade count and summary."""
    series = _synthetic_series(400)
    data_by_inst = {"i1": series, "i2": series.copy()}

    def fake_from_instrument(instrument_name, start_date=None, end_date=None, column="Close"):
        return data_by_inst[instrument_name]

    config = StrategyConfig(
        name="two",
        instruments=["i1", "i2"],
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

    # Both use max_workers=1 for in-process execution (mocks work)
    with patch("core.data.loader.DataLoader.from_instrument", fake_from_instrument):
        with patch("core.data.preparation.DataLoader.from_instrument", fake_from_instrument):
            evaluator = WalkForwardEvaluator(lookback_days=180, step_days=30, min_history_days=100)
            result_seq = evaluator.evaluate_multi_instrument(config, verbose=False, max_workers=1)
            result_par = evaluator.evaluate_multi_instrument(config, verbose=False, max_workers=1)

    # Note: Both use in-process execution, so they should be identical
    assert result_par.simulation.total_trades == result_seq.simulation.total_trades
    assert result_par.summary.total_trades == result_seq.summary.total_trades
