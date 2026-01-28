"""
Tests for SignalDetector: Elliott Wave (regular + inverted) and technical indicators.

Covers:
- _invert_price_data: formula (max + min - price) and idempotence of double-invert.
- Inverted Elliott Wave: computed correctly, source elliott_inverted/combined_inverted,
  differ from normal EW on the same data.
- Regular Elliott Wave: source elliott/combined, reasoning refers to Wave; signal set
  differs from RSI/EMA/MACD-only on the same data.
- RSI, EMA, MACD: each single-indicator run produces signals with source 'indicator'
  and indicator name in reasoning; pairwise different (timestamp, type) sets.
"""
import pytest
import pandas as pd
import numpy as np

from core.signals.detector import SignalDetector
from core.signals.config import SignalConfig
from core.shared.types import SignalType


def _config(use_elliott_wave: bool, use_elliott_wave_inverted: bool, use_elliott_wave_inverted_exit: bool = False, min_confidence: float = 0.0, min_confidence_inverted: float = 0.0, min_wave_size: float = 0.0, min_wave_size_inverted: float = 0.0):
    """Minimal SignalConfig for EW / inverted EW only."""
    return SignalConfig(
        use_elliott_wave=use_elliott_wave,
        use_elliott_wave_inverted=use_elliott_wave_inverted,
        use_elliott_wave_inverted_exit=use_elliott_wave_inverted_exit,
        use_rsi=False,
        use_ema=False,
        use_macd=False,
        min_confidence=min_confidence,
        min_wave_size=min_wave_size,
        min_confidence_inverted=min_confidence_inverted,
        min_wave_size_inverted=min_wave_size_inverted,
    )


def _technical_config(use_rsi: bool, use_ema: bool, use_macd: bool):
    """SignalConfig for technical indicators only (no Elliott Wave)."""
    return SignalConfig(
        use_elliott_wave=False,
        use_elliott_wave_inverted=False,
        use_rsi=use_rsi,
        use_ema=use_ema,
        use_macd=use_macd,
    )


class TestInvertPriceData:
    """Inverse price computation: max + min - price."""

    def test_invert_formula(self):
        """_invert_price_data uses max + min - price and flips direction."""
        config = _config(use_elliott_wave=False, use_elliott_wave_inverted=True)
        detector = SignalDetector(config)
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2020-01-01", periods=5, freq="D"))
        inverted = detector._invert_price_data(data)
        assert data.index.equals(inverted.index)
        expected = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=data.index)
        pd.testing.assert_series_equal(inverted, expected)

    def test_double_invert_recovers_original(self):
        """Applying _invert_price_data twice returns the original series."""
        config = _config(use_elliott_wave=False, use_elliott_wave_inverted=True)
        detector = SignalDetector(config)
        data = pd.Series(100 + 10 * np.sin(np.linspace(0, 4 * np.pi, 200)) + 0.05 * np.arange(200), index=pd.date_range("2020-01-01", periods=200, freq="D"))
        once = detector._invert_price_data(data)
        twice = detector._invert_price_data(once)
        pd.testing.assert_series_equal(twice, data)


@pytest.fixture
def asymmetric_prices():
    """Asymmetric price series with drift so normal and inverted EW see different structure."""
    t = np.arange(200)
    values = 100.0 + 10.0 * np.sin(t * 0.1) + 0.1 * t
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    return pd.Series(values, index=dates)


@pytest.fixture
def technical_prices():
    """Deterministic price series for technical-indicator signals (length >= INDICATOR_WARMUP_PERIOD)."""
    t = np.arange(200)
    values = 100.0 + 10.0 * np.sin(t * 0.1) + 0.1 * t
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    return pd.Series(values, index=dates)


class TestInvertedElliottWave:
    """Inverted Elliott Wave is computed and differs from normal Elliott Wave."""

    def test_inverted_ew_signals_differ_from_normal_on_same_data(self, asymmetric_prices):
        """Normal EW signals and inverted EW signals on the same data are not identical."""
        data = asymmetric_prices
        config_normal = _config(use_elliott_wave=True, use_elliott_wave_inverted=False, min_confidence=0.0, min_wave_size=0.0)
        config_inverted = _config(use_elliott_wave=False, use_elliott_wave_inverted=True, min_confidence_inverted=0.0, min_wave_size_inverted=0.0)

        det_normal = SignalDetector(config_normal)
        det_inverted = SignalDetector(config_inverted)

        normal_signals = det_normal._get_elliott_wave_signals(data, None)
        inverted_signals = det_inverted._get_inverted_elliott_wave_signals(data, None)

        def key(s):
            return (s.timestamp, s.signal_type.name)

        normal_set = {key(s) for s in normal_signals}
        inverted_set = {key(s) for s in inverted_signals}
        assert normal_set != inverted_set, "Normal and inverted Elliott Wave must produce different (timestamp, type) sets on asymmetric data."

    def test_inverted_ew_signals_have_source_elliott_inverted(self, asymmetric_prices):
        """Inverted EW signals use source 'elliott_inverted' (or 'combined_inverted' when confirmed)."""
        data = asymmetric_prices
        config = _config(use_elliott_wave=False, use_elliott_wave_inverted=True, min_confidence_inverted=0.0, min_wave_size_inverted=0.0)
        det = SignalDetector(config)
        signals = det._get_inverted_elliott_wave_signals(data, None)
        for s in signals:
            assert s.source in ("elliott_inverted", "combined_inverted")

    def test_inverted_ew_exit_sells_have_close_long_only(self, asymmetric_prices):
        """When use_elliott_wave_inverted_exit=True, SELLs from inverted EW have close_long_only=True."""
        data = asymmetric_prices
        config = _config(use_elliott_wave=False, use_elliott_wave_inverted=False, use_elliott_wave_inverted_exit=True, min_confidence_inverted=0.0, min_wave_size_inverted=0.0)
        det = SignalDetector(config)
        signals = det.detect_signals(data)
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        for s in sell_signals:
            assert getattr(s, 'close_long_only', False) is True, "Inverted exit SELLs must have close_long_only=True"

    def test_inverted_ew_open_short_sells_have_no_close_long_only(self, asymmetric_prices):
        """When use_elliott_wave_inverted=True and use_elliott_wave_inverted_exit=False, SELLs have close_long_only=False."""
        data = asymmetric_prices
        config = _config(use_elliott_wave=False, use_elliott_wave_inverted=True, use_elliott_wave_inverted_exit=False, min_confidence_inverted=0.0, min_wave_size_inverted=0.0)
        det = SignalDetector(config)
        signals = det.detect_signals(data)
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        for s in sell_signals:
            assert getattr(s, 'close_long_only', False) is False


def _signal_set(signals):
    """(timestamp, signal_type.name) set for comparison."""
    return {(s.timestamp, s.signal_type.name) for s in signals}


class TestRegularElliottWave:
    """Regular (normal) Elliott Wave is computed correctly and differs from inverted/technical."""

    def test_regular_ew_signals_computed_correctly(self, asymmetric_prices):
        """Regular EW signals have source in ('elliott', 'combined') and reasoning refers to Elliott Wave."""
        data = asymmetric_prices
        config = _config(use_elliott_wave=True, use_elliott_wave_inverted=False, min_confidence=0.0, min_wave_size=0.0)
        det = SignalDetector(config)
        signals = det._get_elliott_wave_signals(data, None)
        for s in signals:
            assert s.source in ("elliott", "combined")
            assert "Wave" in s.reasoning or "correction" in s.reasoning

    def test_regular_ew_signal_set_differs_from_technical_only(self, technical_prices):
        """Regular EW-only signal set differs from RSI-only, EMA-only, and MACD-only on the same data."""
        data = technical_prices
        config_ew = _config(use_elliott_wave=True, use_elliott_wave_inverted=False, min_confidence=0.0, min_wave_size=0.0)
        det_ew = SignalDetector(config_ew)
        ew_signals = det_ew.detect_signals(data)
        ew_set = _signal_set(ew_signals)

        for use_rsi, use_ema, use_macd in [(True, False, False), (False, True, False), (False, False, True)]:
            det_tech = SignalDetector(_technical_config(use_rsi, use_ema, use_macd))
            tech_signals = det_tech.detect_signals(data)
            tech_set = _signal_set(tech_signals)
            assert ew_set != tech_set, "Regular EW-only and single technical-indicator runs must produce different (timestamp, type) sets."


class TestTechnicalIndicatorSignals:
    """RSI, EMA, MACD detector signals are computed correctly and produce pairwise different sets."""

    def test_rsi_only_signals_computed_correctly(self, technical_prices):
        """RSI-only signals have source 'indicator' and 'RSI' in reasoning."""
        det = SignalDetector(_technical_config(use_rsi=True, use_ema=False, use_macd=False))
        signals = det.detect_signals(technical_prices)
        for s in signals:
            assert s.source == "indicator"
            assert "RSI" in s.reasoning

    def test_ema_only_signals_computed_correctly(self, technical_prices):
        """EMA-only signals have source 'indicator' and 'EMA' in reasoning."""
        det = SignalDetector(_technical_config(use_rsi=False, use_ema=True, use_macd=False))
        signals = det.detect_signals(technical_prices)
        for s in signals:
            assert s.source == "indicator"
            assert "EMA" in s.reasoning

    def test_macd_only_signals_computed_correctly(self, technical_prices):
        """MACD-only signals have source 'indicator' and 'MACD' in reasoning."""
        det = SignalDetector(_technical_config(use_rsi=False, use_ema=False, use_macd=True))
        signals = det.detect_signals(technical_prices)
        for s in signals:
            assert s.source == "indicator"
            assert "MACD" in s.reasoning

    def test_single_indicator_configs_produce_pairwise_different_signal_sets(self, technical_prices):
        """RSI-only, EMA-only, and MACD-only produce pairwise different (timestamp, type) sets."""
        data = technical_prices
        rsi_set = _signal_set(SignalDetector(_technical_config(True, False, False)).detect_signals(data))
        ema_set = _signal_set(SignalDetector(_technical_config(False, True, False)).detect_signals(data))
        macd_set = _signal_set(SignalDetector(_technical_config(False, False, True)).detect_signals(data))
        assert rsi_set != ema_set
        assert rsi_set != macd_set
        assert ema_set != macd_set
