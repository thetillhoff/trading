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
from typing import Optional

from core.signals.detector import SignalDetector
from core.signals.config import SignalConfig
from core.shared.types import SignalType, TradingSignal
from core.indicators.elliott_wave import Wave, WaveType, WaveLabel
from types import SimpleNamespace


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

        normal_signals, _ = det_normal._get_elliott_wave_signals(data, None)
        inverted_signals, _ = det_inverted._get_inverted_elliott_wave_signals(data, None)

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
        signals, _ = det._get_inverted_elliott_wave_signals(data, None)
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
        signals, _ = det._get_elliott_wave_signals(data, None)
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


class TestRegimeDetection:
    def test_invert_signals_in_bull_is_honored_for_ew_signals(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = pd.Series(np.linspace(100, 150, 50), index=dates)

        wave = Wave(
            start_idx=0,
            end_idx=10,
            start_price=float(prices.iloc[0]),
            end_price=float(prices.iloc[10]),
            wave_type=WaveType.IMPULSE,
            label=WaveLabel.WAVE_2,
            direction="up",
            confidence=1.0,
        )

        # Regime on, but inversion disabled => keep original BUY on BULL
        cfg_no_invert = SimpleNamespace(
            use_regime_detection=True,
            invert_signals_in_bull=False,
            use_rsi=False,
            use_ema=False,
            use_macd=False,
        )
        det_no_invert = SignalDetector(cfg_no_invert)
        sig_no_invert = det_no_invert._wave_to_signal(wave, prices, indicator_df=None, market_regime="BULL")
        assert sig_no_invert is not None
        assert sig_no_invert.signal_type == SignalType.BUY

        # Regime on, inversion enabled => invert BUY->SELL on BULL
        cfg_invert = SimpleNamespace(
            use_regime_detection=True,
            invert_signals_in_bull=True,
            use_rsi=False,
            use_ema=False,
            use_macd=False,
        )
        det_invert = SignalDetector(cfg_invert)
        sig_invert = det_invert._wave_to_signal(wave, prices, indicator_df=None, market_regime="BULL")
        assert sig_invert is not None
        assert sig_invert.signal_type == SignalType.SELL

    def test_regime_mode_dispatches_to_correct_classifier(self):
        ts = pd.Timestamp("2020-01-10")
        indicator_df = pd.DataFrame(
            {
                "price": [100.0],
                "adx": [50.0],
                "ma_slope": [1.0],
                "volatility_20": [0.01],
            },
            index=[ts],
        )
        prices = pd.Series([100.0], index=[ts])

        cfg = SimpleNamespace(
            use_regime_detection=True,
            regime_mode="trend_vol",
        )
        det = SignalDetector(cfg)

        det._detect_market_regime_trend_vol = lambda df, t: "CALLED_TREND_VOL"
        det._detect_market_regime_adx_ma = lambda df, t: "CALLED_ADX_MA"

        assert det._detect_market_regime(prices, indicator_df, ts) == "CALLED_TREND_VOL"

        cfg.regime_mode = "adx_ma"
        assert det._detect_market_regime(prices, indicator_df, ts) == "CALLED_ADX_MA"


class TestVolatilityFilter:
    """Volatility filter: when use_volatility_filter and volatility_20 > volatility_max, confirmation is False."""

    def test_volatility_filter_rejects_high_vol(self):
        """When volatility_20 > volatility_max, _check_indicator_confirmation returns (False, ..., 0, ...)."""
        config = SignalConfig(
            use_elliott_wave=False,
            use_rsi=True,
            use_ema=False,
            use_macd=False,
            use_volatility_filter=True,
            volatility_max=0.01,
        )
        detector = SignalDetector(config)
        ts = pd.Timestamp("2020-06-15")
        # Row with high vol (0.02 > 0.01) but RSI would confirm buy
        row = {
            "price": 100.0,
            "rsi": 30.0,
            "rsi_oversold": True,
            "rsi_overbought": False,
            "ema_short": 99.0,
            "ema_long": 98.0,
            "price_above_ema_short": True,
            "price_above_ema_long": True,
            "ema_bullish_cross": False,
            "ema_bearish_cross": False,
            "macd_line": 0.1,
            "macd_signal": 0.05,
            "macd_histogram": 0.05,
            "macd_bullish": True,
            "macd_bearish": False,
            "atr": 2.0,
            "atr_pct": 0.02,
            "volatility_20": 0.02,
        }
        indicator_df = pd.DataFrame([row], index=[ts])
        signal = TradingSignal(timestamp=ts, signal_type=SignalType.BUY, price=100.0, confidence=0.0, reasoning="test")
        confirmed, reason, count, _ = detector._check_indicator_confirmation(signal, indicator_df)
        assert confirmed is False
        assert "Volatility too high" in reason
        assert count == 0

    def test_volatility_filter_allows_low_vol(self):
        """When volatility_20 <= volatility_max, confirmation proceeds (RSI can confirm)."""
        config = SignalConfig(
            use_elliott_wave=False,
            use_rsi=True,
            use_ema=False,
            use_macd=False,
            use_volatility_filter=True,
            volatility_max=0.03,
        )
        detector = SignalDetector(config)
        ts = pd.Timestamp("2020-06-15")
        row = {
            "price": 100.0,
            "rsi": 30.0,
            "rsi_oversold": True,
            "rsi_overbought": False,
            "ema_short": 99.0,
            "ema_long": 98.0,
            "price_above_ema_short": True,
            "price_above_ema_long": True,
            "ema_bullish_cross": False,
            "ema_bearish_cross": False,
            "macd_line": 0.1,
            "macd_signal": 0.05,
            "macd_histogram": 0.05,
            "macd_bullish": True,
            "macd_bearish": False,
            "atr": 2.0,
            "atr_pct": 0.02,
            "volatility_20": 0.01,
        }
        indicator_df = pd.DataFrame([row], index=[ts])
        signal = TradingSignal(timestamp=ts, signal_type=SignalType.BUY, price=100.0, confidence=0.0, reasoning="test")
        confirmed, reason, count, _ = detector._check_indicator_confirmation(signal, indicator_df)
        assert confirmed is True
        assert count >= 1


class TestFilterSignalsByQuality:
    """Min confirmations and min certainty filter in detector."""

    def _make_signal(self, confirmations: int, confirmation_score: Optional[float] = None, ts=None):
        if ts is None:
            ts = pd.Timestamp("2020-06-01")
        sig = TradingSignal(
            signal_type=SignalType.BUY,
            timestamp=ts,
            price=100.0,
            confidence=0.5,
            reasoning="test",
            indicator_confirmations=confirmations,
            confirmation_score=confirmation_score,
        )
        return sig

    def test_min_confirmations_filters(self):
        """With min_confirmations=2 only signals with >=2 confirmations pass."""
        config = _technical_config(use_rsi=True, use_ema=True, use_macd=True)
        config.min_confirmations = 2
        config.min_certainty = None
        detector = SignalDetector(config)
        signals = [
            self._make_signal(0),
            self._make_signal(1),
            self._make_signal(2),
            self._make_signal(3),
        ]
        result = detector._filter_signals_by_quality(signals)
        assert len(result) == 2
        assert all(getattr(s, 'indicator_confirmations', 0) >= 2 for s in result)

    def test_min_certainty_filters(self):
        """With min_certainty=0.66 only signals with effective certainty >= 0.66 pass."""
        config = _technical_config(use_rsi=True, use_ema=True, use_macd=True)
        config.min_confirmations = None
        config.min_certainty = 0.66
        detector = SignalDetector(config)
        # Use confirmation_score when set, else count/3 -> 0.5 and 0.7
        low = self._make_signal(1, confirmation_score=0.5)
        high = self._make_signal(2, confirmation_score=0.7)
        signals = [low, high]
        result = detector._filter_signals_by_quality(signals)
        assert len(result) == 1
        assert result[0].confirmation_score == 0.7

    def test_min_certainty_uses_count_when_no_score(self):
        """When confirmation_score is None, effective certainty is indicator_confirmations/3."""
        config = _technical_config(use_rsi=True, use_ema=True, use_macd=True)
        config.min_confirmations = None
        config.min_certainty = 0.66  # 2/3 = 0.666... passes, 1/3 does not
        detector = SignalDetector(config)
        one_conf = self._make_signal(1, confirmation_score=None)   # 1/3 < 0.66
        two_conf = self._make_signal(2, confirmation_score=None)  # 2/3 >= 0.66
        result = detector._filter_signals_by_quality([one_conf, two_conf])
        assert len(result) == 1
        assert result[0].indicator_confirmations == 2

    def test_no_filter_when_both_none(self):
        """When min_confirmations and min_certainty are None, no signals dropped."""
        config = _technical_config(use_rsi=True, use_ema=True, use_macd=True)
        config.min_confirmations = None
        config.min_certainty = None
        detector = SignalDetector(config)
        signals = [self._make_signal(0), self._make_signal(3)]
        result = detector._filter_signals_by_quality(signals)
        assert len(result) == 2
