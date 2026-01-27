"""
Tests for Elliott Wave indicator (ElliottWaveDetector, Wave, WaveType, WaveLabel).
"""
import pytest
import pandas as pd
import numpy as np
from core.indicators.elliott_wave import (
    ElliottWaveDetector,
    Wave,
    WaveType,
    WaveLabel,
)


@pytest.fixture
def short_prices():
    """Too short for wave detection (need 5+ extrema)."""
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    return pd.Series(100 + np.arange(20) * 0.5, index=dates)


@pytest.fixture
def sine_like_prices():
    """Oscillating prices that can produce extrema."""
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    t = np.arange(200)
    values = 100 + 10 * np.sin(t * 0.1) + 0.1 * t
    return pd.Series(values, index=dates)


class TestWaveType:
    """Test WaveType enum."""

    def test_impulse_exists(self):
        assert WaveType.IMPULSE.value == "impulse"

    def test_corrective_exists(self):
        assert WaveType.CORRECTIVE.value == "corrective"


class TestWaveLabel:
    """Test WaveLabel enum."""

    def test_wave_labels_exist(self):
        assert WaveLabel.WAVE_1.value == "1"
        assert WaveLabel.WAVE_5.value == "5"
        assert WaveLabel.WAVE_A.value == "a"
        assert WaveLabel.WAVE_C.value == "c"
        assert WaveLabel.UNKNOWN.value == "?"


class TestWave:
    """Test Wave dataclass."""

    def test_wave_has_required_fields(self):
        w = Wave(
            start_idx=0,
            end_idx=10,
            start_price=100.0,
            end_price=105.0,
            wave_type=WaveType.IMPULSE,
            label=WaveLabel.WAVE_1,
            direction="up",
            confidence=0.8,
        )
        assert w.start_idx == 0
        assert w.end_idx == 10
        assert w.start_price == 100.0
        assert w.end_price == 105.0
        assert w.wave_type == WaveType.IMPULSE
        assert w.label == WaveLabel.WAVE_1
        assert w.direction == "up"
        assert w.confidence == 0.8


class TestElliottWaveDetector:
    """Test ElliottWaveDetector."""

    def test_init_default(self):
        det = ElliottWaveDetector()
        assert det.min_wave_length is None
        assert det.max_wave_length is None
        assert det.retracement_threshold == 0.236

    def test_init_with_params(self):
        det = ElliottWaveDetector(
            min_wave_length=5,
            max_wave_length=100,
            retracement_threshold=0.382,
        )
        assert det.min_wave_length == 5
        assert det.max_wave_length == 100
        assert det.retracement_threshold == 0.382

    def test_detect_waves_short_data_returns_empty(self, short_prices):
        """With too little data, detect_waves returns []."""
        det = ElliottWaveDetector()
        waves = det.detect_waves(short_prices)
        assert waves == []

    def test_detect_waves_min_length_returns_empty(self, short_prices):
        """When min_wave_length * 2 > len(data), returns []."""
        det = ElliottWaveDetector(min_wave_length=20)
        waves = det.detect_waves(short_prices)
        assert waves == []

    def test_detect_waves_returns_list(self, sine_like_prices):
        """detect_waves returns a list (possibly empty)."""
        det = ElliottWaveDetector()
        waves = det.detect_waves(sine_like_prices)
        assert isinstance(waves, list)
        for w in waves:
            assert isinstance(w, Wave)

    def test_detect_waves_min_confidence_filters(self, sine_like_prices):
        """Higher min_confidence can reduce number of waves."""
        det = ElliottWaveDetector()
        waves_0 = det.detect_waves(sine_like_prices, min_confidence=0.0)
        waves_high = det.detect_waves(sine_like_prices, min_confidence=0.99)
        assert len(waves_high) <= len(waves_0)

    def test_detect_waves_only_complete_patterns(self, sine_like_prices):
        """only_complete_patterns=True returns subset or same as False."""
        det = ElliottWaveDetector()
        waves_any = det.detect_waves(sine_like_prices, only_complete_patterns=False)
        waves_complete = det.detect_waves(sine_like_prices, only_complete_patterns=True)
        assert len(waves_complete) <= len(waves_any)
