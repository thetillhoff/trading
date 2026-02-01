"""
Tests for strategy configuration.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from core.signals.config import StrategyConfig, BASELINE_CONFIG, generate_grid_configs
from core.signals.config_loader import load_config_from_yaml
from core.shared.defaults import (
    RSI_PERIOD, EMA_SHORT_PERIOD, MACD_FAST,
    ELLIOTT_MIN_CONFIDENCE, RISK_REWARD_RATIO
)


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""
    
    def test_baseline_config_exists(self):
        """Baseline config should be defined."""
        assert BASELINE_CONFIG is not None
        assert isinstance(BASELINE_CONFIG, StrategyConfig)
    
    def test_baseline_uses_defaults(self):
        """Baseline config should use centralized defaults."""
        # These should match defaults.py
        assert BASELINE_CONFIG.risk_reward == RISK_REWARD_RATIO
    
    def test_config_creation(self):
        """Should be able to create custom config."""
        config = StrategyConfig(
            name="test_config",
            use_elliott_wave=True,
            use_rsi=False,
            use_ema=True,
            use_macd=True,
        )
        
        assert config.name == "test_config"
        assert config.use_elliott_wave is True
        assert config.use_rsi is False


class TestConfigValidation:
    """Config validation fails fast with clear errors."""

    def test_ema_short_must_be_less_than_long(self):
        with pytest.raises(ValueError, match="EMA short_period.*must be less than long_period"):
            StrategyConfig(
                name="bad",
                ema_short_period=50,
                ema_long_period=20,
            )

    def test_rsi_oversold_must_be_less_than_overbought(self):
        with pytest.raises(ValueError, match="RSI oversold.*must be less than overbought"):
            StrategyConfig(
                name="bad",
                rsi_oversold=80,
                rsi_overbought=25,
            )

    def test_risk_reward_must_be_positive(self):
        with pytest.raises(ValueError, match="risk_reward must be > 0"):
            StrategyConfig(name="bad", risk_reward=0)
        with pytest.raises(ValueError, match="risk_reward must be > 0"):
            StrategyConfig(name="bad", risk_reward=-1.0)

    def test_position_size_pct_must_be_in_zero_one(self):
        with pytest.raises(ValueError, match="position_size_pct must be in"):
            StrategyConfig(name="bad", position_size_pct=0)
        with pytest.raises(ValueError, match="position_size_pct must be in"):
            StrategyConfig(name="bad", position_size_pct=1.5)

    def test_min_certainty_must_be_in_zero_one(self):
        with pytest.raises(ValueError, match="min_certainty must be in"):
            StrategyConfig(name="bad", min_certainty=1.5)

    def test_multi_timeframe_weekly_ema_period_must_be_at_least_one(self):
        with pytest.raises(ValueError, match="multi_timeframe_weekly_ema_period must be >= 1"):
            StrategyConfig(name="bad", multi_timeframe_weekly_ema_period=0)

    def test_min_confirmations_must_be_non_negative(self):
        with pytest.raises(ValueError, match="min_confirmations must be >= 0"):
            StrategyConfig(name="bad", min_confirmations=-1)

    def test_signal_config_validates_ema_rsi(self):
        from core.signals.config import SignalConfig
        with pytest.raises(ValueError, match="EMA short_period"):
            SignalConfig(ema_short_period=50, ema_long_period=20)


class TestGridGeneration:
    """Test grid configuration generation."""
    
    def test_generate_configs(self):
        """Should generate multiple configs."""
        configs = generate_grid_configs(
            name_prefix="test",
            include_parameter_variations=False
        )
        
        assert len(configs) > 0
        assert all(isinstance(c, StrategyConfig) for c in configs)
    
    def test_unique_names(self):
        """Generated configs should have unique names."""
        configs = generate_grid_configs(
            name_prefix="test",
            include_parameter_variations=False
        )
        
        names = [c.name for c in configs]
        assert len(names) == len(set(names))
    
    def test_no_empty_configs(self):
        """All configs should have at least one indicator enabled."""
        configs = generate_grid_configs(
            name_prefix="test",
            include_parameter_variations=False
        )
        
        for config in configs:
            has_indicator = (
                config.use_elliott_wave or
                config.use_rsi or
                config.use_ema or
                config.use_macd
            )
            assert has_indicator, f"Config {config.name} has no indicators"


class TestConfigLoaderMinConfirmationsCertainty:
    """Config loader reads min_confirmations and min_certainty from signals section."""

    def test_load_min_confirmations_and_min_certainty(self):
        """YAML with min_confirmations and min_certainty sets them on StrategyConfig."""
        yaml_content = """
name: test_quality
description: test
indicators:
  elliott_wave: { enabled: false }
  rsi: { enabled: true }
  ema: { enabled: false }
  macd: { enabled: false }
risk:
  risk_reward: 2.0
  position_size_pct: 0.2
  max_positions: 5
signals:
  signal_types: all
  min_confirmations: 2
  min_certainty: 0.66
regime:
  use_regime_detection: false
costs: {}
evaluation:
  step_days: 1
  lookback_days: 365
data:
  instruments: [djia]
  start_date: '2000-01-01'
  end_date: '2020-01-01'
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            path = Path(f.name)
        try:
            config = load_config_from_yaml(path)
            assert config.min_confirmations == 2
            assert config.min_certainty == 0.66
        finally:
            path.unlink(missing_ok=True)


class TestConfigLoaderInstrumentsDefault:
    """When data.instruments is missing or empty, config uses all tickers in data/tickers/."""

    def test_empty_instruments_uses_list_available_tickers(self):
        """YAML with instruments: [] or no data.instruments → config.instruments from list_available_tickers()."""
        yaml_content = """
name: test_tickers
description: test
indicators:
  elliott_wave: { enabled: false }
  rsi: { enabled: true }
  ema: { enabled: false }
  macd: { enabled: false }
risk: {}
signals: { signal_types: all }
regime: { use_regime_detection: false }
costs: {}
evaluation: { step_days: 1, lookback_days: 365 }
data:
  instruments: []
  start_date: '2000-01-01'
  end_date: '2020-01-01'
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            path = Path(f.name)
        try:
            with patch('core.signals.config_loader.list_available_tickers', return_value=['A', 'B', 'NVDA']):
                config = load_config_from_yaml(path)
            assert config.instruments == ['A', 'B', 'NVDA']
        finally:
            path.unlink(missing_ok=True)

    def test_missing_instruments_uses_list_available_tickers(self):
        """YAML with no data.instruments key → config.instruments from list_available_tickers()."""
        yaml_content = """
name: test_tickers
description: test
indicators:
  elliott_wave: { enabled: false }
  rsi: { enabled: true }
  ema: { enabled: false }
  macd: { enabled: false }
risk: {}
signals: { signal_types: all }
regime: { use_regime_detection: false }
costs: {}
evaluation: { step_days: 1, lookback_days: 365 }
data:
  start_date: '2000-01-01'
  end_date: '2020-01-01'
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            path = Path(f.name)
        try:
            with patch('core.signals.config_loader.list_available_tickers', return_value=['X', 'Y']):
                config = load_config_from_yaml(path)
            assert config.instruments == ['X', 'Y']
        finally:
            path.unlink(missing_ok=True)

    def test_explicit_instruments_unchanged(self):
        """YAML with data.instruments: [djia] keeps [djia], does not call list_available_tickers."""
        yaml_content = """
name: test_explicit
description: test
indicators:
  elliott_wave: { enabled: false }
  rsi: { enabled: true }
  ema: { enabled: false }
  macd: { enabled: false }
risk: {}
signals: { signal_types: all }
regime: { use_regime_detection: false }
costs: {}
evaluation: { step_days: 1, lookback_days: 365 }
data:
  instruments: [djia]
  start_date: '2000-01-01'
  end_date: '2020-01-01'
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            path = Path(f.name)
        try:
            with patch('core.signals.config_loader.list_available_tickers') as m:
                config = load_config_from_yaml(path)
            m.assert_not_called()
            assert config.instruments == ['djia']
        finally:
            path.unlink(missing_ok=True)
