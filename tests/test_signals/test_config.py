"""
Tests for strategy configuration.
"""
import pytest
from core.signals.config import StrategyConfig, BASELINE_CONFIG, generate_grid_configs
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
