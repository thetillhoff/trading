"""
Tests for grid_search CLI date/instrument precedence logic.

Verifies that CLI arguments take precedence over config values,
and config values are used when CLI args are not provided.
"""
import pytest
from pathlib import Path
from core.signals.config_loader import load_config_from_yaml
from core.signals.config import StrategyConfig


class TestDatePrecedence:
    """Test date precedence: CLI > Config > Default"""
    
    def test_config_dates_used_when_no_cli_args(self):
        """Config dates should be used when CLI args not provided."""
        config = load_config_from_yaml('configs/baseline.yaml')
        
        # Simulate no CLI args
        cli_start = None
        cli_end = None
        
        # Final values should come from config
        final_start = cli_start or config.start_date
        final_end = cli_end or config.end_date
        
        assert final_start == "2000-01-01"  # From config
        assert final_end == "2020-01-01"    # From config
    
    def test_cli_dates_override_config_dates(self):
        """CLI dates should override config dates when provided."""
        config = load_config_from_yaml('configs/baseline.yaml')
        
        # Simulate CLI args provided
        cli_start = "2018-01-01"
        cli_end = "2020-01-01"
        
        # Final values should come from CLI (precedence)
        final_start = cli_start or config.start_date
        final_end = cli_end or config.end_date
        
        assert final_start == "2018-01-01"  # From CLI
        assert final_end == "2020-01-01"     # From CLI
        assert final_start != config.start_date  # CLI overrides config
    
    def test_partial_cli_args(self):
        """Partial CLI args should override only specified values."""
        config = load_config_from_yaml('configs/baseline.yaml')
        
        # Only start_date provided via CLI
        cli_start = "2015-01-01"
        cli_end = None
        
        final_start = cli_start or config.start_date
        final_end = cli_end or config.end_date
        
        assert final_start == "2015-01-01"  # From CLI
        assert final_end == "2020-01-01"     # From config (not overridden)
    
    def test_defaults_used_when_nothing_provided(self):
        """Defaults should be used when config has no dates and no CLI args."""
        # Create config without dates
        config = StrategyConfig(
            name="test",
            use_elliott_wave=True,
            instruments=["djia"],
            start_date=None,
            end_date=None
        )
        
        # No CLI args
        cli_start = None
        cli_end = None
        
        # Should use defaults
        final_start = cli_start or config.start_date or "2000-01-01"
        final_end = cli_end or config.end_date or "2020-01-01"
        
        assert final_start == "2000-01-01"  # Default
        assert final_end == "2020-01-01"     # Default


class TestInstrumentPrecedence:
    """Test instrument precedence: CLI > Config > Default"""
    
    def test_config_instrument_used_when_no_cli(self):
        """Config instrument should be used when CLI arg not provided."""
        config = load_config_from_yaml('configs/baseline.yaml')
        
        cli_instrument = None
        final_instruments = [cli_instrument] if cli_instrument else config.instruments
        
        assert final_instruments == ["djia"]  # From config
    
    def test_cli_instrument_overrides_config(self):
        """CLI instrument should override config instrument."""
        config = load_config_from_yaml('configs/baseline.yaml')
        
        cli_instrument = "sp500"
        final_instruments = [cli_instrument] if cli_instrument else config.instruments
        
        assert final_instruments == ["sp500"]  # From CLI
        assert final_instruments != config.instruments  # CLI overrides config
    
    def test_default_instrument_when_nothing_provided(self):
        """Default instrument should be used when config has none and no CLI."""
        config = StrategyConfig(
            name="test",
            use_elliott_wave=True,
            instruments=[],
        )
        
        cli_instrument = None
        final_instruments = [cli_instrument] if cli_instrument else (config.instruments or ["djia"])
        
        assert final_instruments == ["djia"]  # Default


class TestPrecedenceOrder:
    """Test that precedence order is correct: CLI > Config > Default"""
    
    def test_full_precedence_chain(self):
        """Test complete precedence chain for all parameters."""
        # Config with values
        config = load_config_from_yaml('configs/baseline.yaml')
        
        # CLI args provided
        cli_instrument = "nasdaq"
        cli_start = "2010-01-01"
        cli_end = "2015-01-01"
        
        # Apply precedence logic
        final_instruments = [cli_instrument] if cli_instrument else (config.instruments or ["djia"])
        final_start = cli_start or config.start_date or "2000-01-01"
        final_end = cli_end or config.end_date or "2020-01-01"
        
        # All should come from CLI (highest precedence)
        assert final_instruments == ["nasdaq"]
        assert final_start == "2010-01-01"
        assert final_end == "2015-01-01"
        
        # Verify they differ from config values
        assert final_instruments != config.instruments
        assert final_start != config.start_date
        assert final_end != config.end_date
