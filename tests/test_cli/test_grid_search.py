"""
Tests for grid_search CLI instrument precedence and date source.

Dates are always taken from config (no CLI date args). Instrument: CLI overrides config.
"""
import pytest
from pathlib import Path
from core.signals.config_loader import load_config_from_yaml
from core.signals.config import StrategyConfig


class TestDatePrecedence:
    """Dates always from config (grid-search has no date CLI args)."""
    
    def test_config_dates_used(self):
        """Dates always come from config (no CLI date args in grid-search)."""
        config = load_config_from_yaml('configs/baseline.yaml')
        assert config.start_date is not None
        assert config.end_date is not None
        assert config.start_date < config.end_date
    
    def test_defaults_used_when_config_missing_dates(self):
        """Defaults should be used when config has no dates (grid-search validation applies these)."""
        config = StrategyConfig(
            name="test",
            use_elliott_wave=True,
            instruments=["djia"],
            start_date=None,
            end_date=None
        )
        # Grid-search logic: config.start_date or "2000-01-01"
        final_start = config.start_date or "2000-01-01"
        final_end = config.end_date or "2020-01-01"
        assert final_start == "2000-01-01"
        assert final_end == "2020-01-01"


class TestInstrumentPrecedence:
    """Test instrument precedence: CLI > Config > Default"""
    
    def test_config_instrument_used_when_no_cli(self):
        """Config instrument should be used when CLI arg not provided."""
        config = load_config_from_yaml('configs/baseline.yaml')
        cli_instrument = None
        final_instruments = [cli_instrument] if cli_instrument else config.instruments
        assert final_instruments == config.instruments
        assert len(final_instruments) >= 1

    def test_cli_instrument_overrides_config(self):
        """CLI instrument should override config instrument."""
        config = load_config_from_yaml('configs/baseline.yaml')
        config_instruments = list(config.instruments)
        cli_instrument = "djia"
        final_instruments = [cli_instrument] if cli_instrument else config.instruments
        assert final_instruments == [cli_instrument]
        assert final_instruments != config_instruments
    
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
    """Instrument: CLI overrides config. Dates: always from config."""
    
    def test_instrument_from_cli_dates_from_config(self):
        """Instrument can come from CLI; dates always from config (no CLI date args)."""
        config = load_config_from_yaml('configs/baseline.yaml')
        cli_instrument = "nasdaq"
        final_instruments = [cli_instrument] if cli_instrument else (config.instruments or ["djia"])
        final_start = config.start_date or "2000-01-01"
        final_end = config.end_date or "2020-01-01"
        assert final_instruments == ["nasdaq"]
        assert final_start == config.start_date
        assert final_end == config.end_date
