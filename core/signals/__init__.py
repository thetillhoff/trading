"""
Signal generation module.

Unified signal detector that uses all indicators (RSI, EMA, MACD, Elliott Wave)
to generate trading signals. All indicators are treated equally - they calculate
values, and signal generation interprets those values.
"""
from .detector import SignalDetector
from .config import StrategyConfig, SignalConfig, BASELINE_CONFIG, PRESET_CONFIGS, generate_grid_configs
from .target_calculator import TargetCalculator

__all__ = [
    'SignalDetector',
    'StrategyConfig',
    'SignalConfig',
    'BASELINE_CONFIG',
    'PRESET_CONFIGS',
    'generate_grid_configs',
    'TargetCalculator',
]
