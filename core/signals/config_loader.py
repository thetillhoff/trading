"""
YAML configuration loader for trading strategies.

Loads strategy configurations from YAML files, allowing easy sharing
and modification of strategies without code changes.
"""
import yaml
from pathlib import Path
from typing import Union, Optional
from dataclasses import asdict

from .config import StrategyConfig
from ..shared.defaults import *


def load_config_from_yaml(yaml_path: Union[str, Path]) -> StrategyConfig:
    """
    Load strategy configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        StrategyConfig object
        
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML is invalid or missing required fields
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if not config_dict:
        raise ValueError(f"Empty config file: {yaml_path}")
    
    # Extract values from nested structure
    name = config_dict.get('name', yaml_path.stem)
    description = config_dict.get('description', '')
    
    # Indicators
    indicators = config_dict.get('indicators', {})
    elliott = indicators.get('elliott_wave', {})
    elliott_inverted = indicators.get('elliott_wave_inverted', {})
    elliott_inverted_exit = indicators.get('elliott_wave_inverted_exit', {})
    rsi = indicators.get('rsi', {})
    ema = indicators.get('ema', {})
    macd = indicators.get('macd', {})
    
    # Risk management
    risk = config_dict.get('risk', {})
    
    # Signals
    signals = config_dict.get('signals', {})
    
    # Regime detection
    regime = config_dict.get('regime', {})
    
    # Evaluation
    evaluation = config_dict.get('evaluation', {})
    
    # Data & execution parameters
    data_params = config_dict.get('data', {})
    
    # Build StrategyConfig
    return StrategyConfig(
        name=name,
        description=description,
        
        # Elliott Wave
        use_elliott_wave=elliott.get('enabled', False),
        min_confidence=elliott.get('min_confidence', ELLIOTT_MIN_CONFIDENCE),
        min_wave_size=elliott.get('min_wave_size', ELLIOTT_MIN_WAVE_SIZE),
        
        # Inverted Elliott Wave
        use_elliott_wave_inverted=elliott_inverted.get('enabled', False),
        use_elliott_wave_inverted_exit=elliott_inverted_exit.get('enabled', False),
        min_confidence_inverted=elliott_inverted.get('min_confidence', ELLIOTT_INVERTED_MIN_CONFIDENCE),
        min_wave_size_inverted=elliott_inverted.get('min_wave_size', ELLIOTT_INVERTED_MIN_WAVE_SIZE),
        
        # RSI
        use_rsi=rsi.get('enabled', False),
        rsi_period=rsi.get('period', RSI_PERIOD),
        rsi_oversold=rsi.get('oversold', RSI_OVERSOLD),
        rsi_overbought=rsi.get('overbought', RSI_OVERBOUGHT),
        
        # EMA
        use_ema=ema.get('enabled', False),
        ema_short_period=ema.get('short_period', EMA_SHORT_PERIOD),
        ema_long_period=ema.get('long_period', EMA_LONG_PERIOD),
        
        # MACD
        use_macd=macd.get('enabled', False),
        macd_fast=macd.get('fast', MACD_FAST),
        macd_slow=macd.get('slow', MACD_SLOW),
        macd_signal=macd.get('signal', MACD_SIGNAL),
        
        # Signals
        signal_types=signals.get('signal_types', 'all'),
        require_all_indicators=signals.get('require_all_indicators', False),
        use_trend_filter=signals.get('use_trend_filter', USE_TREND_FILTER),
        
        # Risk management
        risk_reward=risk.get('risk_reward', RISK_REWARD_RATIO),
        position_size_pct=risk.get('position_size_pct', POSITION_SIZE_PCT),
        max_positions=risk.get('max_positions', MAX_POSITIONS),
        max_positions_per_instrument=risk.get('max_positions_per_instrument', MAX_POSITIONS_PER_INSTRUMENT),
        use_confidence_sizing=risk.get('use_confidence_sizing', USE_CONFIDENCE_SIZING),
        confidence_size_multiplier=risk.get('confidence_size_multiplier', CONFIDENCE_SIZE_MULTIPLIER),
        use_confirmation_modulation=risk.get('use_confirmation_modulation', USE_CONFIRMATION_MODULATION),
        use_flexible_sizing=risk.get('use_flexible_sizing', USE_FLEXIBLE_SIZING),
        flexible_sizing_method=risk.get('flexible_sizing_method', FLEXIBLE_SIZING_METHOD),
        flexible_sizing_target_rr=risk.get('flexible_sizing_target_rr', FLEXIBLE_SIZING_TARGET_RR),
        
        # Regime detection
        use_regime_detection=regime.get('use_regime_detection', False),
        invert_signals_in_bull=regime.get('invert_signals_in_bull', True),
        adx_threshold=regime.get('adx_threshold', 30.0),
        
        # Evaluation
        step_days=evaluation.get('step_days', STEP_DAYS),
        lookback_days=evaluation.get('lookback_days', LOOKBACK_DAYS),
        
        # Data
        column='Close',
        granularity='daily',
        instruments=data_params.get('instruments', ['djia']),
        start_date=data_params.get('start_date'),
        end_date=data_params.get('end_date'),
    )


def save_config_to_yaml(config: StrategyConfig, yaml_path: Union[str, Path]):
    """
    Save strategy configuration to YAML file.
    
    Args:
        config: StrategyConfig object to save
        yaml_path: Path where to save YAML file
    """
    yaml_path = Path(yaml_path)
    
    # Build nested structure
    config_dict = {
        'name': config.name,
        'description': config.description,
        
        'indicators': {
            'elliott_wave': {
                'enabled': config.use_elliott_wave,
                'min_confidence': config.min_confidence,
                'min_wave_size': config.min_wave_size,
            },
            'elliott_wave_inverted': {
                'enabled': config.use_elliott_wave_inverted,
                'min_confidence': config.min_confidence_inverted,
                'min_wave_size': config.min_wave_size_inverted,
            },
            'elliott_wave_inverted_exit': {
                'enabled': config.use_elliott_wave_inverted_exit,
            },
            'rsi': {
                'enabled': config.use_rsi,
                'period': config.rsi_period,
                'oversold': config.rsi_oversold,
                'overbought': config.rsi_overbought,
            },
            'ema': {
                'enabled': config.use_ema,
                'short_period': config.ema_short_period,
                'long_period': config.ema_long_period,
            },
            'macd': {
                'enabled': config.use_macd,
                'fast': config.macd_fast,
                'slow': config.macd_slow,
                'signal': config.macd_signal,
            },
        },
        
        'risk': {
            'risk_reward': config.risk_reward,
            'position_size_pct': config.position_size_pct,
            'max_positions': config.max_positions,
            'max_positions_per_instrument': config.max_positions_per_instrument,
            'use_confidence_sizing': config.use_confidence_sizing,
            'confidence_size_multiplier': config.confidence_size_multiplier,
            'use_flexible_sizing': config.use_flexible_sizing,
            'flexible_sizing_method': config.flexible_sizing_method,
            'flexible_sizing_target_rr': config.flexible_sizing_target_rr,
        },
        
        'signals': {
            'signal_types': config.signal_types,
            'require_all_indicators': config.require_all_indicators,
            'use_trend_filter': config.use_trend_filter,
        },
        
        'regime': {
            'use_regime_detection': config.use_regime_detection,
            'invert_signals_in_bull': config.invert_signals_in_bull,
            'adx_threshold': config.adx_threshold,
        },
        
        'evaluation': {
            'step_days': config.step_days,
            'lookback_days': config.lookback_days,
        },
        
        'data': {
            'instruments': config.instruments,
            'start_date': config.start_date,
            'end_date': config.end_date,
        },
    }
    
    # Ensure parent directory exists
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write YAML
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
