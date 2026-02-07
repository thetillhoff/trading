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
from ..data.loader import list_available_tickers


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
    raw_instruments = data_params.get('instruments')
    if raw_instruments is None or (isinstance(raw_instruments, list) and len(raw_instruments) == 0):
        instruments = list_available_tickers()
    else:
        instruments = raw_instruments if isinstance(raw_instruments, list) else [raw_instruments]

    # Trading costs
    costs = config_dict.get('costs', {})
    
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
        
        # Derive indicator usage from weights (with fallback to enabled flag for backward compatibility)
        indicator_weights=signals.get('indicator_weights'),
        
        # RSI - use weight if present, otherwise fall back to enabled flag
        use_rsi=(
            ('rsi' in signals.get('indicator_weights', {}) and signals.get('indicator_weights', {})['rsi'] > 0)
            if signals.get('indicator_weights') is not None
            else rsi.get('enabled', False)
        ),
        rsi_period=rsi.get('period', RSI_PERIOD),
        rsi_oversold=rsi.get('oversold', RSI_OVERSOLD),
        rsi_overbought=rsi.get('overbought', RSI_OVERBOUGHT),
        
        # EMA - use weight if present, otherwise fall back to enabled flag
        use_ema=(
            ('ema' in signals.get('indicator_weights', {}) and signals.get('indicator_weights', {})['ema'] > 0)
            if signals.get('indicator_weights') is not None
            else ema.get('enabled', False)
        ),
        ema_short_period=ema.get('short_period', EMA_SHORT_PERIOD),
        ema_long_period=ema.get('long_period', EMA_LONG_PERIOD),
        
        # MACD - use weight if present, otherwise fall back to enabled flag
        use_macd=(
            ('macd' in signals.get('indicator_weights', {}) and signals.get('indicator_weights', {})['macd'] > 0)
            if signals.get('indicator_weights') is not None
            else macd.get('enabled', False)
        ),
        macd_fast=macd.get('fast', MACD_FAST),
        macd_slow=macd.get('slow', MACD_SLOW),
        macd_signal=macd.get('signal', MACD_SIGNAL),
        
        # Signals
        signal_types=signals.get('signal_types', 'all'),
        min_confirmations=signals.get('min_confirmations'),
        min_certainty=signals.get('min_certainty'),
        use_trend_filter=signals.get('use_trend_filter', USE_TREND_FILTER),
        use_multi_timeframe=signals.get('use_multi_timeframe', False),
        multi_timeframe_weekly_ema_period=signals.get('multi_timeframe_weekly_ema_period', 8),
        use_multi_timeframe_filter=signals.get('use_multi_timeframe_filter', True),
        
        # Risk management
        risk_reward=risk.get('risk_reward', RISK_REWARD_RATIO),
        position_size_pct=risk.get('position_size_pct', POSITION_SIZE_PCT),
        max_positions=risk.get('max_positions', MAX_POSITIONS),
        max_positions_per_instrument=risk.get('max_positions_per_instrument', MAX_POSITIONS_PER_INSTRUMENT),
        min_position_size=risk.get('min_position_size', MIN_POSITION_SIZE),
        use_confidence_sizing=risk.get('use_confidence_sizing', USE_CONFIDENCE_SIZING),
        use_confirmation_modulation=risk.get('use_confirmation_modulation', USE_CONFIRMATION_MODULATION),
        use_flexible_sizing=risk.get('use_flexible_sizing', USE_FLEXIBLE_SIZING),
        flexible_sizing_method=risk.get('flexible_sizing_method', FLEXIBLE_SIZING_METHOD),
        flexible_sizing_target_rr=risk.get('flexible_sizing_target_rr', FLEXIBLE_SIZING_TARGET_RR),
        use_wave_relationship_targets=risk.get('use_wave_relationship_targets', True),
        use_volatility_sizing=risk.get('use_volatility_sizing', False),
        volatility_threshold=risk.get('volatility_threshold', 0.03),
        volatility_size_reduction=risk.get('volatility_size_reduction', 0.5),
        use_volatility_filter=signals.get('use_volatility_filter', False),
        volatility_max=signals.get('volatility_max', 0.02),
        trade_fee_pct=costs.get('trade_fee_pct'),
        trade_fee_absolute=costs.get('trade_fee_absolute'),
        trade_fee_min=costs.get('trade_fee_min', TRADE_FEE_MIN),
        trade_fee_max=costs.get('trade_fee_max', TRADE_FEE_MAX),
        interest_rate_pa=costs.get('interest_rate_pa', 0.02),
        
        # Regime detection
        use_regime_detection=regime.get('use_regime_detection', False),
        invert_signals_in_bull=regime.get('invert_signals_in_bull', True),
        adx_threshold=regime.get('adx_threshold', 30.0),
        regime_mode=regime.get('regime_mode', 'adx_ma'),
        regime_vol_window=regime.get('vol_window', 20),
        regime_vol_threshold=regime.get('vol_threshold', 0.015),
        regime_slope_window=regime.get('slope_window', 5),
        regime_slope_threshold=regime.get('slope_threshold', 0.0005),
        
        # Evaluation
        step_days=evaluation.get('step_days', STEP_DAYS),
        lookback_days=evaluation.get('lookback_days', LOOKBACK_DAYS),
        initial_capital=float(evaluation.get('initial_capital', INITIAL_CAPITAL)),
        
        # Data
        column='Close',
        granularity='daily',
        instruments=instruments,
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
                'period': config.rsi_period,
                'oversold': config.rsi_oversold,
                'overbought': config.rsi_overbought,
            },
            'ema': {
                'short_period': config.ema_short_period,
                'long_period': config.ema_long_period,
            },
            'macd': {
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
            'use_flexible_sizing': config.use_flexible_sizing,
            'flexible_sizing_method': config.flexible_sizing_method,
            'flexible_sizing_target_rr': config.flexible_sizing_target_rr,
            'use_wave_relationship_targets': config.use_wave_relationship_targets,
            'use_volatility_sizing': config.use_volatility_sizing,
            'volatility_threshold': config.volatility_threshold,
            'volatility_size_reduction': config.volatility_size_reduction,
        },
        
        'costs': {
            'trade_fee_pct': config.trade_fee_pct,
            'trade_fee_absolute': config.trade_fee_absolute,
            'interest_rate_pa': config.interest_rate_pa,
        },
        
        'signals': {
            'signal_types': config.signal_types,
            **({'min_confirmations': config.min_confirmations} if config.min_confirmations is not None else {}),
            **({'min_certainty': config.min_certainty} if config.min_certainty is not None else {}),
            'use_trend_filter': config.use_trend_filter,
            'indicator_weights': config.indicator_weights,
            'use_volatility_filter': config.use_volatility_filter,
            'volatility_max': config.volatility_max,
            'use_multi_timeframe': config.use_multi_timeframe,
            'multi_timeframe_weekly_ema_period': config.multi_timeframe_weekly_ema_period,
            'use_multi_timeframe_filter': config.use_multi_timeframe_filter,
        },
        
        'regime': {
            'use_regime_detection': config.use_regime_detection,
            'invert_signals_in_bull': config.invert_signals_in_bull,
            'adx_threshold': config.adx_threshold,
            'regime_mode': config.regime_mode,
            'vol_window': config.regime_vol_window,
            'vol_threshold': config.regime_vol_threshold,
            'slope_window': config.regime_slope_window,
            'slope_threshold': config.regime_slope_threshold,
        },
        
        'evaluation': {
            'step_days': config.step_days,
            'lookback_days': config.lookback_days,
            'initial_capital': config.initial_capital,
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
