"""
Strategy configuration for trading signals.

Contains strategy configurations, presets, and grid search generation.
All indicators (RSI, EMA, MACD, Elliott Wave) are treated equally.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import sys
from pathlib import Path

# Add paths for imports
core_dir = Path(__file__).parent.parent.parent
project_root = core_dir.parent
sys.path.insert(0, str(project_root))

# Import centralized defaults (single source of truth)
from core.shared.defaults import (
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    EMA_SHORT_PERIOD, EMA_LONG_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ELLIOTT_MIN_CONFIDENCE, ELLIOTT_MIN_WAVE_SIZE,
    ELLIOTT_INVERTED_MIN_CONFIDENCE, ELLIOTT_INVERTED_MIN_WAVE_SIZE,
    RISK_REWARD_RATIO, POSITION_SIZE_PCT, MAX_POSITIONS,
    USE_CONFIDENCE_SIZING, CONFIDENCE_SIZE_MULTIPLIER,
    USE_CONFIRMATION_MODULATION, CONFIRMATION_SIZE_FACTORS,
    USE_FLEXIBLE_SIZING, FLEXIBLE_SIZING_METHOD, FLEXIBLE_SIZING_TARGET_RR,
    MAX_POSITIONS_PER_INSTRUMENT,
    USE_TREND_FILTER,
    STEP_DAYS, LOOKBACK_DAYS,
)


@dataclass
class SignalConfig:
    """Configuration for signal generation (simplified version of StrategyConfig)."""
    # Indicator enable/disable
    use_elliott_wave: bool = False
    use_elliott_wave_inverted: bool = False
    use_elliott_wave_inverted_exit: bool = False  # Inverted EW SELLs close longs (sell-to-close)
    use_rsi: bool = False
    use_ema: bool = False
    use_macd: bool = False
    
    # Elliott Wave parameters
    min_confidence: float = ELLIOTT_MIN_CONFIDENCE
    min_wave_size: float = ELLIOTT_MIN_WAVE_SIZE
    
    # Inverted Elliott Wave parameters (for sell signal generation)
    min_confidence_inverted: float = ELLIOTT_INVERTED_MIN_CONFIDENCE
    min_wave_size_inverted: float = ELLIOTT_INVERTED_MIN_WAVE_SIZE
    
    # Technical indicator parameters
    rsi_period: int = RSI_PERIOD
    rsi_oversold: int = RSI_OVERSOLD
    rsi_overbought: int = RSI_OVERBOUGHT
    ema_short_period: int = EMA_SHORT_PERIOD
    ema_long_period: int = EMA_LONG_PERIOD
    macd_fast: int = MACD_FAST
    macd_slow: int = MACD_SLOW
    macd_signal: int = MACD_SIGNAL
    
    # Signal filtering
    signal_types: str = "all"  # "buy", "sell", or "all"
    require_all_indicators: bool = False
    use_trend_filter: bool = USE_TREND_FILTER  # Only trade in direction of EMA trend


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    
    name: str
    description: str = ""
    
    # Elliott Wave detection parameters (from shared.defaults)
    use_elliott_wave: bool = False  # Elliott Wave is now just another indicator
    min_confidence: float = ELLIOTT_MIN_CONFIDENCE
    min_wave_size: float = ELLIOTT_MIN_WAVE_SIZE
    
    # Inverted Elliott Wave detection parameters (for sell signal generation via price inversion)
    use_elliott_wave_inverted: bool = False
    use_elliott_wave_inverted_exit: bool = False  # Inverted EW SELLs close longs (sell-to-close)
    min_confidence_inverted: float = ELLIOTT_INVERTED_MIN_CONFIDENCE
    min_wave_size_inverted: float = ELLIOTT_INVERTED_MIN_WAVE_SIZE

    # Indicator on/off settings (all indicators work the same way)
    use_rsi: bool = False
    use_ema: bool = False
    use_macd: bool = False
    require_all_indicators: bool = False  # All enabled indicators must confirm
    
    # RSI parameters (from shared.defaults)
    rsi_period: int = RSI_PERIOD
    rsi_oversold: int = RSI_OVERSOLD
    rsi_overbought: int = RSI_OVERBOUGHT
    
    # EMA parameters (from shared.defaults)
    ema_short_period: int = EMA_SHORT_PERIOD
    ema_long_period: int = EMA_LONG_PERIOD
    
    # MACD parameters (from shared.defaults)
    macd_fast: int = MACD_FAST
    macd_slow: int = MACD_SLOW
    macd_signal: int = MACD_SIGNAL
    
    # Signal detection parameters
    signal_types: str = "all"  # "buy", "sell", or "all"
    
    # Target/stop-loss parameters (from shared.defaults)
    risk_reward: float = RISK_REWARD_RATIO
    use_atr_stops: bool = False  # Use ATR-based stops instead of fixed percentage
    atr_stop_multiplier: float = 2.0  # Stop loss = entry ± (multiplier × ATR)
    atr_period: int = 14  # Period for ATR calculation
    
    # Trade evaluation parameters
    max_days: Optional[int] = None
    require_both_targets: bool = False
    hold_through_stop_loss: bool = False
    
    # Position sizing parameters (from shared.defaults)
    position_size_pct: float = POSITION_SIZE_PCT  # Base position size (% of capital)
    max_positions: int = MAX_POSITIONS
    
    # Volatility-adjusted position sizing
    use_volatility_sizing: bool = False  # Adjust position size based on volatility (ATR)
    volatility_threshold: float = 0.03  # Reduce size when ATR/price > this threshold (3%)
    volatility_size_reduction: float = 0.5  # Multiply position size by this when volatile (50%)
    
    # Confidence-based position sizing (from shared.defaults)
    use_confidence_sizing: bool = USE_CONFIDENCE_SIZING  # Scale position size with indicator confirmations
    confidence_size_multiplier: float = CONFIDENCE_SIZE_MULTIPLIER  # Additional % per confirmation
    
    # Confirmation-based position sizing modulation (from shared.defaults)
    use_confirmation_modulation: bool = USE_CONFIRMATION_MODULATION  # Multiplicative sizing based on confirmations
    confirmation_size_factors: Dict[int, float] = field(default_factory=lambda: CONFIRMATION_SIZE_FACTORS.copy())
    
    # Flexible position sizing (from shared.defaults)
    use_flexible_sizing: bool = USE_FLEXIBLE_SIZING  # Enable flexible sizing based on signal quality
    flexible_sizing_method: str = FLEXIBLE_SIZING_METHOD  # "confidence", "risk_reward", or "combined"
    flexible_sizing_target_rr: float = FLEXIBLE_SIZING_TARGET_RR  # Target risk/reward ratio for risk_reward method
    
    # Per-instrument position limits
    max_positions_per_instrument: Optional[int] = MAX_POSITIONS_PER_INSTRUMENT  # None = no limit, otherwise max positions per instrument
    
    # Trend filtering (from shared.defaults)
    use_trend_filter: bool = USE_TREND_FILTER  # Only trade in direction of EMA trend
    
    # Market regime detection and adaptive signals
    use_regime_detection: bool = False  # Enable market regime detection (ADX + MA slope)
    invert_signals_in_bull: bool = True  # Invert EW signals in bull markets (counter-trend trading)
    adx_threshold: float = 30.0  # ADX threshold for regime detection (default: 30)
    
    # Walk-forward parameters (from shared.defaults)
    step_days: int = STEP_DAYS
    lookback_days: int = LOOKBACK_DAYS
    
    # Data parameters
    column: str = "Close"
    granularity: str = "daily"
    
    # Instruments and date ranges (for config-based execution)
    instruments: List[str] = field(default_factory=lambda: ["djia"])  # Default to DJIA for backward compat
    start_date: Optional[str] = None  # Start date (YYYY-MM-DD), None = use all available data
    end_date: Optional[str] = None    # End date (YYYY-MM-DD), None = use all available data


# Current best baseline configuration
# Hypothesis test results (2026-01-24): ew_all_indicators +45.75% Alpha across all 10 periods
# Elliott Wave + RSI + EMA + MACD combination is the clear winner
# The "dilution hypothesis" is WRONG - combining indicators dramatically improves performance
BASELINE_CONFIG = StrategyConfig(
    name="baseline",
    description="Elliott Wave + RSI + EMA + MACD - Best performing strategy (+45.75% alpha)",
    use_elliott_wave=True,   # Core indicator - required for best performance
    use_rsi=True,            # Combined with EW: +28.71% alpha (standalone RSI fails)
    use_ema=True,            # Combined with EW: improves signal quality
    use_macd=True,           # Combined with EW: +25.71% alpha (best win rate: 49.1%)
    require_all_indicators=False,
    signal_types="all",      # Allow both buy and sell signals
    risk_reward=RISK_REWARD_RATIO,  # 2.0 - validated optimal
    max_days=None,
    require_both_targets=False,
    hold_through_stop_loss=False,
    position_size_pct=POSITION_SIZE_PCT,        # 0.2 (20% per trade)
    max_positions=MAX_POSITIONS,                # 5 concurrent positions
    use_confidence_sizing=USE_CONFIDENCE_SIZING,  # True (scale with confidence)
    confidence_size_multiplier=CONFIDENCE_SIZE_MULTIPLIER,  # 0.1 (+10% per confirmation)
    step_days=STEP_DAYS,          # 1 (daily evaluation)
    lookback_days=LOOKBACK_DAYS,  # 365 days
    column="Close",
    granularity="daily",
)


# Alternative configurations for comparison
PRESET_CONFIGS = {
    "baseline": BASELINE_CONFIG,
    
    # Elliott Wave only
    "elliott_only": StrategyConfig(
        name="elliott_only",
        description="Elliott Wave only",
        use_elliott_wave=True,
        use_rsi=False,
        use_ema=False,
        use_macd=False,
        signal_types="buy",
    ),
    
    # Indicator combinations
    "rsi_only": StrategyConfig(
        name="rsi_only",
        description="RSI signals only",
        use_elliott_wave=False,
        use_rsi=True,
        use_ema=False,
        use_macd=False,
        signal_types="all",  # Allow both buy and sell signals
    ),
    
    "ema_only": StrategyConfig(
        name="ema_only",
        description="EMA crossover signals only",
        use_elliott_wave=False,
        use_rsi=False,
        use_ema=True,
        use_macd=False,
        signal_types="buy",
    ),
    
    "macd_only": StrategyConfig(
        name="macd_only",
        description="MACD signals only",
        use_elliott_wave=False,
        use_rsi=False,
        use_ema=False,
        use_macd=True,
        signal_types="buy",
    ),
    
    "indicators_combined": StrategyConfig(
        name="indicators_comb",
        description="All indicators combined (RSI + EMA + MACD)",
        use_elliott_wave=False,
        use_rsi=True,
        use_ema=True,
        use_macd=True,
        signal_types="buy",
    ),
}


def generate_grid_configs(
    name_prefix: str = "grid",
    use_elliott_wave_values: list = None,
    use_rsi_values: list = None,
    use_ema_values: list = None,
    use_macd_values: list = None,
    signal_types_values: list = None,
    include_parameter_variations: bool = True,
) -> list:
    """
    Generate a grid of configurations testing indicator combinations and parameters.
    
    Args:
        name_prefix: Prefix for generated config names
        use_elliott_wave_values: List of [True, False] to test with/without Elliott
        use_rsi_values: List of [True, False] to test with/without RSI
        use_ema_values: List of [True, False] to test with/without EMA
        use_macd_values: List of [True, False] to test with/without MACD
        signal_types_values: List of signal types ("buy", "all")
        include_parameter_variations: If True, test different parameter values
        
    Returns:
        List of StrategyConfig objects for all valid combinations
    """
    # Default values - test all combinations
    if use_elliott_wave_values is None:
        use_elliott_wave_values = [True, False]
    if use_rsi_values is None:
        use_rsi_values = [True, False]
    if use_ema_values is None:
        use_ema_values = [True, False]
    if use_macd_values is None:
        use_macd_values = [True, False]
    if signal_types_values is None:
        signal_types_values = ["buy"]  # Focus on buy-only for now (clearer results)
    
    # Parameter variations to test (when include_parameter_variations=True)
    # NOTE: Indicator parameters are already optimized and set as defaults.
    # We focus on:
    # 1. Indicator combinations (on/off) - to find best indicator mix
    # 2. Risk management parameters - to optimize position sizing and risk/reward
    if include_parameter_variations:
        # Indicator parameters: Use defaults only (already optimized)
        elliott_params = []  # Use defaults only
        rsi_params = []  # Use defaults only
        ema_params = []  # Use defaults only
        macd_params = []  # Use defaults only
        
        # Risk management parameters (test alternatives - default 2.0/0.2/5/0.1 is baseline)
        risk_reward_params = [1.5, 2.5, 3.0]  # Default 2.0 is baseline
        position_size_params = [0.1, 0.3, 0.4]  # Default 0.2 (20%) is baseline
        max_positions_params = [3, 7, 10]  # Default 5 is baseline
        confidence_multiplier_params = [0.05, 0.15, 0.2]  # Default 0.1 is baseline
    else:
        # Single default value for each (from shared.defaults - single source of truth)
        elliott_params = [{"min_confidence": ELLIOTT_MIN_CONFIDENCE, "min_wave_size": ELLIOTT_MIN_WAVE_SIZE}]
        rsi_params = [{"rsi_period": RSI_PERIOD, "rsi_oversold": RSI_OVERSOLD, "rsi_overbought": RSI_OVERBOUGHT}]
        ema_params = [{"ema_short_period": EMA_SHORT_PERIOD, "ema_long_period": EMA_LONG_PERIOD}]
        macd_params = [{"macd_fast": MACD_FAST, "macd_slow": MACD_SLOW, "macd_signal": MACD_SIGNAL}]
        risk_reward_params = [RISK_REWARD_RATIO]
        position_size_params = [POSITION_SIZE_PCT]
        max_positions_params = [MAX_POSITIONS]
        confidence_multiplier_params = [CONFIDENCE_SIZE_MULTIPLIER]
    
    configs = []
    
    for use_elliott in use_elliott_wave_values:
        for use_rsi in use_rsi_values:
            for use_ema in use_ema_values:
                for use_macd in use_macd_values:
                    for signal_type in signal_types_values:
                        # Skip invalid combinations
                        has_indicators = use_rsi or use_ema or use_macd
                        
                        # Must have at least one signal source
                        if not use_elliott and not has_indicators:
                            continue
                        
                        # Default parameter sets (from shared.defaults - single source of truth)
                        ew_default = {"min_confidence": ELLIOTT_MIN_CONFIDENCE, "min_wave_size": ELLIOTT_MIN_WAVE_SIZE}
                        rsi_default = {"rsi_period": RSI_PERIOD, "rsi_oversold": RSI_OVERSOLD, "rsi_overbought": RSI_OVERBOUGHT}
                        ema_default = {"ema_short_period": EMA_SHORT_PERIOD, "ema_long_period": EMA_LONG_PERIOD}
                        macd_default = {"macd_fast": MACD_FAST, "macd_slow": MACD_SLOW, "macd_signal": MACD_SIGNAL}
                        
                        # Build list of parameter combinations to test
                        param_combos = []
                        
                        # Always include the default combination
                        param_combos.append((
                            ew_default if use_elliott else {},
                            rsi_default if use_rsi else {},
                            ema_default if use_ema else {},
                            macd_default if use_macd else {},
                        ))
                        
                        if include_parameter_variations:
                            # Test Elliott Wave variations (keep others at default)
                            if use_elliott:
                                for ew_p in elliott_params:
                                    if (ew_p.get('min_confidence') != ew_default.get('min_confidence') or
                                        ew_p.get('min_wave_size') != ew_default.get('min_wave_size')):
                                        param_combos.append((
                                            ew_p,
                                            rsi_default if use_rsi else {},
                                            ema_default if use_ema else {},
                                            macd_default if use_macd else {},
                                        ))
                            
                            # Test RSI variations (keep others at default)
                            if use_rsi:
                                for rsi_p in rsi_params:
                                    if (rsi_p.get('rsi_period') != rsi_default.get('rsi_period') or
                                        rsi_p.get('rsi_oversold') != rsi_default.get('rsi_oversold') or
                                        rsi_p.get('rsi_overbought') != rsi_default.get('rsi_overbought')):
                                        param_combos.append((
                                            ew_default if use_elliott else {},
                                            rsi_p,
                                            ema_default if use_ema else {},
                                            macd_default if use_macd else {},
                                        ))
                            
                            # Test EMA variations (keep others at default)
                            if use_ema:
                                for ema_p in ema_params:
                                    if (ema_p.get('ema_short_period') != ema_default.get('ema_short_period') or
                                        ema_p.get('ema_long_period') != ema_default.get('ema_long_period')):
                                        param_combos.append((
                                            ew_default if use_elliott else {},
                                            rsi_default if use_rsi else {},
                                            ema_p,
                                            macd_default if use_macd else {},
                                        ))
                            
                            # Test MACD variations (keep others at default)
                            if use_macd:
                                for macd_p in macd_params:
                                    if (macd_p.get('macd_fast') != macd_default.get('macd_fast') or
                                        macd_p.get('macd_slow') != macd_default.get('macd_slow') or
                                        macd_p.get('macd_signal') != macd_default.get('macd_signal')):
                                        param_combos.append((
                                            ew_default if use_elliott else {},
                                            rsi_default if use_rsi else {},
                                            ema_default if use_ema else {},
                                            macd_p,
                                        ))
                        
                        # Generate configs for each parameter combination
                        # Test risk management parameters one at a time
                        # BUT: Only test risk management params on baseline indicator config (all defaults)
                        for ew_p, rsi_p, ema_p, macd_p in param_combos:
                            # Check if this is the baseline indicator configuration
                            # Baseline = ALL indicators enabled (EW + RSI + EMA + MACD) with default parameters
                            is_baseline = (
                                use_elliott and use_rsi and use_ema and use_macd and
                                (ew_p == ew_default) and
                                (rsi_p == rsi_default) and
                                (ema_p == ema_default) and
                                (macd_p == macd_default)
                            )
                            
                            # Test risk management parameters
                            risk_params_list = [
                                {
                                    'risk_reward': RISK_REWARD_RATIO,
                                    'position_size_pct': POSITION_SIZE_PCT,
                                    'max_positions': MAX_POSITIONS,
                                    'confidence_size_multiplier': CONFIDENCE_SIZE_MULTIPLIER,
                                }
                            ]
                            
                            # Add risk management parameter variations (one at a time)
                            # Only test on baseline indicator config to keep search space manageable
                            if include_parameter_variations and is_baseline:
                                for rr in risk_reward_params:
                                    if rr != RISK_REWARD_RATIO:
                                        risk_params_list.append({
                                            'risk_reward': rr,
                                            'position_size_pct': POSITION_SIZE_PCT,
                                            'max_positions': MAX_POSITIONS,
                                            'confidence_size_multiplier': CONFIDENCE_SIZE_MULTIPLIER,
                                        })
                                
                                for ps in position_size_params:
                                    if ps != POSITION_SIZE_PCT:
                                        risk_params_list.append({
                                            'risk_reward': RISK_REWARD_RATIO,
                                            'position_size_pct': ps,
                                            'max_positions': MAX_POSITIONS,
                                            'confidence_size_multiplier': CONFIDENCE_SIZE_MULTIPLIER,
                                        })
                                
                                for mp in max_positions_params:
                                    if mp != MAX_POSITIONS:
                                        risk_params_list.append({
                                            'risk_reward': RISK_REWARD_RATIO,
                                            'position_size_pct': POSITION_SIZE_PCT,
                                            'max_positions': mp,
                                            'confidence_size_multiplier': CONFIDENCE_SIZE_MULTIPLIER,
                                        })
                                
                                for cm in confidence_multiplier_params:
                                    if cm != CONFIDENCE_SIZE_MULTIPLIER:
                                        risk_params_list.append({
                                            'risk_reward': RISK_REWARD_RATIO,
                                            'position_size_pct': POSITION_SIZE_PCT,
                                            'max_positions': MAX_POSITIONS,
                                            'confidence_size_multiplier': cm,
                                        })
                            
                            # Generate config for each risk management parameter set
                            for risk_params in risk_params_list:
                                # Build name
                                parts = []
                                if use_elliott:
                                    parts.append("EW")
                                if use_rsi:
                                    parts.append("RSI")
                                if use_ema:
                                    parts.append("EMA")
                                if use_macd:
                                    parts.append("MACD")
                                
                                base_name = f"{name_prefix}_" + "_".join(parts)
                                
                                # Add parameter info to name for uniqueness
                                param_parts = []
                                if use_elliott and ew_p:
                                    param_parts.append(f"c{ew_p.get('min_confidence', 0.65):.2f}")
                                    param_parts.append(f"w{ew_p.get('min_wave_size', 0.03):.2f}")
                                if use_rsi and rsi_p:
                                    param_parts.append(f"rp{rsi_p.get('rsi_period', 14)}")
                                if use_ema and ema_p:
                                    param_parts.append(f"es{ema_p.get('ema_short_period', 20)}")
                                    param_parts.append(f"el{ema_p.get('ema_long_period', 50)}")
                                if use_macd and macd_p:
                                    param_parts.append(f"mf{macd_p.get('macd_fast', 12)}")
                                    param_parts.append(f"ms{macd_p.get('macd_slow', 26)}")
                                
                                # Add risk management params to name if different from defaults
                                if risk_params['risk_reward'] != RISK_REWARD_RATIO:
                                    param_parts.append(f"rr{risk_params['risk_reward']:.1f}")
                                if risk_params['position_size_pct'] != POSITION_SIZE_PCT:
                                    param_parts.append(f"ps{int(risk_params['position_size_pct']*100)}")
                                if risk_params['max_positions'] != MAX_POSITIONS:
                                    param_parts.append(f"mp{risk_params['max_positions']}")
                                if risk_params['confidence_size_multiplier'] != CONFIDENCE_SIZE_MULTIPLIER:
                                    param_parts.append(f"cm{risk_params['confidence_size_multiplier']:.2f}")
                                
                                if param_parts:
                                    name = f"{base_name}_{'_'.join(param_parts)}"
                                else:
                                    name = base_name
                                
                                if signal_type != "buy":
                                    name += f"_{signal_type}"
                                
                                # Build description
                                desc_parts = []
                                if use_elliott:
                                    desc_parts.append("Elliott Wave")
                                if use_rsi:
                                    desc_parts.append("RSI")
                                if use_ema:
                                    desc_parts.append("EMA")
                                if use_macd:
                                    desc_parts.append("MACD")
                                description = " + ".join(desc_parts)
                                
                                # Merge all parameters
                                config = StrategyConfig(
                                    name=name,
                                    description=description,
                                    use_elliott_wave=use_elliott,
                                    min_confidence=ew_p.get('min_confidence', ELLIOTT_MIN_CONFIDENCE) if ew_p else ELLIOTT_MIN_CONFIDENCE,
                                    min_wave_size=ew_p.get('min_wave_size', ELLIOTT_MIN_WAVE_SIZE) if ew_p else ELLIOTT_MIN_WAVE_SIZE,
                                    use_rsi=use_rsi,
                                    use_ema=use_ema,
                                    use_macd=use_macd,
                                    require_all_indicators=False,
                                    rsi_period=rsi_p.get('rsi_period', RSI_PERIOD) if rsi_p else RSI_PERIOD,
                                    rsi_oversold=rsi_p.get('rsi_oversold', RSI_OVERSOLD) if rsi_p else RSI_OVERSOLD,
                                    rsi_overbought=rsi_p.get('rsi_overbought', RSI_OVERBOUGHT) if rsi_p else RSI_OVERBOUGHT,
                                    ema_short_period=ema_p.get('ema_short_period', EMA_SHORT_PERIOD) if ema_p else EMA_SHORT_PERIOD,
                                    ema_long_period=ema_p.get('ema_long_period', EMA_LONG_PERIOD) if ema_p else EMA_LONG_PERIOD,
                                    macd_fast=macd_p.get('macd_fast', MACD_FAST) if macd_p else MACD_FAST,
                                    macd_slow=macd_p.get('macd_slow', MACD_SLOW) if macd_p else MACD_SLOW,
                                    macd_signal=macd_p.get('macd_signal', MACD_SIGNAL) if macd_p else MACD_SIGNAL,
                                    signal_types=signal_type,
                                    risk_reward=risk_params['risk_reward'],
                                    max_days=None,
                                    require_both_targets=False,
                                    hold_through_stop_loss=False,
                                    position_size_pct=risk_params['position_size_pct'],
                                    max_positions=risk_params['max_positions'],
                                    use_confidence_sizing=USE_CONFIDENCE_SIZING,
                                    confidence_size_multiplier=risk_params['confidence_size_multiplier'],
                                    step_days=STEP_DAYS,
                                    lookback_days=LOOKBACK_DAYS,
                                    column="Close",
                                    granularity="daily",
                                )
                                configs.append(config)
    
    return configs


class BaselineConfig:
    """Helper class to access baseline and preset configurations."""
    
    @staticmethod
    def get_baseline() -> StrategyConfig:
        """Get the current baseline configuration."""
        return BASELINE_CONFIG
    
    @staticmethod
    def get_preset(name: str) -> StrategyConfig:
        """Get a preset configuration by name."""
        if name not in PRESET_CONFIGS:
            available = ", ".join(PRESET_CONFIGS.keys())
            raise ValueError(f"Unknown preset: {name}. Available: {available}")
        return PRESET_CONFIGS[name]
    
    @staticmethod
    def list_presets() -> list:
        """List all available preset names."""
        return list(PRESET_CONFIGS.keys())
    
    @staticmethod
    def create_custom(
        name: str,
        description: str = "",
        **kwargs
    ) -> StrategyConfig:
        """Create a custom configuration with optional overrides from baseline."""
        # Start with baseline values
        config_dict = {
            'name': name,
            'description': description,
            'min_confidence': BASELINE_CONFIG.min_confidence,
            'min_wave_size': BASELINE_CONFIG.min_wave_size,
            'signal_types': BASELINE_CONFIG.signal_types,
            'risk_reward': BASELINE_CONFIG.risk_reward,
            'max_days': BASELINE_CONFIG.max_days,
            'require_both_targets': BASELINE_CONFIG.require_both_targets,
            'hold_through_stop_loss': BASELINE_CONFIG.hold_through_stop_loss,
            'column': BASELINE_CONFIG.column,
            'granularity': BASELINE_CONFIG.granularity,
        }
        # Override with provided values
        config_dict.update(kwargs)
        return StrategyConfig(**config_dict)
