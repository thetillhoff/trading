#!/usr/bin/env python3
"""
Parameter reference CLI.

Shows all configurable parameters, their valid ranges, and defaults.
"""
import sys
from pathlib import Path

# Add core to path
core_dir = Path(__file__).parent.parent / "core"
sys.path.insert(0, str(core_dir.parent))

from core.shared.defaults import (
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    EMA_SHORT_PERIOD, EMA_LONG_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ELLIOTT_MIN_CONFIDENCE, ELLIOTT_MIN_WAVE_SIZE,
    RISK_REWARD_RATIO, POSITION_SIZE_PCT, MAX_POSITIONS,
    USE_CONFIDENCE_SIZING, CONFIDENCE_SIZE_MULTIPLIER,
    STEP_DAYS, LOOKBACK_DAYS,
)


def main():
    """Print all configurable parameters with their ranges and defaults."""
    
    print("=" * 80)
    print("TRADING SYSTEM PARAMETER REFERENCE")
    print("=" * 80)
    print()
    
    # Technical Indicators
    print("TECHNICAL INDICATORS")
    print("-" * 80)
    print()
    
    print("RSI (Relative Strength Index):")
    print(f"  --rsi-period        Period: {RSI_PERIOD} (default)")
    print(f"                      Range: 5-30, recommended: 7-21")
    print(f"  --rsi-oversold      Oversold threshold: {RSI_OVERSOLD} (default)")
    print(f"                      Range: 10-40, recommended: 20-30")
    print(f"  --rsi-overbought    Overbought threshold: {RSI_OVERBOUGHT} (default)")
    print(f"                      Range: 60-90, recommended: 70-80")
    print()
    
    print("EMA (Exponential Moving Average):")
    print(f"  --ema-short-period  Short period: {EMA_SHORT_PERIOD} (default)")
    print(f"                      Range: 5-50, recommended: 9-21")
    print(f"  --ema-long-period   Long period: {EMA_LONG_PERIOD} (default)")
    print(f"                      Range: 20-200, recommended: 50-100")
    print("                      Note: Long period should be > Short period")
    print()
    
    print("MACD (Moving Average Convergence Divergence):")
    print(f"  --macd-fast         Fast period: {MACD_FAST} (default)")
    print(f"                      Range: 8-15, recommended: 12")
    print(f"  --macd-slow         Slow period: {MACD_SLOW} (default)")
    print(f"                      Range: 20-30, recommended: 26")
    print(f"  --macd-signal       Signal period: {MACD_SIGNAL} (default)")
    print(f"                      Range: 6-15, recommended: 9-12")
    print()
    
    # Elliott Wave
    print("ELLIOTT WAVE DETECTION:")
    print("-" * 80)
    print()
    print(f"  --min-confidence    Minimum confidence: {ELLIOTT_MIN_CONFIDENCE} (default)")
    print(f"                      Range: 0.0-1.0, recommended: 0.5-0.8")
    print(f"                      Lower = more patterns detected (more false positives)")
    print(f"                      Higher = fewer patterns (more reliable but may miss opportunities)")
    print()
    print(f"  --min-wave-size     Minimum wave size: {ELLIOTT_MIN_WAVE_SIZE} (default)")
    print(f"                      Range: 0.01-0.10, recommended: 0.02-0.05")
    print(f"                      Ratio of price range (e.g., 0.03 = 3% of price range)")
    print(f"                      Lower = smaller waves detected")
    print()
    
    # Indicator Enable/Disable
    print("INDICATOR SELECTION:")
    print("-" * 80)
    print()
    print("  --use-elliott-wave   Enable Elliott Wave indicator (default: False)")
    print("  --use-rsi            Enable RSI indicator (default: False)")
    print("  --use-ema            Enable EMA indicator (default: True in baseline)")
    print("  --use-macd           Enable MACD indicator (default: True in baseline)")
    print()
    print("  Note: Baseline configuration uses EMA + MACD (best performing combination)")
    print()
    
    # Risk Management
    print("RISK MANAGEMENT:")
    print("-" * 80)
    print()
    print(f"  Risk/Reward Ratio: {RISK_REWARD_RATIO} (default)")
    print(f"                     Range: 1.0-5.0, recommended: 1.5-3.0")
    print(f"                     How much profit target vs stop loss")
    print(f"                     Example: 2.0 = target 2x the stop loss distance")
    print()
    print(f"  Position Size: {POSITION_SIZE_PCT*100:.0f}% (default)")
    print(f"                  Range: 5-50%, recommended: 10-30%")
    print(f"                  Percentage of capital per trade")
    print()
    print(f"  Max Positions: {MAX_POSITIONS} (default)")
    print(f"                  Range: 1-10, recommended: 3-7")
    print(f"                  Maximum concurrent open positions")
    print()
    
    # Confidence-based Sizing
    print("CONFIDENCE-BASED POSITION SIZING:")
    print("-" * 80)
    print()
    print(f"  Enabled: {USE_CONFIDENCE_SIZING} (default)")
    print(f"  Multiplier: {CONFIDENCE_SIZE_MULTIPLIER*100:.0f}% (default)")
    print(f"              Range: 0.05-0.20, recommended: 0.10")
    print(f"              Additional position size per indicator confirmation")
    print()
    print(f"  Example: Base 20%, multiplier 10%, 2 confirmations = 20% + (10% * 2) = 40%")
    print()
    
    # Walk-forward Evaluation
    print("WALK-FORWARD EVALUATION:")
    print("-" * 80)
    print()
    print(f"  Step Days: {STEP_DAYS} (default)")
    print(f"             Range: 1-30, recommended: 1 (daily) or 7-14")
    print(f"             Days between evaluation points")
    print(f"             Lower = more accurate but slower")
    print()
    print(f"  Lookback Days: {LOOKBACK_DAYS} (default)")
    print(f"                 Range: 90-730, recommended: 180-365")
    print(f"                 Days of historical data for signal generation")
    print()
    
    # Signal Types
    print("SIGNAL TYPES:")
    print("-" * 80)
    print()
    print("  --signal-types      Options: 'buy', 'sell', 'all'")
    print("                      Default: 'buy' (buy-only performs better)")
    print()
    
    # Trade Management
    print("TRADE MANAGEMENT:")
    print("-" * 80)
    print()
    print("  --max-days          Maximum days to hold a trade")
    print("                      Range: None (no limit) or 30-365")
    print("                      Default: None (hold until target/stop)")
    print()
    print("  --hold-through-stop-loss")
    print("                      Hold through stop-loss until recovery")
    print("                      Default: False")
    print()
    
    # Data Parameters
    print("DATA PARAMETERS:")
    print("-" * 80)
    print()
    print("  --instrument        Instrument to evaluate")
    print("                      Options: djia, sp500, dax, gold, eurusd, msci_world")
    print("                      Default: djia")
    print()
    print("  --column            Price column to use")
    print("                      Options: Close, High, Low, Open")
    print("                      Default: Close")
    print()
    print("  --start-date        Start date for evaluation (YYYY-MM-DD)")
    print("  --end-date          End date for evaluation (YYYY-MM-DD)")
    print()
    
    # Presets
    print("PRESET CONFIGURATIONS:")
    print("-" * 80)
    print()
    print("  --preset            Use a preset configuration")
    print("                      Options: baseline, ema_only, macd_only, rsi_only,")
    print("                              all_indicators, conservative, aggressive")
    print()
    print("  Baseline (default): EMA + MACD - optimized best performer")
    print()
    
    print("=" * 80)
    print()
    print("USAGE EXAMPLES:")
    print("-" * 80)
    print()
    print("  # Use defaults (baseline: EMA + MACD)")
    print("  make evaluate ARGS='--instrument djia'")
    print()
    print("  # Custom RSI period")
    print("  make evaluate ARGS='--instrument djia --use-rsi --rsi-period 14'")
    print()
    print("  # All indicators with custom Elliott Wave")
    print("  make evaluate ARGS='--instrument sp500 --use-elliott-wave --use-rsi")
    print("                     --use-ema --use-macd --min-confidence 0.7'")
    print()
    print("  # Custom risk management")
    print("  make evaluate ARGS='--instrument djia --preset baseline'")
    print("                     # (risk management params are in config, not CLI yet)")
    print()
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
