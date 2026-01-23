"""
Centralized default values for indicator parameters.

This is the SINGLE SOURCE OF TRUTH for all indicator parameter defaults.
All modules should import from here to ensure consistency.

Optimized values from comprehensive testing (2000-2010 DJIA, Jan 2026):
- Elliott Wave 0.65/0.02: BEST PERFORMER (+13.05% alpha, 48.2% win rate)
- MACD standalone: Strong performance (+9.39% alpha)
- RSI standalone: AVOID (failed all configurations, -12% to +2% alpha)
- EMA standalone: AVOID (wrong signals in bull markets, -8.26% alpha)
- Combinations: Single strong indicator > weak combinations
"""

# RSI (Relative Strength Index) defaults
RSI_PERIOD = 7  # Optimized from grid search (vs default 14)
RSI_OVERSOLD = 25  # Optimized from grid search (vs default 30)
RSI_OVERBOUGHT = 75  # Optimized from grid search (vs default 70)

# EMA (Exponential Moving Average) defaults
EMA_SHORT_PERIOD = 20  # Optimized from grid search
EMA_LONG_PERIOD = 50  # Optimized from grid search

# MACD (Moving Average Convergence Divergence) defaults
MACD_FAST = 12  # Standard default
MACD_SLOW = 26  # Standard default
MACD_SIGNAL = 12  # Optimized from grid search (vs default 9, shows +0.44% Alpha)

# Elliott Wave defaults (BEST STRATEGY: +13.05% alpha, 48.2% win rate)
ELLIOTT_MIN_CONFIDENCE = 0.65  # Optimal confidence threshold
ELLIOTT_MIN_WAVE_SIZE = 0.02  # Optimized: 0.02 outperforms 0.03 (+13.05% vs +12.27% alpha)

# Trade management defaults
RISK_REWARD_RATIO = 2.0
POSITION_SIZE_PCT = 0.2  # 20% of capital per trade (base size)
MAX_POSITIONS = 5  # Maximum concurrent positions

# Confidence-based position sizing
# If enabled, position size scales with number of indicator confirmations
# Example: base_size=0.2, multiplier=0.1 means:
#   0 confirmations: 0.2 (20%)
#   1 confirmation:  0.3 (30%)
#   2 confirmations: 0.4 (40%)
#   3 confirmations: 0.5 (50%)
USE_CONFIDENCE_SIZING = True  # Enable confidence-based position sizing
CONFIDENCE_SIZE_MULTIPLIER = 0.1  # Additional % per indicator confirmation (0.1 = +10% per confirmation)

# Confirmation-based position sizing modulation (multiplicative instead of additive)
# If enabled, position size is multiplied by factors based on confirmation count
# More aggressive than confidence sizing - can skip low-quality signals entirely
USE_CONFIRMATION_MODULATION = False  # Enable confirmation-based position size modulation
CONFIRMATION_SIZE_FACTORS = {
    0: 0.0,   # Skip trades with 0 confirmations (no indicators agree)
    1: 0.5,   # Half size for 1 confirmation (only one indicator)
    2: 2.0,   # Double size for 2 confirmations (two indicators agree)
    3: 2.0,   # Double size for 3 confirmations (all three indicators agree)
}

# Trend filtering
# Only trade in direction of EMA trend (requires EMA to be enabled)
# BUY signals only when EMA_short > EMA_long (bullish trend)
# SELL signals only when EMA_short < EMA_long (bearish trend)
USE_TREND_FILTER = False  # Enable trend-based signal filtering

# Walk-forward evaluation defaults
STEP_DAYS = 1  # Days between evaluation points (daily evaluation for maximum accuracy)
LOOKBACK_DAYS = 365  # Days of history for signal generation
