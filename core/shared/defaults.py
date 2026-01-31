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

# Inverted Elliott Wave defaults (for sell signal generation via price inversion)
ELLIOTT_INVERTED_MIN_CONFIDENCE = 0.65  # Default, can be optimized separately
ELLIOTT_INVERTED_MIN_WAVE_SIZE = 0.02  # Default, can be optimized separately

# Trade management defaults
RISK_REWARD_RATIO = 2.0
POSITION_SIZE_PCT = 0.2  # Max fraction of portfolio per trade (0-1); actual size = this × quality_factor
MAX_POSITIONS = None  # Maximum concurrent positions (None = unlimited when omitted in config)

# Confidence-based position sizing: quality factor from confirmation_score or confirmations/3 (0-1)
USE_CONFIDENCE_SIZING = True

# Confirmation-based position sizing: quality factor from confirmation_size_factors (normalized to 0-1)
USE_CONFIRMATION_MODULATION = False
CONFIRMATION_SIZE_FACTORS = {
    0: 0.0,   # Skip trades with no confirmations
    1: 0.5,
    2: 2.0,
    3: 2.0,
}

# Flexible position sizing (confidence/risk-reward based)
USE_FLEXIBLE_SIZING = False  # Enable flexible sizing based on signal quality
FLEXIBLE_SIZING_METHOD = "confidence"  # "confidence", "risk_reward", or "combined"
FLEXIBLE_SIZING_TARGET_RR = 2.5  # Target risk/reward ratio for risk_reward method

# Per-instrument position limits
MAX_POSITIONS_PER_INSTRUMENT = None  # None = no limit, otherwise max positions per instrument

# Minimal position size (absolute value, same units as capital; None = no minimum)
MIN_POSITION_SIZE = None  # Skip opening a position if position capital would be below this

# Trend filtering (only trade in direction of EMA trend)
USE_TREND_FILTER = False

# Walk-forward evaluation defaults
STEP_DAYS = 1  # Days between evaluation points (daily evaluation for maximum accuracy)
LOOKBACK_DAYS = 365  # Days of history for signal generation
INITIAL_CAPITAL = 10000.0  # Starting portfolio capital for backtest (same units as prices/fees)

# Signal detection defaults
INDICATOR_WARMUP_PERIOD = 50  # Skip first N data points where indicators aren't fully calculated
ADX_REGIME_THRESHOLD = 30  # ADX > threshold indicates strong trend (bull/bear detection)

# Regime detection defaults
MA_SLOPE_PERIOD = 50  # Period for moving average slope calculation in regime detection

# Volatility indicator (ATR) defaults
ATR_PERIOD = 14  # ATR period for risk/sizing (same as typical ADX)
VOLATILITY_WINDOW = 20  # Rolling window for return-volatility (e.g. filter/confirmation)

# Trading cost defaults (None = no fee; applied per side on entry and exit)
TRADE_FEE_PCT = None  # e.g. 0.001 for 0.1% of trade value per side
TRADE_FEE_ABSOLUTE = None  # e.g. 1.0 per trade per side
TRADE_FEE_MIN = None  # Minimum fee per side (absolute); fee is clamped to >= this when set
TRADE_FEE_MAX = None  # Maximum fee per side (absolute); fee is clamped to <= this when set
