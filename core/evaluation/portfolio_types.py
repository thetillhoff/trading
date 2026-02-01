"""
Portfolio and simulation types: positions, wallet state, simulation result.

Extracted for reuse and to keep portfolio.py focused on simulation logic.
New strategies or evaluators can import these types without pulling in PortfolioSimulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd


class PositionStatus(Enum):
    """Status of a trading position."""
    OPEN = "open"
    CLOSED_TARGET = "closed_target"
    CLOSED_STOP = "closed_stop"
    CLOSED_TIMEOUT = "closed_timeout"
    CLOSED_SIGNAL = "closed_signal"  # Closed by sell-to-close signal (no short opened)
    CLOSED_END = "closed_end"  # Still open at end of simulation


@dataclass
class Position:
    """A single trading position."""
    entry_timestamp: pd.Timestamp
    entry_price: float
    shares: float  # Number of shares/units bought
    cost_basis: float  # Total cost to enter position
    target_price: Optional[float]
    stop_loss: Optional[float]
    signal_type: str  # "buy" or "sell"

    # Indicator data that triggered the trade
    rsi_value: Optional[float] = None  # RSI value at signal time
    ema_short: Optional[float] = None  # Short EMA value
    ema_long: Optional[float] = None   # Long EMA value
    macd_value: Optional[float] = None  # MACD value
    macd_signal: Optional[float] = None  # MACD signal value
    macd_histogram: Optional[float] = None  # MACD histogram
    indicator_confirmations: int = 0  # Number of indicators that confirmed
    original_signal_type: str = ""  # Track if this was originally a sell signal

    # Additional trade metadata
    certainty: float = 0.0  # Confidence score (0-1) based on indicator confirmations
    risk_amount: float = 0.0  # Risk amount (entry - stop_loss for buys, stop_loss - entry for sells)
    risk_reward_ratio: float = 0.0  # Target distance / risk amount
    projection_price: Optional[float] = None  # Expected price projection (could be target or calculated)
    position_size_method: str = "base"  # Which sizing method was used: "base", "confidence_sizing", "confirmation_modulation"
    quality_factor: float = 0.0  # Sizing factor in [0, 1]; actual_size_pct = position_size_pct * quality_factor
    trend_filter_active: bool = False  # Whether trend filter was used for this trade
    trend_direction: str = ""  # "bullish" or "bearish" at entry

    # Filled when position closes
    exit_timestamp: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    pnl: float = 0.0  # Profit/loss in currency units

    # Instrument identifier (for multi-instrument trading)
    instrument: Optional[str] = None  # Instrument symbol (e.g., "djia", "sp500")


@dataclass
class WalletState:
    """Snapshot of wallet at a point in time."""
    timestamp: pd.Timestamp
    cash: float
    invested_value: float  # Current value of open positions
    total_value: float  # cash + invested_value
    return_pct: float  # Return percentage from initial capital


@dataclass
class SimulationResult:
    """Results from a portfolio simulation."""
    initial_capital: float
    final_capital: float
    total_return_pct: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Time series of wallet value
    wallet_history: List[WalletState]

    # All positions (open and closed)
    positions: List[Position]

    # Metrics
    max_drawdown_pct: float
    avg_position_size: float
    avg_days_held: float

    # Exposure metrics
    avg_exposure_pct: float = 100.0  # Average % of capital invested over time

    # Risk/reward metrics
    avg_win_pct: float = 0.0  # Average % gain per winning trade
    avg_loss_pct: float = 0.0  # Average % loss per losing trade (negative number)
    profit_factor: float = 0.0  # Total gains / Total losses (>1 is good)
    expectancy_pct: float = 0.0  # Expected % return per trade

    # Trading costs (sum of all entry + exit fees paid)
    total_trading_costs: float = 0.0
