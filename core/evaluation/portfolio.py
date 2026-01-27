"""
Portfolio simulator with wallet-based capital management.

Simulates actual trading with a wallet that:
- Starts with initial capital (e.g., 100 units)
- Cannot be overextended (can't invest more than available)
- Tracks positions over time
- Properly compounds returns
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import sys
from pathlib import Path

# Add parent directories to path for imports
core_dir = Path(__file__).parent.parent
project_root = core_dir.parent
sys.path.insert(0, str(project_root))

from core.shared.types import SignalType


class PositionStatus(Enum):
    """Status of a trading position."""
    OPEN = "open"
    CLOSED_TARGET = "closed_target"
    CLOSED_STOP = "closed_stop"
    CLOSED_TIMEOUT = "closed_timeout"
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


class PortfolioSimulator:
    """
    Simulates trading with proper capital management.
    
    Features:
    - Tracks available cash and invested capital
    - Prevents overextension (can't invest more than available)
    - Properly compounds returns
    - Supports position sizing (fixed amount or % of portfolio)
    """
    
    def __init__(
        self,
        initial_capital: float = 100.0,
        position_size_pct: float = 1.0,  # 1.0 = 100% of available cash per trade
        max_positions: Optional[int] = 1,  # Maximum concurrent positions (None = unlimited)
        max_positions_per_instrument: Optional[int] = None,  # Maximum positions per instrument (None = no limit)
        max_days: Optional[int] = None,  # Maximum days to hold a position
        use_confidence_sizing: bool = False,  # Scale position size with indicator confirmations
        confidence_size_multiplier: float = 0.1,  # Additional % per confirmation
        use_confirmation_modulation: bool = False,  # Use multiplicative sizing based on confirmations
        confirmation_size_factors: Optional[Dict[int, float]] = None,  # Sizing factors by confirmation count
        use_volatility_sizing: bool = False,  # Adjust position size based on volatility
        volatility_threshold: float = 0.03,  # ATR/price threshold for high volatility
        volatility_size_reduction: float = 0.5,  # Multiply size by this when volatile
        use_flexible_sizing: bool = False,  # Enable flexible sizing based on signal quality
        flexible_sizing_method: str = "confidence",  # "confidence", "risk_reward", or "combined"
        flexible_sizing_target_rr: float = 2.5,  # Target risk/reward ratio for risk_reward method
    ):
        """
        Initialize the portfolio simulator.
        
        Args:
            initial_capital: Starting capital (default: 100)
            position_size_pct: Base fraction of available cash to use per trade (0.0-1.0)
            max_positions: Maximum number of concurrent positions
            max_positions_per_instrument: Maximum positions per instrument (None = no limit)
            max_days: Maximum days to hold a position (None = no limit)
            use_confidence_sizing: If True, scale position size based on indicator confirmations (additive)
            confidence_size_multiplier: Additional % per indicator confirmation (e.g., 0.1 = +10% per confirmation)
            use_confirmation_modulation: If True, multiply position size by factors based on confirmations (multiplicative)
            confirmation_size_factors: Dict mapping confirmation count to size multiplier (e.g., {0: 0.0, 1: 0.5, 2: 2.0})
            use_volatility_sizing: If True, reduce position size when volatility (ATR/price) is high
            volatility_threshold: ATR/price ratio above which to reduce position size (default: 0.03 = 3%)
            volatility_size_reduction: Multiplier for position size in high volatility (default: 0.5 = 50%)
            use_flexible_sizing: If True, scale position size based on signal confidence/risk-reward
            flexible_sizing_method: Method for flexible sizing ("confidence", "risk_reward", or "combined")
            flexible_sizing_target_rr: Target risk/reward ratio for risk_reward method
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions if max_positions is not None else 999999  # None = unlimited
        self.max_positions_per_instrument = max_positions_per_instrument
        self.max_days = max_days
        self.use_confidence_sizing = use_confidence_sizing
        self.confidence_size_multiplier = confidence_size_multiplier
        self.use_confirmation_modulation = use_confirmation_modulation
        self.confirmation_size_factors = confirmation_size_factors or {0: 0.0, 1: 0.5, 2: 2.0, 3: 2.0}
        self.use_volatility_sizing = use_volatility_sizing
        self.volatility_threshold = volatility_threshold
        self.volatility_size_reduction = volatility_size_reduction
        self.use_flexible_sizing = use_flexible_sizing
        self.flexible_sizing_method = flexible_sizing_method
        self.flexible_sizing_target_rr = flexible_sizing_target_rr
    
    def simulate_buy_and_hold(
        self,
        prices: pd.Series,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> SimulationResult:
        """
        Simulate buy-and-hold strategy.
        
        Invests all capital on day 0 and holds until the end.
        """
        if start_date is None:
            start_date = prices.index.min()
        if end_date is None:
            end_date = prices.index.max()
        
        # Filter prices to date range
        prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]
        
        if len(prices) < 2:
            return self._empty_result()
        
        entry_price = prices.iloc[0]
        shares = self.initial_capital / entry_price
        
        # Build wallet history
        wallet_history = []
        for timestamp, price in prices.items():
            current_value = shares * price
            return_pct = ((current_value - self.initial_capital) / self.initial_capital) * 100
            wallet_history.append(WalletState(
                timestamp=timestamp,
                cash=0.0,
                invested_value=current_value,
                total_value=current_value,
                return_pct=return_pct,
            ))
        
        final_value = shares * prices.iloc[-1]
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Create single position
        position = Position(
            entry_timestamp=prices.index[0],
            entry_price=entry_price,
            shares=shares,
            cost_basis=self.initial_capital,
            target_price=None,
            stop_loss=None,
            signal_type="buy",
            exit_timestamp=prices.index[-1],
            exit_price=prices.iloc[-1],
            status=PositionStatus.CLOSED_END,
            pnl=final_value - self.initial_capital,
        )
        
        return SimulationResult(
            initial_capital=self.initial_capital,
            final_capital=final_value,
            total_return_pct=total_return,
            total_trades=1,
            winning_trades=1 if total_return > 0 else 0,
            losing_trades=0 if total_return > 0 else 1,
            win_rate=100.0 if total_return > 0 else 0.0,
            wallet_history=wallet_history,
            positions=[position],
            max_drawdown_pct=self._calculate_max_drawdown(wallet_history),
            avg_position_size=self.initial_capital,
            avg_days_held=(prices.index[-1] - prices.index[0]).days,
            avg_win_pct=total_return if total_return > 0 else 0.0,
            avg_loss_pct=total_return if total_return <= 0 else 0.0,
            profit_factor=float('inf') if total_return > 0 else 0.0,
            expectancy_pct=total_return,
        )
    
    def simulate_strategy(
        self,
        prices: pd.Series,
        signals: List,  # List of Signal objects
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> SimulationResult:
        """
        Simulate a trading strategy with proper capital management.
        
        Args:
            prices: Price series with datetime index
            signals: List of trading signals (must have timestamp, price, 
                    signal_type, target_price, stop_loss attributes)
            start_date: Simulation start date
            end_date: Simulation end date
            
        Returns:
            SimulationResult with complete simulation data
        """
        if start_date is None:
            start_date = prices.index.min()
        if end_date is None:
            end_date = prices.index.max()
        
        # Filter prices and signals to date range
        prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]
        signals = [s for s in signals if start_date <= s.timestamp <= end_date]
        
        if len(prices) < 2:
            return self._empty_result()
        
        # Sort signals by timestamp
        signals = sorted(signals, key=lambda s: s.timestamp)
        
        # Initialize wallet
        cash = self.initial_capital
        open_positions: List[Position] = []
        closed_positions: List[Position] = []
        wallet_history: List[WalletState] = []
        
        # Create signal lookup by date for efficient processing
        signal_by_date = {}
        for sig in signals:
            date = sig.timestamp.date() if hasattr(sig.timestamp, 'date') else sig.timestamp
            if date not in signal_by_date:
                signal_by_date[date] = []
            signal_by_date[date].append(sig)
        
        # Simulate day by day
        total_days = len(prices)
        progress_interval = max(1, total_days // 10)  # Show ~10 progress updates
        import time
        sim_start = time.time()
        
        for day_idx, (timestamp, price) in enumerate(prices.items()):
            date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
            
            # 1. Check for position exits (target hit, stop hit, timeout)
            still_open = []
            for pos in open_positions:
                exit_reason = self._check_position_exit(pos, price, timestamp)
                
                if exit_reason:
                    # Close the position
                    pos.exit_timestamp = timestamp
                    pos.exit_price = price
                    pos.status = exit_reason
                    
                    # Calculate P&L
                    if pos.signal_type == "buy":
                        pos.pnl = (price - pos.entry_price) * pos.shares
                    else:  # sell/short
                        pos.pnl = (pos.entry_price - price) * pos.shares
                    
                    # Return capital to cash
                    cash += pos.cost_basis + pos.pnl
                    closed_positions.append(pos)
                else:
                    still_open.append(pos)
            
            open_positions = still_open
            
            # 2. Check for new signals on this date
            if date in signal_by_date:
                for sig in signal_by_date[date]:
                    # Store original signal type for tracking
                    original_signal_type = sig.signal_type
                    
                    # Check if we can open a new position
                    if len(open_positions) >= self.max_positions:
                        continue  # Already at max positions
                    
                    # Check per-instrument position limit
                    instrument = getattr(sig, 'instrument', None)
                    if instrument and self.max_positions_per_instrument is not None:
                        instrument_positions = sum(1 for pos in open_positions if pos.instrument == instrument)
                        if instrument_positions >= self.max_positions_per_instrument:
                            continue  # Already at max positions for this instrument
                    
                    if cash <= 0:
                        continue  # No cash available
                    
                    # Determine position size based on total portfolio value
                    invested_value = sum(pos.shares * price for pos in open_positions)
                    total_portfolio_value = cash + invested_value
                    
                    # Base position size
                    base_size_pct = self.position_size_pct
                    
                    # Get confirmation count from signal (default to 0 if not set)
                    confirmation_count = getattr(sig, 'indicator_confirmations', 0)
                    
                    # Apply position sizing strategy
                    position_size_method = "base"
                    if self.use_confirmation_modulation:
                        # Multiplicative sizing: multiply base by factor based on confirmation count
                        # Factor of 0.0 means skip the trade entirely
                        size_factor = self.confirmation_size_factors.get(confirmation_count, 1.0)
                        if size_factor == 0.0:
                            continue  # Skip this trade
                        adjusted_size_pct = base_size_pct * size_factor
                        # Cap at 100% to prevent overextension
                        adjusted_size_pct = min(adjusted_size_pct, 1.0)
                        position_size_method = "confirmation_modulation"
                    elif self.use_confidence_sizing:
                        # Additive sizing: base + (multiplier * confirmations)
                        # Example: base=0.2, multiplier=0.1, 2 confirmations = 0.2 + (0.1*2) = 0.4 (40%)
                        adjusted_size_pct = base_size_pct + (self.confidence_size_multiplier * confirmation_count)
                        # Cap at 100% to prevent overextension
                        adjusted_size_pct = min(adjusted_size_pct, 1.0)
                        position_size_method = "confidence_sizing"
                    else:
                        adjusted_size_pct = base_size_pct
                    
                    # Apply volatility-based adjustment if enabled
                    if self.use_volatility_sizing:
                        volatility_ratio = self._calculate_volatility_ratio(prices, current_date)
                        if volatility_ratio > self.volatility_threshold:
                            # High volatility - reduce position size
                            adjusted_size_pct *= self.volatility_size_reduction
                            position_size_method += "_vol_adjusted"
                    
                    # Apply flexible sizing if enabled (based on signal quality)
                    if self.use_flexible_sizing:
                        # Get signal confidence (0-1)
                        signal_confidence = getattr(sig, 'confidence', 0.5)  # Default to 0.5 if not set
                        
                        # Calculate risk-reward ratio (will be calculated later, but we need it here)
                        target_price = getattr(sig, 'target_price', None)
                        stop_loss = getattr(sig, 'stop_loss', None)
                        risk_reward_ratio = 0.0
                        if target_price and stop_loss:
                            entry_price = sig.price
                            if original_signal_type == SignalType.BUY:
                                risk_amount = entry_price - stop_loss
                                reward_amount = target_price - entry_price
                            else:  # SELL
                                risk_amount = stop_loss - entry_price
                                reward_amount = entry_price - target_price
                            if risk_amount > 0:
                                risk_reward_ratio = reward_amount / risk_amount
                        
                        # Apply flexible sizing based on method
                        if self.flexible_sizing_method == "confidence":
                            # Scale by confidence: size = base * confidence
                            flexible_factor = signal_confidence
                        elif self.flexible_sizing_method == "risk_reward":
                            # Scale by risk/reward ratio: size = base * min(1.0, rr / target_rr)
                            if risk_reward_ratio > 0:
                                flexible_factor = min(1.0, risk_reward_ratio / self.flexible_sizing_target_rr)
                            else:
                                flexible_factor = 0.5  # Default if no RR calculated
                        else:  # "combined"
                            # Weighted combination: 60% confidence, 40% risk/reward
                            confidence_factor = signal_confidence
                            rr_factor = min(1.0, risk_reward_ratio / self.flexible_sizing_target_rr) if risk_reward_ratio > 0 else 0.5
                            flexible_factor = 0.6 * confidence_factor + 0.4 * rr_factor
                        
                        # Apply flexible factor to adjusted size
                        adjusted_size_pct = adjusted_size_pct * flexible_factor
                        position_size_method += "_flexible"
                    
                    position_capital = total_portfolio_value * adjusted_size_pct
                    
                    # Can't invest more than available cash
                    position_capital = min(position_capital, cash)
                    
                    if position_capital <= 0:
                        continue
                    
                    # Open position
                    entry_price = sig.price
                    shares = position_capital / entry_price
                    
                    # Calculate certainty based on indicator confirmations
                    # Normalize to 0-1 scale (more confirmations = higher certainty)
                    max_possible_confirmations = 3  # RSI, EMA, MACD
                    certainty = min(1.0, getattr(sig, 'indicator_confirmations', 0) / max_possible_confirmations)

                    # Calculate risk amount and risk-reward ratio
                    risk_amount = 0.0
                    risk_reward_ratio = 0.0
                    target_price = getattr(sig, 'target_price', None)
                    stop_loss = getattr(sig, 'stop_loss', None)

                    if target_price and stop_loss:
                        if original_signal_type == SignalType.BUY:
                            risk_amount = entry_price - stop_loss  # Distance to stop
                            reward_amount = target_price - entry_price  # Distance to target
                        else:  # SELL
                            risk_amount = stop_loss - entry_price  # Distance to stop
                            reward_amount = entry_price - target_price  # Distance to target

                        if risk_amount > 0:
                            risk_reward_ratio = reward_amount / risk_amount

                    # Calculate projection (using target price as projection)
                    projection_price = target_price

                    # Determine trend direction from EMA values
                    ema_short = getattr(sig, 'ema_short', None)
                    ema_long = getattr(sig, 'ema_long', None)
                    trend_direction = ""
                    if ema_short is not None and ema_long is not None:
                        trend_direction = "bullish" if ema_short > ema_long else "bearish"

                    pos = Position(
                        entry_timestamp=timestamp,
                        entry_price=entry_price,
                        shares=shares,
                        cost_basis=position_capital,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        signal_type=sig.signal_type.value,  # Store as string for Position dataclass
                        rsi_value=getattr(sig, 'rsi_value', None),
                        ema_short=ema_short,
                        ema_long=ema_long,
                        macd_value=getattr(sig, 'macd_value', None),
                        macd_signal=getattr(sig, 'macd_signal', None),
                        macd_histogram=getattr(sig, 'macd_histogram', None),
                        indicator_confirmations=getattr(sig, 'indicator_confirmations', 0),
                        original_signal_type=original_signal_type.value,  # Track original signal type
                        certainty=certainty,
                        risk_amount=risk_amount,
                        risk_reward_ratio=risk_reward_ratio,
                        projection_price=projection_price,
                        position_size_method=position_size_method,
                        trend_filter_active=getattr(sig, 'trend_filter_active', False),
                        trend_direction=trend_direction,
                        instrument=instrument,  # Store instrument identifier
                    )
                    
                    open_positions.append(pos)
                    cash -= position_capital
            
            # 3. Calculate current wallet state
            invested_value = sum(pos.shares * price for pos in open_positions)
            total_value = cash + invested_value
            return_pct = ((total_value - self.initial_capital) / self.initial_capital) * 100
            
            wallet_history.append(WalletState(
                timestamp=timestamp,
                cash=cash,
                invested_value=invested_value,
                total_value=total_value,
                return_pct=return_pct,
            ))
            
            # Show progress periodically
            if (day_idx + 1) % progress_interval == 0 or day_idx == total_days - 1:
                pct = ((day_idx + 1) / total_days) * 100
                elapsed = time.time() - sim_start
                print(f"    Portfolio simulation: {day_idx + 1}/{total_days} ({pct:.0f}%) - {len(closed_positions)} trades closed - {elapsed:.1f}s", 
                      end='\r', flush=True)
        
        # Clear progress line
        print(" " * 80, end='\r', flush=True)
        
        # Close any remaining open positions at final price
        final_price = prices.iloc[-1]
        final_timestamp = prices.index[-1]
        
        for pos in open_positions:
            pos.exit_timestamp = final_timestamp
            pos.exit_price = final_price
            pos.status = PositionStatus.CLOSED_END
            
            if pos.signal_type == "buy":
                pos.pnl = (final_price - pos.entry_price) * pos.shares
            else:
                pos.pnl = (pos.entry_price - final_price) * pos.shares
            
            cash += pos.cost_basis + pos.pnl
            closed_positions.append(pos)
        
        # Calculate final metrics
        all_positions = closed_positions
        winning = [p for p in all_positions if p.pnl > 0]
        losing = [p for p in all_positions if p.pnl <= 0]
        
        total_trades = len(all_positions)
        winning_trades = len(winning)
        losing_trades = len(losing)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        final_value = wallet_history[-1].total_value if wallet_history else self.initial_capital
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        avg_position_size = (
            sum(p.cost_basis for p in all_positions) / len(all_positions)
            if all_positions else 0.0
        )
        
        days_held = [
            (p.exit_timestamp - p.entry_timestamp).days
            for p in all_positions
            if p.exit_timestamp and p.entry_timestamp
        ]
        avg_days_held = sum(days_held) / len(days_held) if days_held else 0.0
        
        # Calculate average exposure (% of capital invested over time)
        if wallet_history:
            exposures = [
                (state.invested_value / state.total_value * 100) 
                if state.total_value > 0 else 0.0
                for state in wallet_history
            ]
            avg_exposure_pct = sum(exposures) / len(exposures)
        else:
            avg_exposure_pct = 0.0
        
        # Calculate risk/reward metrics (as % of position cost)
        win_pcts = [(p.pnl / p.cost_basis * 100) for p in winning if p.cost_basis > 0]
        loss_pcts = [(p.pnl / p.cost_basis * 100) for p in losing if p.cost_basis > 0]
        
        avg_win_pct = sum(win_pcts) / len(win_pcts) if win_pcts else 0.0
        avg_loss_pct = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.0
        
        total_gains = sum(p.pnl for p in winning)
        total_losses = abs(sum(p.pnl for p in losing))
        profit_factor = (total_gains / total_losses) if total_losses > 0 else (float('inf') if total_gains > 0 else 0.0)
        
        # Expectancy: average expected return per trade
        # = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        win_rate_decimal = win_rate / 100
        expectancy_pct = (win_rate_decimal * avg_win_pct) + ((1 - win_rate_decimal) * avg_loss_pct)
        
        return SimulationResult(
            initial_capital=self.initial_capital,
            final_capital=final_value,
            total_return_pct=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            wallet_history=wallet_history,
            positions=all_positions,
            max_drawdown_pct=self._calculate_max_drawdown(wallet_history),
            avg_position_size=avg_position_size,
            avg_days_held=avg_days_held,
            avg_exposure_pct=avg_exposure_pct,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            profit_factor=profit_factor,
            expectancy_pct=expectancy_pct,
        )
    
    def _check_position_exit(
        self,
        position: Position,
        current_price: float,
        current_timestamp: pd.Timestamp,
    ) -> Optional[PositionStatus]:
        """Check if a position should be exited."""
        # Check max days
        if self.max_days:
            days_held = (current_timestamp - position.entry_timestamp).days
            if days_held >= self.max_days:
                return PositionStatus.CLOSED_TIMEOUT
        
        if position.signal_type == "buy":
            # Check target hit (price >= target)
            if position.target_price and current_price >= position.target_price:
                return PositionStatus.CLOSED_TARGET
            # Check stop loss hit (price <= stop)
            if position.stop_loss and current_price <= position.stop_loss:
                return PositionStatus.CLOSED_STOP
        else:  # sell/short
            # Check target hit (price <= target)
            if position.target_price and current_price <= position.target_price:
                return PositionStatus.CLOSED_TARGET
            # Check stop loss hit (price >= stop)
            if position.stop_loss and current_price >= position.stop_loss:
                return PositionStatus.CLOSED_STOP
        
        return None
    
    def _calculate_max_drawdown(self, wallet_history: List[WalletState]) -> float:
        """Calculate maximum drawdown percentage."""
        if not wallet_history:
            return 0.0
        
        peak = wallet_history[0].total_value
        max_drawdown = 0.0
        
        for state in wallet_history:
            if state.total_value > peak:
                peak = state.total_value
            
            drawdown = ((peak - state.total_value) / peak) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _calculate_volatility_ratio(self, prices: pd.Series, current_date: pd.Timestamp, period: int = 14) -> float:
        """
        Calculate volatility ratio (ATR / current price).
        
        Args:
            prices: Price series
            current_date: Current date for calculation
            period: ATR period (default: 14)
            
        Returns:
            Volatility ratio (ATR/price), or 0.0 if insufficient data
        """
        # Get price data up to current date
        prices_until_now = prices[prices.index <= current_date]
        
        if len(prices_until_now) < period:
            return 0.0  # Not enough data
        
        # Calculate simple ATR using price range
        recent_prices = prices_until_now.tail(period)
        price_range = recent_prices.max() - recent_prices.min()
        atr = price_range / period
        
        # Current price
        current_price = prices_until_now.iloc[-1]
        
        if current_price == 0:
            return 0.0
        
        # Volatility ratio
        volatility_ratio = atr / current_price
        
        return volatility_ratio
    
    def _empty_result(self) -> SimulationResult:
        """Return an empty result for edge cases."""
        return SimulationResult(
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            wallet_history=[],
            positions=[],
            max_drawdown_pct=0.0,
            avg_position_size=0.0,
            avg_days_held=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            profit_factor=0.0,
            expectancy_pct=0.0,
        )
