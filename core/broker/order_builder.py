"""
Order builder for converting TradingSignals to IBKR orders.

Handles position sizing, risk management, and order construction.
"""
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..shared.types import TradingSignal, SignalType


logger = logging.getLogger(__name__)


@dataclass
class OrderParameters:
    """Parameters for placing an order."""
    ticker: str
    quantity: int
    entry_price: Optional[float]
    stop_loss: float
    target_price: float
    position_size_usd: float
    signal_certainty: float


class OrderBuilder:
    """
    Builds IBKR orders from TradingSignals.
    
    Handles:
    - Position sizing based on available capital and config
    - Risk validation (min size, max size, account balance)
    - Order parameter extraction from TradingSignal
    """
    
    def __init__(
        self,
        position_size_pct: float = 0.1,
        max_position_size_usd: Optional[float] = None,
        min_position_size_usd: float = 20.0,
        min_account_balance: float = 1000.0
    ):
        """
        Initialize OrderBuilder.
        
        Args:
            position_size_pct: Fraction of available capital per position (0.1 = 10%)
            max_position_size_usd: Hard cap per position in USD (None = no cap)
            min_position_size_usd: Minimum position size in USD (skip if below)
            min_account_balance: Don't trade if account balance below this
        """
        self.position_size_pct = position_size_pct
        self.max_position_size_usd = max_position_size_usd
        self.min_position_size_usd = min_position_size_usd
        self.min_account_balance = min_account_balance
    
    def build_order_from_signal(
        self,
        signal: TradingSignal,
        available_capital: float
    ) -> Optional[OrderParameters]:
        """
        Build order parameters from a TradingSignal.
        
        Args:
            signal: TradingSignal object with entry, stop, target prices
            available_capital: Available cash in account
            
        Returns:
            OrderParameters object, or None if order should be skipped
        """
        # Validate account balance
        if available_capital < self.min_account_balance:
            logger.warning(
                f"Account balance ${available_capital:.2f} below minimum ${self.min_account_balance:.2f}, "
                "skipping order"
            )
            return None
        
        # Validate signal has required fields
        if signal.entry_price is None or signal.stop_loss is None or signal.target_price is None:
            logger.error(
                f"Signal for {signal.instrument} missing required price fields: "
                f"entry={signal.entry_price}, stop={signal.stop_loss}, target={signal.target_price}"
            )
            return None
        
        # Only support LONG signals for now
        if signal.signal_type != SignalType.LONG:
            logger.warning(f"SHORT signals not yet supported, skipping {signal.instrument}")
            return None
        
        # Calculate position size in USD
        position_size_usd = available_capital * self.position_size_pct
        
        # Apply max cap if configured
        if self.max_position_size_usd is not None:
            position_size_usd = min(position_size_usd, self.max_position_size_usd)
        
        # Check minimum size
        if position_size_usd < self.min_position_size_usd:
            logger.warning(
                f"Position size ${position_size_usd:.2f} below minimum ${self.min_position_size_usd:.2f}, "
                f"skipping {signal.instrument}"
            )
            return None
        
        # Calculate quantity (number of shares)
        quantity = int(position_size_usd / signal.entry_price)
        
        if quantity <= 0:
            logger.warning(
                f"Calculated quantity {quantity} for {signal.instrument} is zero or negative, skipping"
            )
            return None
        
        # Build order parameters
        order_params = OrderParameters(
            ticker=signal.instrument,
            quantity=quantity,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            position_size_usd=quantity * signal.entry_price,
            signal_certainty=signal.certainty
        )
        
        logger.info(
            f"Built order for {signal.instrument}: "
            f"qty={quantity}, entry=${signal.entry_price:.2f}, "
            f"stop=${signal.stop_loss:.2f}, target=${signal.target_price:.2f}, "
            f"size=${order_params.position_size_usd:.2f}, certainty={signal.certainty:.2f}"
        )
        
        return order_params
    
    def validate_order_params(self, params: OrderParameters) -> bool:
        """
        Validate order parameters before placement.
        
        Args:
            params: OrderParameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check quantity is positive
        if params.quantity <= 0:
            logger.error(f"Invalid quantity {params.quantity} for {params.ticker}")
            return False
        
        # Check prices are positive
        if params.entry_price is not None and params.entry_price <= 0:
            logger.error(f"Invalid entry price {params.entry_price} for {params.ticker}")
            return False
        
        if params.stop_loss <= 0 or params.target_price <= 0:
            logger.error(
                f"Invalid prices for {params.ticker}: "
                f"stop={params.stop_loss}, target={params.target_price}"
            )
            return False
        
        # Check stop < entry < target for LONG
        if params.entry_price is not None:
            if not (params.stop_loss < params.entry_price < params.target_price):
                logger.error(
                    f"Invalid price ordering for {params.ticker}: "
                    f"stop={params.stop_loss} should be < entry={params.entry_price} < target={params.target_price}"
                )
                return False
        
        # Check position size
        if params.position_size_usd < self.min_position_size_usd:
            logger.error(
                f"Position size ${params.position_size_usd:.2f} below minimum ${self.min_position_size_usd:.2f}"
            )
            return False
        
        if self.max_position_size_usd is not None and params.position_size_usd > self.max_position_size_usd:
            logger.error(
                f"Position size ${params.position_size_usd:.2f} exceeds maximum ${self.max_position_size_usd:.2f}"
            )
            return False
        
        return True
