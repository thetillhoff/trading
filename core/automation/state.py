"""
State manager for tracking placed orders and preventing duplicates.

Persists order state to JSON file for crash recovery.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    CLOSED = "closed"  # Position closed (stop or target hit)


@dataclass
class OrderRecord:
    """Record of a placed order."""
    date: str  # Date order was placed (YYYY-MM-DD)
    instrument: str
    parent_order_id: int
    stop_order_id: int
    target_order_id: int
    entry_price: float
    stop_loss: float
    target_price: float
    quantity: int
    status: str
    certainty: float
    timestamp: str  # ISO format timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderRecord":
        """Create from dictionary."""
        return cls(**data)


class StateManager:
    """
    Manages state of placed orders.
    
    Responsibilities:
    - Track which instruments have been traded on which days
    - Prevent duplicate orders for same instrument/day
    - Persist state to JSON file
    - Clean up old records
    """
    
    def __init__(self, state_file: Path):
        """
        Initialize state manager.
        
        Args:
            state_file: Path to JSON file for state persistence
        """
        self.state_file = Path(state_file)
        self.orders: Dict[str, List[OrderRecord]] = {}  # date -> list of orders
        self._load_state()
    
    def _load_state(self):
        """Load state from JSON file."""
        if not self.state_file.exists():
            logger.info(f"State file {self.state_file} does not exist, starting fresh")
            return
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            # Convert dict to OrderRecord objects
            for date, orders in data.items():
                self.orders[date] = [OrderRecord.from_dict(o) for o in orders]
            
            logger.info(f"Loaded state from {self.state_file}: {len(self.orders)} dates")
        except Exception as e:
            logger.error(f"Failed to load state from {self.state_file}: {e}")
            self.orders = {}
    
    def _save_state(self):
        """Save state to JSON file."""
        try:
            # Ensure parent directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert OrderRecord objects to dicts
            data = {}
            for date, orders in self.orders.items():
                data[date] = [o.to_dict() for o in orders]
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state to {self.state_file}: {e}")
    
    def has_order_for_instrument_today(self, instrument: str, date: Optional[datetime] = None) -> bool:
        """
        Check if we've already placed an order for this instrument today.
        
        Args:
            instrument: Instrument ticker
            date: Date to check (default: today)
            
        Returns:
            True if order exists for this instrument on this date
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        
        if date_str not in self.orders:
            return False
        
        for order in self.orders[date_str]:
            if order.instrument == instrument:
                return True
        
        return False
    
    def get_orders_for_date(self, date: Optional[datetime] = None) -> List[OrderRecord]:
        """
        Get all orders for a specific date.
        
        Args:
            date: Date to query (default: today)
            
        Returns:
            List of OrderRecord objects
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        return self.orders.get(date_str, [])
    
    def add_order(
        self,
        instrument: str,
        parent_order_id: int,
        stop_order_id: int,
        target_order_id: int,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        quantity: int,
        certainty: float,
        status: OrderStatus = OrderStatus.PENDING,
        date: Optional[datetime] = None
    ) -> OrderRecord:
        """
        Add a new order to the state.
        
        Args:
            instrument: Instrument ticker
            parent_order_id: Parent (entry) order ID
            stop_order_id: Stop-loss order ID
            target_order_id: Take-profit order ID
            entry_price: Entry price
            stop_loss: Stop-loss price
            target_price: Take-profit price
            quantity: Number of shares
            certainty: Signal certainty
            status: Order status (default: PENDING)
            date: Date of order (default: today)
            
        Returns:
            Created OrderRecord
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        
        order = OrderRecord(
            date=date_str,
            instrument=instrument,
            parent_order_id=parent_order_id,
            stop_order_id=stop_order_id,
            target_order_id=target_order_id,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            quantity=quantity,
            status=status.value,
            certainty=certainty,
            timestamp=datetime.now().isoformat()
        )
        
        if date_str not in self.orders:
            self.orders[date_str] = []
        
        self.orders[date_str].append(order)
        self._save_state()
        
        logger.info(f"Added order for {instrument} on {date_str}: order_ids=[{parent_order_id}, {stop_order_id}, {target_order_id}]")
        return order
    
    def update_order_status(self, order_id: int, new_status: OrderStatus) -> bool:
        """
        Update status of an order by ID.
        
        Args:
            order_id: Any of the order IDs (parent, stop, or target)
            new_status: New status to set
            
        Returns:
            True if order found and updated, False otherwise
        """
        for date, orders in self.orders.items():
            for order in orders:
                if order_id in [order.parent_order_id, order.stop_order_id, order.target_order_id]:
                    order.status = new_status.value
                    self._save_state()
                    logger.info(f"Updated order {order_id} status to {new_status.value}")
                    return True
        
        logger.warning(f"Order {order_id} not found in state")
        return False
    
    def cleanup_old_records(self, keep_days: int = 30):
        """
        Remove records older than keep_days.
        
        Args:
            keep_days: Number of days to keep (default: 30)
        """
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        
        dates_to_remove = [date for date in self.orders.keys() if date < cutoff_str]
        
        for date in dates_to_remove:
            del self.orders[date]
            logger.info(f"Removed old records for {date}")
        
        if dates_to_remove:
            self._save_state()
    
    def get_all_instruments_with_orders_today(self, date: Optional[datetime] = None) -> List[str]:
        """
        Get list of all instruments that have orders today.
        
        Args:
            date: Date to check (default: today)
            
        Returns:
            List of instrument tickers
        """
        orders = self.get_orders_for_date(date)
        return [order.instrument for order in orders]
