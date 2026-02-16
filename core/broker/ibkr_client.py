"""
IBKR API client for connection management and order placement.

Uses ib_insync library to communicate with TWS/IB Gateway.
"""
import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime

from ib_insync import IB, Stock, Order, MarketOrder, LimitOrder, BracketOrder, Trade
from ib_insync import util


logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Current position information from IBKR."""
    ticker: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float


@dataclass
class BracketOrderResult:
    """Result of placing a bracket order."""
    parent_order_id: int
    stop_order_id: int
    target_order_id: int
    parent_trade: Trade
    stop_trade: Trade
    target_trade: Trade


class IBKRClient:
    """
    Client for Interactive Brokers API.
    
    Manages connection to TWS/IB Gateway and provides methods for:
    - Placing bracket orders (entry + stop-loss + target)
    - Querying account positions and balances
    - Checking market status
    """
    
    def __init__(self):
        """Initialize IBKR client (not connected)."""
        self.ib = IB()
        self._connected = False
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._client_id: Optional[int] = None
        
    def connect(
        self, 
        host: str = "127.0.0.1", 
        port: int = 7497, 
        client_id: int = 1,
        timeout: int = 10
    ) -> bool:
        """
        Connect to TWS or IB Gateway.
        
        Args:
            host: TWS/Gateway host (default: 127.0.0.1)
            port: TWS/Gateway port (7497=paper, 7496=live for TWS; 4002=paper, 4001=live for Gateway)
            client_id: Unique client ID (1-32)
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            self.ib.connect(host, port, clientId=client_id, timeout=timeout)
            self._connected = True
            self._host = host
            self._port = port
            self._client_id = client_id
            logger.info(f"Connected to IBKR at {host}:{port} (client_id={client_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Disconnect from TWS/IB Gateway."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")
    
    def is_connected(self) -> bool:
        """Check if currently connected to IBKR."""
        return self._connected and self.ib.isConnected()
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect using previous connection parameters.
        
        Returns:
            True if reconnected successfully, False otherwise
        """
        if not self._host or not self._port or not self._client_id:
            logger.error("Cannot reconnect: no previous connection parameters")
            return False
        
        logger.info("Attempting to reconnect to IBKR...")
        self.disconnect()
        return self.connect(self._host, self._port, self._client_id)
    
    def get_account_value(self, field: str = "AvailableFunds") -> Optional[float]:
        """
        Get account value for a specific field.
        
        Args:
            field: Account field to retrieve (e.g., "AvailableFunds", "NetLiquidation", "BuyingPower")
            
        Returns:
            Account value as float, or None if not available
        """
        if not self.is_connected():
            logger.error("Not connected to IBKR")
            return None
        
        try:
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == field and av.currency == "USD":
                    return float(av.value)
            
            logger.warning(f"Account field '{field}' not found")
            return None
        except Exception as e:
            logger.error(f"Failed to get account value: {e}")
            return None
    
    def get_open_positions(self) -> List[Position]:
        """
        Get all currently open positions.
        
        Returns:
            List of Position objects
        """
        if not self.is_connected():
            logger.error("Not connected to IBKR")
            return []
        
        try:
            positions = self.ib.positions()
            result = []
            
            for pos in positions:
                if pos.position != 0:  # Only active positions
                    ticker = pos.contract.symbol
                    result.append(Position(
                        ticker=ticker,
                        quantity=pos.position,
                        avg_cost=pos.avgCost,
                        market_value=pos.position * pos.avgCost,  # Approximate
                        unrealized_pnl=0.0  # Would need current price to calculate
                    ))
            
            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def is_market_open(self) -> bool:
        """
        Check if US stock market is currently open.
        
        Note: This is a simple time-based check. For production, consider using
        IBKR's market data or a market calendar API.
        
        Returns:
            True if market is open, False otherwise
        """
        try:
            # Request market data for a common stock to check if market is open
            # This is a placeholder - in production you'd want a more robust check
            contract = Stock("SPY", "SMART", "USD")
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data
            
            # If we get valid data, market is likely open
            is_open = ticker.last > 0 or ticker.bid > 0 or ticker.ask > 0
            
            self.ib.cancelMktData(contract)
            return is_open
        except Exception as e:
            logger.warning(f"Failed to check market status: {e}")
            return False
    
    def place_bracket_order(
        self,
        ticker: str,
        quantity: int,
        entry_price: Optional[float] = None,
        stop_loss: float = None,
        target_price: float = None,
        exchange: str = "SMART"
    ) -> Optional[BracketOrderResult]:
        """
        Place a bracket order (entry + stop-loss + target profit).
        
        Args:
            ticker: Stock ticker symbol
            quantity: Number of shares (positive for long)
            entry_price: Limit price for entry (None = market order)
            stop_loss: Stop-loss price (required)
            target_price: Take-profit price (required)
            exchange: Exchange routing (default: SMART)
            
        Returns:
            BracketOrderResult with order IDs and Trade objects, or None if failed
        """
        if not self.is_connected():
            logger.error("Not connected to IBKR")
            return None
        
        if stop_loss is None or target_price is None:
            logger.error("stop_loss and target_price are required for bracket orders")
            return None
        
        if quantity <= 0:
            logger.error("quantity must be positive")
            return None
        
        try:
            # Create contract
            contract = Stock(ticker, exchange, "USD")
            self.ib.qualifyContracts(contract)
            
            # Create bracket order
            if entry_price is None:
                # Market order entry
                parent = MarketOrder("BUY", quantity)
            else:
                # Limit order entry
                parent = LimitOrder("BUY", quantity, entry_price)
            
            # Create stop and target orders
            bracket = self.ib.bracketOrder(
                "BUY",
                quantity,
                limitPrice=entry_price if entry_price else 0,
                takeProfitPrice=target_price,
                stopLossPrice=stop_loss
            )
            
            # Place all orders
            parent_trade = self.ib.placeOrder(contract, bracket[0])
            stop_trade = self.ib.placeOrder(contract, bracket[1])
            target_trade = self.ib.placeOrder(contract, bracket[2])
            
            # Wait for orders to be acknowledged
            self.ib.sleep(1)
            
            result = BracketOrderResult(
                parent_order_id=parent_trade.order.orderId,
                stop_order_id=stop_trade.order.orderId,
                target_order_id=target_trade.order.orderId,
                parent_trade=parent_trade,
                stop_trade=stop_trade,
                target_trade=target_trade
            )
            
            logger.info(
                f"Placed bracket order for {ticker}: "
                f"entry={entry_price or 'market'}, stop={stop_loss}, target={target_price}, "
                f"qty={quantity}, orders=[{result.parent_order_id}, {result.stop_order_id}, {result.target_order_id}]"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to place bracket order for {ticker}: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order by ID.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        if not self.is_connected():
            logger.error("Not connected to IBKR")
            return False
        
        try:
            # Find the order
            trades = self.ib.trades()
            for trade in trades:
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancelled order {order_id}")
                    return True
            
            logger.warning(f"Order {order_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: int) -> Optional[Dict]:
        """
        Get status of an order by ID.
        
        Args:
            order_id: Order ID to query
            
        Returns:
            Dict with order status info, or None if not found
        """
        if not self.is_connected():
            logger.error("Not connected to IBKR")
            return None
        
        try:
            trades = self.ib.trades()
            for trade in trades:
                if trade.order.orderId == order_id:
                    return {
                        "order_id": order_id,
                        "status": trade.orderStatus.status,
                        "filled": trade.orderStatus.filled,
                        "remaining": trade.orderStatus.remaining,
                        "avg_fill_price": trade.orderStatus.avgFillPrice,
                    }
            
            logger.warning(f"Order {order_id} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None
