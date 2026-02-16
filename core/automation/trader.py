"""
Automated trader that analyzes signals and places orders.

Reuses signal detection logic from cli/recommend.py and integrates with broker.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from ..signals.config_loader import load_config_from_yaml
from ..signals.detector import SignalDetector
from ..signals.target_calculator import TargetCalculator
from ..data.loader import DataLoader
from ..shared.types import TradingSignal, SignalType
from ..broker.ibkr_client import IBKRClient
from ..broker.order_builder import OrderBuilder
from .state import StateManager, OrderStatus


logger = logging.getLogger(__name__)


class AutomatedTrader:
    """
    Automated trading orchestrator.
    
    Responsibilities:
    - Load strategy config (baseline.yaml)
    - Analyze signals for all instruments
    - Filter signals based on existing positions and state
    - Place orders via IBKR
    - Update state manager
    """
    
    def __init__(
        self,
        ibkr_client: IBKRClient,
        order_builder: OrderBuilder,
        state_manager: StateManager,
        strategy_config_path: Path,
        lookback_days: int = 365,
        column: str = "Close"
    ):
        """
        Initialize automated trader.
        
        Args:
            ibkr_client: IBKR client instance (must be connected)
            order_builder: Order builder instance
            state_manager: State manager instance
            strategy_config_path: Path to strategy config (e.g., configs/baseline.yaml)
            lookback_days: Days of historical data to load for signal detection
            column: Price column to use (default: Close)
        """
        self.ibkr_client = ibkr_client
        self.order_builder = order_builder
        self.state_manager = state_manager
        self.strategy_config_path = Path(strategy_config_path)
        self.lookback_days = lookback_days
        self.column = column
        
        # Load strategy config
        self.config = load_config_from_yaml(str(self.strategy_config_path))
        logger.info(f"Loaded strategy config from {self.strategy_config_path}")
        
        # Initialize signal detector and target calculator
        self.detector = SignalDetector(self.config)
        self.target_calculator = TargetCalculator(
            risk_reward_ratio=getattr(self.config, 'risk_reward', 3.0),
            use_atr_stops=True,
            atr_stop_multiplier=2.0
        )
    
    def get_latest_signal_for_instrument(
        self,
        instrument: str,
        target_date: Optional[datetime] = None
    ) -> Optional[Tuple[TradingSignal, datetime, str]]:
        """
        Get the most recent signal for an instrument.
        
        This is adapted from cli/recommend.py logic.
        
        Args:
            instrument: Instrument ticker
            target_date: Date to analyze for (default: today)
            
        Returns:
            Tuple of (signal, data_date, warning) or None if no signal/data
            warning is empty string if no issues, otherwise contains warning message
        """
        if target_date is None:
            target_date = datetime.now()
        
        try:
            # Calculate date range
            start_date = target_date - timedelta(days=self.lookback_days)
            
            # Load data up to target_date
            data = DataLoader.from_instrument(
                instrument,
                start_date=str(start_date.date()),
                end_date=str(target_date.date()),
                column=self.column
            )
            
            if data is None or len(data) == 0:
                logger.warning(f"No data available for {instrument}")
                return None
            
            # Get the most recent date in the data
            latest_date = data.index[-1]
            days_old = (target_date.date() - latest_date.date()).days
            
            # Create warning if data is stale
            warning = ""
            if days_old > 3:
                warning = f"Data is {days_old} days old"
                logger.warning(f"{instrument}: {warning}")
            
            # Detect signals
            signals = self.detector.detect_signals(data)
            
            if not signals:
                logger.debug(f"No signals detected for {instrument}")
                return None
            
            # Calculate targets and stop-loss
            signals = self.target_calculator.calculate_targets(signals, data)
            
            # Filter signals to only those on the most recent date
            latest_signals = [s for s in signals if s.date.date() == latest_date.date()]
            
            if not latest_signals:
                logger.debug(f"No signals on latest date for {instrument}")
                return None
            
            # Return best signal (highest certainty)
            best_signal = max(latest_signals, key=lambda s: s.certainty)
            logger.info(
                f"{instrument}: Found signal with certainty {best_signal.certainty:.2f} "
                f"({best_signal.signal_type.value})"
            )
            
            return (best_signal, latest_date, warning)
            
        except Exception as e:
            logger.error(f"Error analyzing {instrument}: {e}")
            return None
    
    def analyze_all_instruments(self) -> List[Tuple[TradingSignal, datetime, str]]:
        """
        Analyze all instruments from strategy config and collect signals.
        
        Returns:
            List of (signal, data_date, warning) tuples
        """
        signals = []
        
        for instrument in self.config.instruments:
            result = self.get_latest_signal_for_instrument(instrument)
            if result is not None:
                signals.append(result)
        
        logger.info(f"Analyzed {len(self.config.instruments)} instruments, found {len(signals)} signals")
        return signals
    
    def select_best_signal(
        self,
        signals: List[Tuple[TradingSignal, datetime, str]]
    ) -> Optional[Tuple[TradingSignal, datetime, str]]:
        """
        Select the best signal to trade from available signals.
        
        Currently selects highest certainty. Could be enhanced with:
        - Risk/reward ratio comparison
        - Sector diversification
        - Recent performance of instrument
        
        Args:
            signals: List of (signal, data_date, warning) tuples
            
        Returns:
            Best signal tuple, or None if no valid signals
        """
        if not signals:
            return None
        
        # Filter out signals with warnings (stale data)
        fresh_signals = [s for s in signals if not s[2]]
        
        if not fresh_signals:
            logger.warning("All signals have data staleness warnings")
            # Fall back to all signals if all have warnings
            fresh_signals = signals
        
        # Select highest certainty
        best = max(fresh_signals, key=lambda s: s[0].certainty)
        logger.info(f"Selected best signal: {best[0].instrument} (certainty={best[0].certainty:.2f})")
        return best
    
    def should_skip_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Check if signal should be skipped based on various filters.
        
        Args:
            signal: Signal to check
            
        Returns:
            Tuple of (should_skip, reason)
        """
        # Check if we already have an order for this instrument today
        if self.state_manager.has_order_for_instrument_today(signal.instrument):
            return True, f"Already have order for {signal.instrument} today"
        
        # Check if we already have an open position for this instrument
        positions = self.ibkr_client.get_open_positions()
        for pos in positions:
            if pos.ticker == signal.instrument:
                return True, f"Already have open position for {signal.instrument}"
        
        # Check max total positions
        if len(positions) >= 3:  # TODO: Load from config
            return True, f"Already at max positions ({len(positions)}/3)"
        
        return False, ""
    
    def analyze_and_trade(self) -> Dict[str, Any]:
        """
        Main trading logic: analyze signals and place order if appropriate.
        
        Returns:
            Dict with status information:
            - status: "placed", "skipped", "no_signal", "error"
            - instrument: ticker if trade placed
            - reason: explanation
            - details: additional info (order IDs, prices, etc.)
        """
        try:
            # 1. Analyze all instruments
            logger.info("Starting signal analysis...")
            signals = self.analyze_all_instruments()
            
            if not signals:
                logger.info("No signals detected across all instruments")
                return {
                    "status": "no_signal",
                    "reason": "No signals detected across all instruments"
                }
            
            # 2. Select best signal
            best = self.select_best_signal(signals)
            if best is None:
                return {
                    "status": "no_signal",
                    "reason": "No valid signals after filtering"
                }
            
            signal, data_date, warning = best
            
            # 3. Check if we should skip this signal
            should_skip, skip_reason = self.should_skip_signal(signal)
            if should_skip:
                logger.info(f"Skipping signal: {skip_reason}")
                return {
                    "status": "skipped",
                    "instrument": signal.instrument,
                    "reason": skip_reason
                }
            
            # 4. Get available capital
            available_capital = self.ibkr_client.get_account_value("AvailableFunds")
            if available_capital is None:
                logger.error("Failed to get available capital from IBKR")
                return {
                    "status": "error",
                    "reason": "Failed to get available capital from IBKR"
                }
            
            logger.info(f"Available capital: ${available_capital:.2f}")
            
            # 5. Build order parameters
            order_params = self.order_builder.build_order_from_signal(signal, available_capital)
            if order_params is None:
                logger.info("Order builder declined to create order (position too small or other constraint)")
                return {
                    "status": "skipped",
                    "instrument": signal.instrument,
                    "reason": "Order builder declined (size/capital constraints)"
                }
            
            # 6. Validate order parameters
            if not self.order_builder.validate_order_params(order_params):
                logger.error("Order parameters failed validation")
                return {
                    "status": "error",
                    "instrument": signal.instrument,
                    "reason": "Order parameters failed validation"
                }
            
            # 7. Place bracket order via IBKR
            logger.info(f"Placing bracket order for {order_params.ticker}...")
            result = self.ibkr_client.place_bracket_order(
                ticker=order_params.ticker,
                quantity=order_params.quantity,
                entry_price=order_params.entry_price,
                stop_loss=order_params.stop_loss,
                target_price=order_params.target_price
            )
            
            if result is None:
                logger.error("Failed to place bracket order")
                return {
                    "status": "error",
                    "instrument": signal.instrument,
                    "reason": "Failed to place bracket order via IBKR"
                }
            
            # 8. Record order in state
            self.state_manager.add_order(
                instrument=order_params.ticker,
                parent_order_id=result.parent_order_id,
                stop_order_id=result.stop_order_id,
                target_order_id=result.target_order_id,
                entry_price=order_params.entry_price,
                stop_loss=order_params.stop_loss,
                target_price=order_params.target_price,
                quantity=order_params.quantity,
                certainty=signal.certainty,
                status=OrderStatus.PENDING
            )
            
            logger.info(
                f"Successfully placed order for {order_params.ticker}: "
                f"order_ids=[{result.parent_order_id}, {result.stop_order_id}, {result.target_order_id}]"
            )
            
            return {
                "status": "placed",
                "instrument": order_params.ticker,
                "reason": "Order placed successfully",
                "details": {
                    "parent_order_id": result.parent_order_id,
                    "stop_order_id": result.stop_order_id,
                    "target_order_id": result.target_order_id,
                    "quantity": order_params.quantity,
                    "entry_price": order_params.entry_price,
                    "stop_loss": order_params.stop_loss,
                    "target_price": order_params.target_price,
                    "position_size_usd": order_params.position_size_usd,
                    "certainty": signal.certainty,
                    "data_warning": warning
                }
            }
            
        except Exception as e:
            logger.exception(f"Error in analyze_and_trade: {e}")
            return {
                "status": "error",
                "reason": f"Exception: {str(e)}"
            }
