#!/usr/bin/env python3
"""
Automated trading service for IBKR.

Long-running service that:
1. Waits for market close + buffer time
2. Downloads latest data
3. Analyzes signals using baseline strategy
4. Places bracket orders via IBKR API
5. Repeats daily

Usage:
    python cli/auto_trade.py [--config CONFIG] [--dry-run]
    
    Or via Docker:
    make auto-trade
"""
import sys
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional
import yaml

from core.automation.scheduler import Scheduler
from core.automation.trader import AutomatedTrader
from core.automation.state import StateManager
from core.broker.ibkr_client import IBKRClient
from core.broker.order_builder import OrderBuilder
from core.data.scraper import download_all


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGTERM, SIGINT)."""
    global shutdown_requested
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


def setup_logging(log_path: Optional[Path] = None, verbose: bool = False):
    """
    Setup logging to stdout and optionally to file.
    
    Args:
        log_path: Path to log file (None = stdout only)
        verbose: If True, use DEBUG level, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_path provided)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def load_ibkr_config(config_path: Path) -> dict:
    """Load IBKR automation config from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_latest_data(logger: logging.Logger):
    """
    Download latest data for all instruments.
    
    Uses the same logic as cli/download.py
    """
    logger.info("Downloading latest market data...")
    try:
        download_all(update_stale=True, quiet=False)
        logger.info("Data download complete")
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        raise


def main():
    """Main service loop."""
    parser = argparse.ArgumentParser(description="IBKR automated trading service")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ibkr_config.yaml",
        help="Path to IBKR config file (default: configs/ibkr_config.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis but don't place orders (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1
    
    config = load_ibkr_config(config_path)
    
    # Setup logging
    log_path = Path(config['automation']['log_path']) if config['automation'].get('log_path') else None
    setup_logging(log_path, args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("IBKR Automated Trading Service Starting")
    logger.info("="*80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Strategy: {config['automation']['strategy_config']}")
    logger.info(f"Account type: {config['broker']['ibkr']['account_type']}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize components
    try:
        # IBKR client
        logger.info("Initializing IBKR client...")
        ibkr_client = IBKRClient()
        
        if not args.dry_run:
            connected = ibkr_client.connect(
                host=config['broker']['ibkr']['host'],
                port=config['broker']['ibkr']['port'],
                client_id=config['broker']['ibkr']['client_id'],
                timeout=config['broker']['ibkr'].get('connection_timeout', 10)
            )
            
            if not connected:
                logger.error("Failed to connect to IBKR")
                return 1
            
            logger.info("Connected to IBKR successfully")
        else:
            logger.info("DRY RUN: Skipping IBKR connection")
        
        # Order builder
        order_builder = OrderBuilder(
            position_size_pct=config['risk']['position_size_pct'],
            max_position_size_usd=config['risk']['max_position_size_usd'],
            min_position_size_usd=config['risk']['min_position_size_usd'],
            min_account_balance=config['risk']['min_account_balance']
        )
        
        # State manager
        state_path = Path(config['automation']['state_path'])
        state_manager = StateManager(state_path)
        
        # Scheduler
        scheduler = Scheduler(
            market_close_hour=config['automation']['market_close_hour'],
            market_close_minute=config['automation']['market_close_minute'],
            wait_after_close_minutes=config['automation']['market_close_wait_minutes'],
            timezone="America/New_York"
        )
        
        # Automated trader
        strategy_config_path = Path(config['automation']['strategy_config'])
        trader = AutomatedTrader(
            ibkr_client=ibkr_client,
            order_builder=order_builder,
            state_manager=state_manager,
            strategy_config_path=strategy_config_path,
            lookback_days=config['automation']['lookback_days'],
            column="Close"
        )
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.exception(f"Failed to initialize components: {e}")
        return 1
    
    # Main loop
    logger.info("Entering main service loop...")
    
    try:
        while not shutdown_requested:
            # 1. Wait for market close + buffer
            logger.info("-" * 80)
            scheduler.wait_for_market_close()
            
            if shutdown_requested:
                break
            
            # 2. Download latest data
            try:
                download_latest_data(logger)
            except Exception as e:
                logger.error(f"Data download failed, skipping this cycle: {e}")
                scheduler.mark_processed()
                scheduler.sleep_until_next_day()
                continue
            
            if shutdown_requested:
                break
            
            # 3. Analyze and trade
            logger.info("Running signal analysis and trading logic...")
            
            if args.dry_run:
                logger.info("DRY RUN: Would analyze and place orders here")
                result = {
                    "status": "dry_run",
                    "reason": "Dry run mode enabled"
                }
            else:
                result = trader.analyze_and_trade()
            
            # 4. Log result
            logger.info("=" * 80)
            logger.info(f"Trading cycle result: {result['status']}")
            logger.info(f"Reason: {result['reason']}")
            
            if result.get('instrument'):
                logger.info(f"Instrument: {result['instrument']}")
            
            if result.get('details'):
                details = result['details']
                logger.info("Order details:")
                for key, value in details.items():
                    logger.info(f"  {key}: {value}")
            
            logger.info("=" * 80)
            
            # 5. Cleanup old state
            state_manager.cleanup_old_records(
                keep_days=config['automation'].get('cleanup_state_days', 30)
            )
            
            # 6. Mark processed
            scheduler.mark_processed()
            
            # 7. Sleep until next day
            if not shutdown_requested:
                scheduler.sleep_until_next_day()
        
    except Exception as e:
        logger.exception(f"Fatal error in main loop: {e}")
        return 1
    
    finally:
        # Cleanup
        logger.info("Shutting down...")
        if not args.dry_run and ibkr_client.is_connected():
            ibkr_client.disconnect()
        logger.info("Service stopped")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
