#!/usr/bin/env python3
"""Download historical data for multiple instruments from Yahoo Finance."""

import os
import argparse
import warnings
import pandas as pd
import yfinance as yf
from typing import Dict, Optional
from pathlib import Path

# Suppress yfinance's pandas deprecation warnings (will be fixed in future yfinance version)
warnings.filterwarnings('ignore', message='.*Timestamp.utcnow.*')

# Get the directory where this module is located
MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR.parent.parent / "data"  # Store data in project root /data

# Instrument definitions: name -> (yahoo_ticker, description)
INSTRUMENTS: Dict[str, tuple] = {
    # Major Indices - ETFs
    "sp500": ("^GSPC", "S&P 500 Index - US large cap"),
    "nasdaq": ("QQQ", "Invesco QQQ Trust - NASDAQ-100 ETF"),
    "dax": ("^GDAXI", "DAX 40 Index - German blue chips"),
    "djia": ("^DJI", "Dow Jones Industrial Average - US blue chips"),
    "emerging_markets": ("EEM", "iShares MSCI Emerging Markets ETF"),
    "small_cap": ("IWM", "iShares Russell 2000 ETF - US small cap"),
    "msci_world": ("URTH", "iShares MSCI World ETF - Global market benchmark"),
    
    # Technology Stocks
    "tech_sector": ("XLK", "Technology Select Sector SPDR ETF"),
    "apple": ("AAPL", "Apple Inc."),
    "microsoft": ("MSFT", "Microsoft Corporation"),
    "amazon": ("AMZN", "Amazon.com Inc."),
    "netflix": ("NFLX", "Netflix Inc."),
    
    # Clean Energy & Renewables
    "clean_energy": ("ICLN", "iShares Global Clean Energy ETF"),
    "solar_energy": ("TAN", "Invesco Solar ETF"),
    "clean_tech": ("QCLN", "First Trust NASDAQ Clean Edge Green Energy ETF"),
    "tesla": ("TSLA", "Tesla Inc. - Electric Vehicles"),
    "nextera": ("NEE", "NextEra Energy - Renewable Utilities"),
    
    # ESG
    "esg_us": ("ESGV", "Vanguard ESG U.S. Stock ETF"),
    
    # Precious Metals (Physical-Backed)
    "gold_physical": ("GLD", "SPDR Gold Shares - Physically backed"),
    "silver_physical": ("SLV", "iShares Silver Trust - Physically backed"),
    
    # Currencies
    "eurusd": ("EURUSD=X", "EUR/USD Exchange Rate"),
}


def download_instrument(
    name: str,
    force_refresh: bool = False,
    start_date: str = "1990-01-01",
) -> Optional[pd.DataFrame]:
    """
    Download data for a single instrument with smart caching.
    
    Features:
    - Incremental updates: Only downloads missing data since last cache
    - Smart validation: Checks if cached data covers requested range
    - Automatic refresh: Updates stale data (older than 1 day)

    Args:
        name: Instrument name (key in INSTRUMENTS dict)
        force_refresh: If True, re-download all data from scratch
        start_date: Start date for historical data

    Returns:
        DataFrame with OHLCV data, or None if download failed
    """
    if name not in INSTRUMENTS:
        print(f"Error: Unknown instrument '{name}'")
        print(f"Available instruments: {list(INSTRUMENTS.keys())}")
        return None

    ticker, description = INSTRUMENTS[name]

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = DATA_DIR / f"{name}.csv"

    # Current time for freshness checks
    now = pd.Timestamp.now()
    yesterday = now - pd.Timedelta(days=1)
    requested_start = pd.Timestamp(start_date)

    # Check if cached data exists and is usable
    if csv_file.exists() and not force_refresh:
        try:
            df_cached = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            if df_cached.empty:
                print(f"  Cached file is empty, re-downloading...")
            else:
                cached_start = df_cached.index.min()
                cached_end = df_cached.index.max()
                
                # Check if cache covers requested range and is fresh
                covers_start = cached_start <= requested_start
                is_fresh = cached_end >= yesterday
                
                if covers_start and is_fresh:
                    # Cache is complete and up-to-date
                    print(f"Loading {name} from cache: {csv_file}")
                    print(f"  Cached: {len(df_cached)} rows ({cached_start.date()} to {cached_end.date()})")
                    return df_cached
                
                # Cache exists but needs update
                if not is_fresh:
                    # Incremental update: download only missing days
                    print(f"Updating {name} cache (last: {cached_end.date()})...")
                    update_start = cached_end + pd.Timedelta(days=1)
                    
                    try:
                        df_new = yf.download(ticker, start=update_start.strftime('%Y-%m-%d'), progress=False)
                        
                        if not df_new.empty:
                            # Flatten multi-level columns if present
                            if isinstance(df_new.columns, pd.MultiIndex):
                                df_new.columns = df_new.columns.get_level_values(0)
                            
                            # Merge cached and new data
                            df = pd.concat([df_cached, df_new]).sort_index()
                            df = df[~df.index.duplicated(keep='last')]  # Remove duplicates, keep latest
                            
                            # Save updated cache
                            df.to_csv(csv_file)
                            print(f"  Updated: +{len(df_new)} new rows")
                            print(f"  Total: {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")
                            return df
                        else:
                            # No new data available (weekend/holiday), use cache
                            print(f"  No new data available (market closed)")
                            return df_cached
                    
                    except Exception as e:
                        print(f"  Error updating cache: {e}")
                        print(f"  Using cached data...")
                        return df_cached
                
                elif not covers_start:
                    # Cache doesn't cover requested start date, need full re-download
                    print(f"  Cache start ({cached_start.date()}) > requested ({requested_start.date()})")
                    print(f"  Re-downloading from {start_date}...")
                    # Fall through to full download below
        
        except Exception as e:
            print(f"  Error reading cache: {e}")
            print(f"  Re-downloading...")
            # Fall through to full download below

    # Full download (no cache or force_refresh or cache error)
    print(f"Downloading {name} ({ticker}): {description}")
    try:
        df = yf.download(ticker, start=start_date, progress=False)

        if df.empty:
            print(f"  Warning: No data returned for {name}")
            return None

        # Flatten multi-level columns if present (yfinance sometimes returns these)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Save to CSV
        df.to_csv(csv_file)
        print(f"  Saved {len(df)} rows to {csv_file}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

        return df

    except Exception as e:
        print(f"  Error downloading {name}: {e}")
        return None


def download_all(force_refresh: bool = False, start_date: str = "1990-01-01") -> Dict[str, pd.DataFrame]:
    """
    Download data for all instruments.

    Args:
        force_refresh: If True, re-download all data
        start_date: Start date for historical data

    Returns:
        Dictionary mapping instrument names to DataFrames
    """
    results = {}

    print("=" * 60)
    print("Downloading all instruments")
    print("=" * 60)

    for name in INSTRUMENTS:
        df = download_instrument(name, force_refresh=force_refresh, start_date=start_date)
        if df is not None:
            results[name] = df
        print()

    print("=" * 60)
    print(f"Successfully downloaded {len(results)}/{len(INSTRUMENTS)} instruments")
    print("=" * 60)

    return results


def list_instruments() -> None:
    """Print available instruments."""
    print("\nAvailable instruments:")
    print("-" * 60)
    print(f"{'Name':<12} {'Ticker':<12} {'Description'}")
    print("-" * 60)
    for name, (ticker, desc) in INSTRUMENTS.items():
        print(f"{name:<12} {ticker:<12} {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data for trading instruments"
    )
    parser.add_argument(
        "instruments",
        nargs="*",
        help="Instrument names to download (default: all)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available instruments",
    )
    parser.add_argument(
        "--refresh", "-r",
        action="store_true",
        help="Force refresh (re-download even if cached)",
    )
    parser.add_argument(
        "--start-date", "-s",
        default="1990-01-01",
        help="Start date for historical data (default: 1990-01-01)",
    )

    args = parser.parse_args()

    if args.list:
        list_instruments()
        return

    if args.instruments:
        # Download specific instruments
        for name in args.instruments:
            download_instrument(name, force_refresh=args.refresh, start_date=args.start_date)
    else:
        # Download all instruments
        download_all(force_refresh=args.refresh, start_date=args.start_date)


if __name__ == "__main__":
    main()