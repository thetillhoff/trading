#!/usr/bin/env python3
"""
Data download CLI.

Downloads historical data for named instruments and updates ticker data.
"""
import argparse
import sys
from pathlib import Path

# Add core to path
core_dir = Path(__file__).parent.parent / "core"
sys.path.insert(0, str(core_dir.parent))

from core.data.scraper import download_instrument, download_all, list_instruments, get_available_instruments
from core.data.download import download_ticker
from core.data.loader import list_available_tickers


def update_all_tickers(quiet: bool = False) -> int:
    """
    Update all tickers in data/tickers/ directory.
    
    Returns:
        Number of tickers successfully updated
    """
    tickers = list_available_tickers()
    if not tickers:
        if not quiet:
            print("No tickers found in data/tickers/ to update")
        return 0
    
    if not quiet:
        print(f"\nUpdating {len(tickers)} tickers...")
    
    updated = 0
    failed = []
    
    for ticker in tickers:
        try:
            df, used_cache = download_ticker(
                ticker,
                force_refresh=False,
                update_stale=True,  # Only fetch new data if stale
                quiet=True
            )
            if df is not None:
                updated += 1
                if not quiet and not used_cache:
                    print(f"  ✓ {ticker}")
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
            if not quiet:
                print(f"  ✗ {ticker}: {e}")
    
    if not quiet:
        print(f"Updated {updated}/{len(tickers)} tickers")
        if failed:
            print(f"Failed: {', '.join(failed[:5])}{' ...' if len(failed) > 5 else ''}")
    
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data for trading instruments and update tickers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available instruments
    python -m cli.download --list

    # Download specific instrument
    python -m cli.download djia

    # Download multiple instruments
    python -m cli.download djia sp500 gold

    # Download all instruments and update tickers
    python -m cli.download

    # Force refresh (re-download)
    python -m cli.download --refresh djia
    
    # Only update tickers (skip named instruments)
    python -m cli.download --tickers-only
        """
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
    parser.add_argument(
        "--tickers-only",
        action="store_true",
        help="Only update tickers (skip named instruments)",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_instruments()
        return 0
    
    # Download named instruments (unless tickers-only)
    if not args.tickers_only:
        if args.instruments:
            # Download specific instruments
            for name in args.instruments:
                result = download_instrument(
                    name,
                    force_refresh=args.refresh,
                    start_date=args.start_date
                )
                if result is None:
                    print(f"Failed to download {name}")
                    return 1
        else:
            # Download all instruments
            results = download_all(
                force_refresh=args.refresh,
                start_date=args.start_date
            )
            if not results:
                print("Failed to download any instruments")
                return 1
    
    # Always update tickers (unless specific instruments were requested)
    if not args.instruments or args.tickers_only:
        updated = update_all_tickers(quiet=False)
        if updated == 0 and not args.tickers_only:
            print("\nNote: No tickers found to update. Tickers are created by asset-analysis or evaluation runs.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
